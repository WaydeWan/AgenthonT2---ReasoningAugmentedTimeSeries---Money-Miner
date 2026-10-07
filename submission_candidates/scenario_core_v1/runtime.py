"""A deliberately narrow, auditable SEP-to-UST2Y research hypothesis.

No outcomes, learned artifacts, requests, model clients, or card titles enter.
Native yield inputs and SEP rates are percentage points (5.25 means 5.25%).
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math

import numpy as np
import pandas as pd

from submission_candidates.text_core.evidence import (
    EvidenceError, date_value, parse_sep_tables, policy_target_features,
)
from .settings import SETTINGS, LIMITATIONS, VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def _day(value):
    return pd.Timestamp(date_value(value))


def integrate_curve(knots, start, end):
    """Exact integral of the declared linear interpolant / actual calendar days.

    Reject every extrapolation, including a single unsupported day. Leap days
    are naturally included. Knots are dated values, not equally spaced steps.
    """
    times = [_day(item['date']) for item in knots]
    values = np.asarray([item['value'] for item in knots], dtype=float)
    begin, finish = _day(start), _day(end)
    if len(times) < 2 or not np.isfinite(values).all():
        raise EvidenceError('insufficient_curve')
    if any(a >= b for a, b in zip(times, times[1:])):
        raise EvidenceError('unordered_curve')
    if begin >= finish or begin < times[0] or finish > times[-1]:
        raise EvidenceError('unsupported_integration_window')
    origin = times[0]
    axis = np.asarray([(t - origin).days for t in times], dtype=float)
    breaks = [begin] + [t for t in times if begin < t < finish] + [finish]
    x = np.asarray([(t - origin).days for t in breaks], dtype=float)
    rates = np.interp(x, axis, values)
    integral = float(np.sum(np.diff(x) * (rates[:-1] + rates[1:]) / 2))
    return {'start': str(begin.date()), 'end': str(finish.date()),
            'calendar_days': (finish - begin).days, 'integral_percent_days': integral,
            'average_percent': integral / (finish - begin).days}


def _policy_path(documents, asof):
    cutoff = _day(asof)
    parsed = []
    identities = set()
    for doc in documents:
        if doc['doc_id'] in identities:
            raise EvidenceError('duplicate_document_identity')
        identities.add(doc['doc_id'])
        if _day(doc['published_at']) > cutoff:
            raise EvidenceError('future_document')
        facts, dots = parse_sep_tables(doc)
        chosen = {}
        for fact in facts:
            if (fact['variable'] != 'policy_rate' or fact['statistic'] != 'median'
                    or fact['vintage'] != 'current' or not fact['period'].isdigit()):
                continue
            if fact['lower'] != fact['upper']:
                raise EvidenceError('non_scalar_median')
            year = int(fact['period'])
            entry = {'year': year, 'value': fact['lower'], 'method': 'literal_median',
                     'citation': fact}
            if year in chosen and chosen[year]['value'] != entry['value']:
                raise EvidenceError('conflicting_medians_in_document')
            chosen[year] = entry
        for dot in dots:
            if not dot['period'].isdigit():
                continue
            year = int(dot['period'])
            if (year in chosen and chosen[year]['method'] == 'median_of_supplied_dot_counts'
                    and chosen[year]['value'] != dot['median']):
                raise EvidenceError('conflicting_dot_medians_in_document')
            if year not in chosen:
                chosen[year] = {'year': year, 'value': dot['median'],
                    'method': 'median_of_supplied_dot_counts', 'citation': dot}
        # A latest publication lacking a policy projection is not a new policy path.
        if chosen:
            parsed.append((doc, chosen))
    if not parsed:
        raise EvidenceError('no_dated_sep_policy_medians')
    latest = max(_day(d['published_at']) for d, _ in parsed)
    candidates = [(d, points) for d, points in parsed if _day(d['published_at']) == latest]
    # Same-day partial or differing publications are ambiguous. Never silently
    # prefer one by title/id or stitch projections across publication vintages.
    signatures = {tuple(sorted((y, p['value']) for y, p in points.items()))
                  for _, points in candidates}
    if len(signatures) != 1:
        raise EvidenceError('conflicting_same_day_sep_paths')
    doc, chosen = sorted(candidates, key=lambda pair: pair[0]['doc_id'])[0]
    years = sorted(y for y in chosen if date(y, 12, 31) > cutoff.date())
    if not years or years != list(range(cutoff.year, max(years) + 1)):
        raise EvidenceError('missing_dated_projection_year')
    policy = policy_target_features(documents, asof)
    if policy['reason']:
        raise EvidenceError(policy['reason'])
    current = policy['facts'][-1]
    knots = [{'date': asof, 'value': policy['features']['policy_current_target_midpoint'],
              'method': 'latest_supplied_current_band_midpoint', 'citation': current}]
    knots += [{'date': f'{y}-12-31', 'value': chosen[y]['value'],
               'method': chosen[y]['method'], 'citation': chosen[y]['citation']} for y in years]
    return {'knots': knots, 'sep_doc_id': doc['doc_id'],
            'sep_published_at': doc['published_at'],
            'same_day_equivalent_sources': [d['doc_id'] for d, _ in candidates],
            'current_policy_age_days': (cutoff - _day(current['published_at'])).days,
            'source_sha256': {d['doc_id']: hashlib.sha256(d['text'].encode()).hexdigest()
                              for d in documents},
            'excluded_undated_longer_run': True}


def _house_guard(guard, documents, asof):
    if guard is None:
        return {'mode': 'literal_sources_only', 'house_used': False}
    if not isinstance(guard, dict) or set(guard) != {'decision', 'citations'}:
        raise EvidenceError('invalid_house_guard_schema')
    if guard['decision'] != 'allow':
        raise EvidenceError('house_abstained')
    citations = guard['citations']
    if not isinstance(citations, list) or not citations:
        raise EvidenceError('house_guard_missing_citations')
    lookup = {d['doc_id']: d for d in documents}
    for item in citations:
        if not isinstance(item, dict) or set(item) != {'doc_id', 'quote'}:
            raise EvidenceError('invalid_house_guard_citation')
        source = lookup.get(item['doc_id'])
        quote = item['quote']
        if (source is None or not isinstance(quote, str) or len(quote.strip()) < 16
                or quote not in source['text'] or _day(source['published_at']) > _day(asof)):
            raise EvidenceError('unsupported_house_guard_citation')
    return {'mode': 'externally_supplied_citation_guard', 'house_used': True,
            'guard_sha256': digest(guard),
            'boundary': 'Exact citation presence verifies grounding only; semantic correctness is not proven.'}


def build_view(*, documents, asof, assets, horizons, anchor,
               target_type='level', frequency='daily', house_guard=None):
    """Build views from cutoff-visible inputs only. Any missing support is typed.

    anchor = {asset, value, observed_at, available_at, unit}; it must originate
    from the cutoff-filtered per-unit panel, in percentage points. Future labels
    are deliberately not accepted by this interface.
    """
    result = {'version': VERSION, 'ready': False, 'reason': None,
              'settings': dict(SETTINGS), 'limitations': list(LIMITATIONS),
              'asof': asof, 'assets': list(assets), 'horizons': list(horizons)}
    try:
        cutoff = _day(asof)
        if str(cutoff.date()) != asof:
            raise EvidenceError('asof_must_be_iso_date')
        if list(assets) != ['UST_2Y']:
            raise EvidenceError('unsupported_asset_grid')
        if target_type != 'level' or frequency != 'daily':
            raise EvidenceError('unsupported_target_clock')
        if (not horizons or len(set(horizons)) != len(horizons)
                or any(type(h) is not int or h < 1 for h in horizons)):
            raise EvidenceError('invalid_horizons')
        if not isinstance(anchor, dict) or set(anchor) != {'asset', 'value', 'observed_at', 'available_at', 'unit'}:
            raise EvidenceError('invalid_anchor_schema')
        if (anchor['asset'] != 'UST_2Y' or anchor['unit'] != 'percentage_points'
                or not math.isfinite(float(anchor['value']))):
            raise EvidenceError('invalid_anchor_value_or_unit')
        observed, available = _day(anchor['observed_at']), _day(anchor['available_at'])
        if observed > cutoff or available > cutoff or available < observed:
            raise EvidenceError('future_or_inconsistent_anchor')
        result['house'] = _house_guard(house_guard, documents, asof)
        path = _policy_path(documents, asof)
        base_end = cutoff + pd.DateOffset(years=SETTINGS['tenor_calendar_years'])
        base = integrate_curve(path['knots'], asof, str(base_end.date()))
        views = []
        for horizon in horizons:
            endpoint = cutoff + pd.offsets.BDay(horizon)
            end = endpoint + pd.DateOffset(years=SETTINGS['tenor_calendar_years'])
            forward = integrate_curve(path['knots'], str(endpoint.date()), str(end.date()))
            shift = forward['average_percent'] - base['average_percent']
            views.append({'asset': 'UST_2Y', 'horizon': horizon,
                          'endpoint': str(endpoint.date()), 'forward_window': forward,
                          'hypothetical_yield_change_pp': shift,
                          'view_yield_pp': float(anchor['value']) + shift})
        result.update(ready=True, path=path, anchor=dict(anchor),
                      anchor_observation_age_days=(cutoff - observed).days,
                      base_window=base, views=views)
    except (EvidenceError, KeyError, TypeError, ValueError, OverflowError) as exc:
        result['reason'] = str(exc) if isinstance(exc, EvidenceError) else 'invalid_input:' + type(exc).__name__
    return result


def _constrained_coupling(weights, rng):
    """Preserve each original row unless a maximal-coupling move is proposed.

    Without acceptance constraints, expected final empirical weights equal q.
    Every accepted source-count change is checked against the actual ESS floor;
    rejected moves are disclosed. Replacement never changes an individual cell.
    """
    n = len(weights)
    indices = np.arange(n, dtype=np.int64)
    counts = np.ones(n, dtype=np.int64)
    scaled = n * weights
    deficit = np.maximum(1 - scaled, 0)
    surplus = np.maximum(scaled - 1, 0)
    proposal_rows = np.flatnonzero(rng.random(n) < deficit)
    rng.shuffle(proposal_rows)
    proposed, rejected, accepted = len(proposal_rows), 0, 0
    if not proposed or surplus.sum() <= 0:
        return indices, counts, {'proposed_rows': proposed, 'rejected_rows': 0}
    recipients = rng.choice(n, size=proposed, p=surplus / surplus.sum())
    square_sum = n
    cap = math.floor(SETTINGS['max_changed_fraction'] * n)
    max_square_sum = n / SETTINGS['minimum_ess_fraction']
    for row, recipient in zip(proposal_rows, recipients):
        if row == recipient:
            continue
        # Each donor is proposed once. Its original source may also have been
        # replicated elsewhere; use actual counts, not an assumed 1 -> 0.
        delta = (-2 * int(counts[row]) + 1) + (2 * int(counts[recipient]) + 1)
        if accepted >= cap or square_sum + delta > max_square_sum + 1e-9:
            rejected += 1
            continue
        indices[row] = recipient
        counts[row] -= 1
        counts[recipient] += 1
        square_sum += delta
        accepted += 1
    return indices, counts, {'proposed_rows': proposed, 'rejected_rows': rejected}


def apply_scenario(cube, view, *, seed=0, control='scenario'):
    """Return (whole-row-selected cube, audit, source_row_indices).

    The same indices select every asset/horizon. beta0 is bitwise identity.
    `shuffled` randomly assigns row scores with a separate fixed RNG namespace.
    Controls are diagnostics, not candidates selected using realized labels.
    """
    baseline = np.asarray(cube)
    if (baseline.ndim != 3 or baseline.shape[0] < 1 or not np.isfinite(baseline).all()
            or baseline.dtype.kind != 'f'):
        raise ValueError('Expected a finite floating [draw,asset,horizon] cube')
    if type(seed) is not int or seed < 0 or control not in ('scenario', 'beta0', 'shuffled'):
        raise ValueError('Invalid fixed control or seed')
    n = len(baseline)
    indices = np.arange(n, dtype=np.int64)
    audit = {'version': VERSION, 'control': control, 'seed': seed,
             'input_sha256': array_digest(baseline), 'view_sha256': digest(view),
             'applied': False, 'changed_rows': 0, 'empirical_ess': float(n),
             'theoretical_ess': float(n), 'reason': None, 'accuracy_evaluated': False}

    def finish(out, reason):
        audit.update(reason=reason, output_sha256=array_digest(out),
                     source_indices_sha256=array_digest(indices),
                     whole_rows_preserved=bool(np.array_equal(out, baseline[indices])))
        if not audit['whole_rows_preserved']:
            raise AssertionError('Joint row invariant violated')
        return out, audit, indices

    if control == 'beta0':
        return finish(baseline.copy(), 'beta0_identity')
    if not view.get('ready'):
        return finish(baseline.copy(), view.get('reason', 'unavailable_view'))
    if view['assets'] != ['UST_2Y'] or baseline.shape[1:] != (1, len(view['horizons'])):
        raise ValueError('Cube/view target grid mismatch')
    targets = np.array([entry['view_yield_pp'] for entry in view['views']], dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError('Nonfinite view')
    iqr = np.subtract(*np.quantile(baseline[:, 0, :], [.75, .25], axis=0))
    if np.any(iqr <= 1e-12):
        return finish(baseline.copy(), 'degenerate_baseline_iqr')
    score = -np.mean(np.minimum(np.abs((baseline[:, 0, :] - targets) / iqr), 2), axis=1) / 2
    if control == 'shuffled':
        score = np.random.default_rng(np.random.SeedSequence([seed, 173])).permutation(score)
    logweight = SETTINGS['beta'] * score
    tilted = np.exp(logweight - np.max(logweight))
    tilted /= np.sum(tilted)
    q = (1 - SETTINGS['mixture_mass']) / n + SETTINGS['mixture_mass'] * tilted
    q /= np.sum(q)
    ess = 1 / np.dot(q, q)
    audit.update(theoretical_ess=float(ess), theoretical_total_variation=float(np.sum(abs(q - 1 / n)) / 2),
                 weights_sha256=array_digest(q), score_sha256=array_digest(score),
                 beta=SETTINGS['beta'], mixture_mass=SETTINGS['mixture_mass'])
    if ess < SETTINGS['minimum_ess_fraction'] * n:
        return finish(baseline.copy(), 'theoretical_ess_below_floor')
    indices, counts, coupling = _constrained_coupling(q, np.random.default_rng(np.random.SeedSequence([seed, 719])))
    output = baseline[indices].copy()
    changed = int(np.sum(indices != np.arange(n)))
    empirical_ess = float(n * n / np.dot(counts.astype(float), counts))
    if changed > math.floor(.10 * n) or empirical_ess + 1e-9 < .98 * n:
        raise AssertionError('Empirical intervention constraint violated')
    audit.update(coupling, applied=bool(changed), changed_rows=changed,
                 empirical_ess=empirical_ess, empirical_total_variation=float(np.sum(abs(counts - 1)) / (2 * n)),
                 empirical_to_theoretical_l1=float(np.sum(abs(counts / n - q))))
    return finish(output, None if changed else 'no_empirical_move')
