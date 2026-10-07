"""Experimental policy fitting from mature labels inside the current unit.

No evaluation result, sibling unit, task ID, title, F label or external data is
an input to selection. Architecture and proposal constants nevertheless have
development-history provenance; local fitting does not erase that provenance.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from time import monotonic

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from qfbench2_track_forecasting import scoring as official

from submission_candidates.numeric_core import runtime as core
from .availability import load_tapes, Unsupported
from submission_candidates.numeric_core.postprocess import rate_sources, rate_combination, tail_sharpen
from . import reference, scales
from .settings import (MODES, FAMILY, INNER_DRAWS, MIN_PREFIX, MAX_ORIGINS,
                       MIN_TRAIN, MIN_CONFIRM, MAX_CELLS, TAILS, settings)

DEFAULT_ADAPTATION_SECONDS = 180.


class AdaptationBudgetExpired(Unsupported):
    """A cooperative stop, never a claim to intercept an OS kill."""


class _Budget:
    def __init__(self, seconds, clock):
        self.seconds, self.clock = seconds, clock
        self.start = float(clock())
        if not math.isfinite(self.start):
            raise ValueError('invalid_adaptation_clock')
        self.deadline = self.start+seconds
        self.last_stage = "start"

    def check(self, stage):
        self.last_stage = stage
        now = float(self.clock())
        if not math.isfinite(now):
            raise ValueError("invalid_adaptation_clock")
        if now >= self.deadline:
            raise AdaptationBudgetExpired("adaptive_budget_exhausted", {"stage": stage})


def _check(budget, stage):
    if budget is not None:
        budget.check(stage)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _base(unit, *, draws, seed):
    options = core.NumericOptions(draws=draws, seed=seed).validate()
    cube, layout, detail = core._base(unit, options)
    if cube.shape != (draws, len(unit.assets), len(unit.horizons)) or not np.isfinite(cube).all():
        raise ValueError('Production base violates finite declared-grid contract')
    return cube, layout, detail


def eligible(unit, layout):
    cells = len(unit.assets)*len(unit.horizons)
    return (layout.frequency == 'daily' and unit.target_type in {'level', 'yield'}
            and all(asset.startswith('UST_') for asset in unit.assets)
            and layout.transforms == ['level']*len(unit.assets) and 1 <= cells <= MAX_CELLS)


def _family(base, unit, layout, seed):
    anchors = [float(unit.histories[a].iloc[-1]) for a in unit.assets]
    soft, location, source_info = rate_sources(base, anchors, seed)
    paired, _ = rate_combination(base, soft, location, 'N1', seed)
    mixture, _ = rate_combination(base, soft, location, 'N2', seed)
    central, _ = tail_sharpen(base, layout.transforms, strength=1.)
    cubes = {'V6': base.copy(), 'paired_SL': paired, 'row_mixture_SL': mixture,
             'location_L': location, 'central_tail': central}
    if tuple(cubes) != FAMILY or any(c.shape != base.shape or not np.isfinite(c).all() for c in cubes.values()):
        raise ValueError('Invalid proposal family; do not discard an unsuccessful proposal')
    return cubes, source_info


def _seed(histories, assets, horizons, origin):
    hasher = hashlib.sha256(json.dumps({'assets': list(assets), 'horizons': list(horizons),
        'origin': str(pd.Timestamp(origin).date()), 'method': 'adaptive-numerical-v1'}, sort_keys=True).encode())
    for asset in assets:
        source = histories[asset]
        hasher.update(asset.encode())
        hasher.update(np.asarray(source.index.asi8, dtype='<i8').tobytes())
        hasher.update(np.asarray(source.to_numpy(), dtype='<f8').tobytes())
    digest = hasher.hexdigest()
    return int.from_bytes(bytes.fromhex(digest)[:8], 'little') & 0x7fffffff, digest


def _snapshots(tapes, cutoff):
    return {tape.asset: tape.snapshot(cutoff) for tape in tapes}


def _prefix(tapes, origin):
    return {tape.asset: tape.snapshot(origin).set_index('date').value.sort_index() for tape in tapes}


def _label_metadata(snapshots, endpoints, assets, end_exclusive):
    """Inspect only row dates/availability; no outcome values are extracted."""
    availability = {}
    for asset in assets:
        frame = snapshots[asset]
        times = []
        for endpoint in endpoints:
            row = frame.loc[frame.date == endpoint, ['date', 'available_at']]
            if len(row) != 1:
                return None
            available = max(pd.Timestamp(endpoint), pd.Timestamp(row.available_at.iloc[0]))
            if available >= end_exclusive:
                return None
            times.append(available.isoformat())
        availability[asset] = times
    return availability


def build_roster(unit, tapes, *, budget=None):
    """Select at most24 origins from timestamp eligibility, without label values."""
    _check(budget, 'roster_start')
    outer = _snapshots(tapes, unit.asof)
    if any(frame.empty for frame in outer.values()):
        raise Unsupported('empty_outer_snapshot')
    start = max(frame.date.min() for frame in outer.values())
    end = pd.Timestamp(unit.asof)+pd.Timedelta(days=1)
    roster = []
    exclusions = {'immature_or_missing_endpoint': 0, 'missing_origin': 0, 'short_prefix': 0}
    for origin in pd.date_range(start, unit.asof, freq=pd.offsets.BMonthEnd()):
        _check(budget, 'roster_origin')
        endpoints = [origin+pd.offsets.BDay(h) for h in unit.horizons]
        meta = _label_metadata(outer, endpoints, unit.assets, end)
        if meta is None:
            exclusions['immature_or_missing_endpoint'] += 1
            continue
        frames = _snapshots(tapes, origin)
        if any(origin not in set(frame.date) for frame in frames.values()):
            exclusions['missing_origin'] += 1
            continue
        if any(len(frame) < MIN_PREFIX for frame in frames.values()):
            exclusions['short_prefix'] += 1
            continue
        roster.append({'origin': str(origin.date()), 'endpoints': [str(x.date()) for x in endpoints],
                       'label_available_at': meta, 'label_vintage_cutoff': unit.asof})
    exclusions['older_than_last24'] = max(0, len(roster)-MAX_ORIGINS)
    _check(budget, 'roster_complete')
    return roster[-MAX_ORIGINS:], exclusions


def split_roster(unit, tapes, roster, *, budget=None):
    _check(budget, 'partition_start')
    if len(roster) < MIN_TRAIN+MIN_CONFIRM:
        raise Unsupported('fewer_than_12_mature_origins', {'eligible_origins': len(roster)})
    count = max(MIN_CONFIRM, len(roster)//4)
    confirm = [deepcopy(r) for r in roster[-count:]]
    boundary = pd.Timestamp(confirm[0]['origin'])
    selection_cutoff = str((boundary-pd.Timedelta(days=1)).date())
    known = _snapshots(tapes, selection_cutoff)
    train, purged = [], []
    for row in roster[:-count]:
        _check(budget, 'partition_origin')
        endpoints = [pd.Timestamp(x) for x in row['endpoints']]
        meta = _label_metadata(known, endpoints, unit.assets, boundary)
        if meta is None:
            purged.append(row['origin'])
            continue
        train.append({**deepcopy(row), 'label_available_at': meta,
                      'label_vintage_cutoff': selection_cutoff})
    if len(train) < MIN_TRAIN:
        raise Unsupported('fewer_than_8_training_origins_after_maturity_purge',
                          {'train_count': len(train), 'confirmation_count': len(confirm),
                           'purged_origins': purged, 'training_labels_asof': selection_cutoff})
    return train, confirm, {'training_labels_asof': selection_cutoff,
        'first_confirmation_origin': confirm[0]['origin'], 'purged_origins': purged,
        'training_count': len(train), 'confirmation_count': len(confirm)}


def _inner(unit, origin, histories):
    targets = {'asset_ids': list(unit.assets), 'horizons': list(unit.horizons),
               'target_type': unit.target_type, 'target_frequency': 'daily',
               'value_unit': unit.card['targets'].get('value_unit', 'native')}
    return replace(unit, asof=origin, histories=histories,
        card={'task': {'id': 'internal-adaptive-numeric-prefix'}, 'targets': targets,
              'provenance': {'data_cutoff': origin}}, spec={'targets': deepcopy(targets)})


def prepare(unit, tapes, train, confirm, *, budget=None):
    """Produce all prefix forecasts and denominators before any explicit label reads."""
    prepared = {'train': [], 'confirm': []}
    for group, rows in [('train', train), ('confirm', confirm)]:
        for row in rows:
            _check(budget, 'historical_origin_start')
            histories = _prefix(tapes, row['origin'])
            inner = _inner(unit, row['origin'], histories)
            seed, prefix_hash = _seed(histories, unit.assets, unit.horizons, row['origin'])
            base, layout, _ = _base(inner, draws=INNER_DRAWS, seed=seed)
            if not eligible(inner, layout):
                raise Unsupported('historical_grid_became_ineligible')
            family, sources = _family(base, inner, layout, seed)
            distribution = reference.from_histories(histories, unit.assets, unit.horizons, row['origin'])
            divisors = scales.expected_divisors(distribution['mean'], distribution['covariance'])
            _check(budget, 'historical_origin_complete')
            identity = {'entry': deepcopy(row), 'seed': seed, 'prefix_sha256': prefix_hash,
                'proposal_sha256': {name: reference.array_hash(cube) for name, cube in family.items()},
                'm0_mean_sha256': distribution['mean_sha256'], 'm0_covariance_sha256': distribution['covariance_sha256'],
                'divisors': divisors, 'rate_source': sources}
            prepared[group].append({'entry': row, 'family': family, 'scales': divisors,
                                    'seed': seed, 'identity': identity})
    identity = {group: [r['identity'] for r in rows] for group, rows in prepared.items()}
    return prepared, identity


def read_labels(unit, tapes, rows, *, budget=None):
    """Read only exact endpoints and the frozen label-vintage cutoff of each row."""
    labels = []
    for row in rows:
        _check(budget, 'historical_label_origin')
        entry = row['entry']
        snapshots = _snapshots(tapes, entry['label_vintage_cutoff'])
        end = pd.Timestamp(entry['label_vintage_cutoff'])+pd.Timedelta(days=1)
        endpoints = [pd.Timestamp(x) for x in entry['endpoints']]
        meta = _label_metadata(snapshots, endpoints, unit.assets, end)
        if meta != entry['label_available_at']:
            raise Unsupported('label_vintage_identity_changed')
        values = np.array([[float(snapshots[a].loc[snapshots[a].date == endpoint, 'value'].iloc[0])
                            for endpoint in endpoints] for a in unit.assets])
        if not np.isfinite(values).all():
            raise Unsupported('nonfinite_historical_label_no_origin_dropped')
        labels.append(values)
    return labels


def loss(cube, label, divisors):
    raw = official._composite(cube.reshape(len(cube), -1), np.asarray(label).reshape(-1),
        weights=(.5, .3, .2), tail_levels=TAILS, joint='variogram', tail_metric='pinball', ref_scale=None)
    score = scales.rescore_raw({'raw': {k: float(raw[k]) for k in scales.COMPONENTS},
                                'cell_count': cube.shape[1]*cube.shape[2], 'n_draws': len(cube)}, divisors)
    return score['loss_clipped']


def _whole_rows(family, weights, seed):
    base = family['V6']
    probabilities = np.array([weights.get(name, 0.) for name in FAMILY], dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or abs(float(probabilities.sum())-1.) > 1e-12):
        raise ValueError('Invalid mixture probabilities')
    quotas = len(base)*probabilities
    counts = np.floor(quotas).astype(int)
    remainder = len(base)-int(counts.sum())
    for index in np.argsort(-(quotas-counts), kind='stable')[:remainder]:
        counts[index] += 1
    derived = int.from_bytes(hashlib.sha256(('adaptive-v1-whole-rows:'+str(seed)).encode()).digest()[:8], 'little')
    order = np.random.default_rng(derived).permutation(len(base))
    out, cursor = base.copy(), 0
    for name, count in zip(FAMILY, counts):
        selected = order[cursor:cursor+count]
        out[selected] = family[name][selected]
        cursor += count
    return out, {name: int(count) for name, count in zip(FAMILY, counts)}


def apply(family, strategy, seed):
    if strategy['kind'] == 'whole_joint_rows':
        return _whole_rows(family, strategy['weights'], seed)
    if strategy['kind'] != 'same_row_shrink' or strategy['proposal'] not in FAMILY:
        raise ValueError('Unknown fixed strategy')
    beta = float(strategy['beta'])
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('Invalid fitted shrink strength')
    base, selected = family['V6'], family[strategy['proposal']]
    if beta == 0.:
        return base.copy(), {'beta': 0.}
    if beta == 1.:
        return selected.copy(), {'beta': 1.}
    return base+beta*(selected-base), {'beta': beta}


def fit_policies(train, labels, *, budget=None):
    """Receives training rows only; confirmation labels cannot affect fitting."""
    if len(train) < MIN_TRAIN or len(train) != len(labels):
        raise ValueError('Insufficient aligned fitting rows')
    _check(budget, 'fit_start')
    losses = {}
    for name in FAMILY:
        losses[name] = []
        for row, label in zip(train, labels):
            _check(budget, 'fit_proposal_origin')
            losses[name].append(loss(row['family'][name], label, row['scales']))
    means = {name: float(np.mean(values)) for name, values in losses.items()}
    selected = min(FAMILY, key=lambda name: means[name])  # V6 wins exact ties.
    evaluated, beta = [{'source': 'identity', 'beta': 0., 'loss': means['V6']}], 0.
    if selected != 'V6':
        def objective(value):
            _check(budget, 'solver_objective_start')
            strategy = {'kind': 'same_row_shrink', 'proposal': selected, 'beta': float(value)}
            values = []
            for row, label in zip(train, labels):
                _check(budget, 'solver_origin')
                values.append(loss(apply(row['family'], strategy, row['seed'])[0], label, row['scales']))
            _check(budget, 'solver_objective_complete')
            return float(np.mean(values))
        optimum = minimize_scalar(objective, bounds=(0., 1.), method='bounded', options={'maxiter': 24, 'xatol': .01})
        _check(budget, 'solver_complete')
        if not optimum.success or not np.isfinite(optimum.x):
            raise Unsupported('bounded_shrink_solver_failed')
        evaluated += [{'source': 'solver', 'beta': float(optimum.x), 'loss': objective(optimum.x)},
                      {'source': 'full_proposal', 'beta': 1., 'loss': means[selected]}]
        beta = min(evaluated, key=lambda row: row['loss'])['beta']
    baseline = means['V6']
    gains = {name: max(0., baseline-means[name])/baseline if baseline > 0 else 0. for name in FAMILY[1:]}
    total = sum(gains.values())
    mass = min(.5, total/(1.+total))
    weights = {'V6': 1.-mass, **{name: mass*gains[name]/total if total else 0. for name in FAMILY[1:]}}
    strategies = {
        'conservative': {'kind': 'same_row_shrink', 'proposal': selected, 'beta': min(.5, beta)},
        'mixture': {'kind': 'whole_joint_rows', 'weights': weights},
        'shrink': {'kind': 'same_row_shrink', 'proposal': selected, 'beta': beta}}
    _check(budget, 'fit_complete')
    return strategies, {'training_mean_losses': means, 'selected_proposal': selected,
        'shrink_evaluations': evaluated, 'training_origins': [row['entry']['origin'] for row in train],
        'strategies_frozen_sha256': object_hash(strategies), 'confirmation_used': False,
        'global_optimality_proven': False}


def confirm_policies(rows, cubes, labels, *, budget=None):
    """One fixed gate per mode. No strategy or coefficient is chosen on this fold."""
    if len(rows) < MIN_CONFIRM or len(labels) != len(rows):
        raise ValueError('Insufficient aligned confirmation rows')
    _check(budget, 'confirmation_start')
    base_values = []
    for row, label in zip(rows, labels):
        _check(budget, 'confirmation_base_origin')
        base_values.append(loss(row['family']['V6'], label, row['scales']))
    base = np.asarray(base_values)
    decisions = {}
    for mode in MODES:
        values = []
        for cube, label, row in zip(cubes[mode], labels, rows):
            _check(budget, 'confirmation_policy_origin')
            values.append(loss(cube, label, row['scales']))
        values = np.asarray(values)
        gain = base-values
        wins, losses = int((gain > 0).sum()), int((gain < 0).sum())
        decisions[mode] = {'apply_adaptation': bool(gain.mean() > 0 and wins > losses),
            'mean_loss': float(values.mean()), 'V6_mean_loss': float(base.mean()),
            'mean_gain': float(gain.mean()), 'wins': wins, 'losses': losses, 'ties': int((gain == 0).sum()),
            'origins': [row['entry']['origin'] for row in rows],
            'uses_only_one_fixed_confirmation_candidate_per_mode': True}
    _check(budget, 'confirmation_complete')
    return decisions


def forecast_all(unit, *, draws=20000, seed=0, adaptation_seconds=DEFAULT_ADAPTATION_SECONDS, clock=monotonic):
    """Return conservative/mixture/shrink cubes and full cutoff-local provenance.

    Current realized outcomes cannot be passed. Unsupported grids return the
    same untouched V6 cube in all three modes, with no adaptive-fit claim.
    """
    if (type(adaptation_seconds) not in (int, float) or not math.isfinite(adaptation_seconds)
            or adaptation_seconds <= 0 or not callable(clock)):
        raise ValueError('invalid_adaptation_budget')
    unit, dropped = core.bounded_unit(unit)
    audit = {'settings': settings(), 'asof': unit.asof, 'assets': list(unit.assets),
        'horizons': list(unit.horizons), 'draws': draws, 'seed': seed,
        'future_rows_discarded': dropped, 'fit_completed': False,
        'current_realized_or_external_scores_used': False,
        'task_id_title_family_used_for_selection': False,
        'policy_compliance_certified': False, 'accuracy_validated': False,
        'deployment_adapter': 'adaptive_core_v1',
        'adaptive_budget': {'seconds': float(adaptation_seconds), 'scope': 'after usable V6 base, before output writing',
                            'cooperative': True, 'can_interrupt_blocking_call_or_OS_kill': False}}
    with threadpool_limits(limits=1):
        base, layout, base_detail = _base(unit, draws=draws, seed=seed)
        audit['base'] = base_detail
        untouched = {mode: base.copy() for mode in MODES}
        if not eligible(unit, layout):
            audit['fallback_reason'] = 'unsupported_target_grid_unmodified_V6'
            return untouched, audit
        try:
            budget = _Budget(float(adaptation_seconds), clock)
            _check(budget, 'raw_tapes_start')
            _, tapes = load_tapes(unit)
            _check(budget, 'raw_tapes_complete')
            audit['vintage_policies'] = {tape.asset: tape.vintage_policy for tape in tapes}
            audit['explicit_missing_available_at_removed'] = {tape.asset: tape.explicit_missing_available_at_removed for tape in tapes}
            roster, exclusions = build_roster(unit, tapes, budget=budget)
            audit.update(eligible_roster=roster, exclusions=exclusions)
            train, validation, partition = split_roster(unit, tapes, roster, budget=budget)
            audit['partition'] = partition
            prepared, identities = prepare(unit, tapes, train, validation, budget=budget)
            audit['historical_prediction_identity'] = identities
            audit['historical_prediction_identity_sha256'] = object_hash(identities)
            audit['all_inner_forecasts_and_divisors_completed_before_label_reads'] = True
            training_labels = read_labels(unit, tapes, prepared['train'], budget=budget)
            strategies, fit_detail = fit_policies(prepared['train'], training_labels, budget=budget)
            audit.update(strategies=deepcopy(strategies), fit=fit_detail)
            # Freeze all validation-policy cubes before reading validation truth.
            validation_cubes = {}
            for mode in MODES:
                validation_cubes[mode] = []
                for row in prepared['confirm']:
                    _check(budget, 'confirmation_prediction_origin')
                    validation_cubes[mode].append(apply(row['family'], strategies[mode], row['seed'])[0])
            audit['confirmation_prediction_sha256'] = {
                mode: [reference.array_hash(cube) for cube in cubes] for mode, cubes in validation_cubes.items()}
            validation_labels = read_labels(unit, tapes, prepared['confirm'], budget=budget)
            decisions = confirm_policies(prepared['confirm'], validation_cubes, validation_labels, budget=budget)
            if object_hash(strategies) != fit_detail['strategies_frozen_sha256']:
                raise ValueError('Fitted strategy changed during confirmation')
            _check(budget, 'current_family_start')
            current_family, source_info = _family(base, unit, layout, seed)
            output, application = {}, {}
            for mode in MODES:
                _check(budget, 'current_application')
                if decisions[mode]['apply_adaptation']:
                    output[mode], application[mode] = apply(current_family, strategies[mode], seed)
                else:
                    output[mode], application[mode] = base.copy(), {'reason': 'confirmation_gate_retains_V6'}
                if output[mode].shape != base.shape or not np.isfinite(output[mode]).all():
                    raise ValueError('Invalid final candidate cube')
            _check(budget, 'adaptation_complete')
            audit.update(fit_completed=True, confirmation=decisions, application=application,
                         current_rate_source=source_info,
                         output_sha256={name: reference.array_hash(cube) for name, cube in output.items()})
            return output, audit
        except OSError:
            audit.update(fallback_reason='adaptive_io_error', fallback_detail={},
                         no_failed_historical_origin_dropped=True)
            return untouched, audit
        except AdaptationBudgetExpired as exc:
            audit.update(fallback_reason='adaptive_budget_exhausted', fallback_detail=exc.detail,
                         no_failed_historical_origin_dropped=True)
            return untouched, audit
        except (Unsupported, ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            audit.update(fallback_reason=f'{type(exc).__name__}: {exc}',
                         fallback_detail=getattr(exc, 'detail', {}),
                         no_failed_historical_origin_dropped=True)
            return untouched, audit


def forecast(unit, *, mode='conservative', draws=20000, seed=0,
             adaptation_seconds=DEFAULT_ADAPTATION_SECONDS, clock=monotonic):
    if mode not in MODES:
        raise ValueError('Unknown adaptive mode')
    cubes, audit = forecast_all(unit, draws=draws, seed=seed, adaptation_seconds=adaptation_seconds, clock=clock)
    return cubes[mode], {**audit, 'returned_mode': mode}
