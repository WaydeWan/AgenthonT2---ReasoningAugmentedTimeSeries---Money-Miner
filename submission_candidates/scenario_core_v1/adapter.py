"""Apply the frozen weak scenario only to a supported current-unit input.

The caller supplies the Unit returned by submission_candidates.io.read_unit.
That loader is responsible for selecting vintages available at the task cutoff.
This adapter does not reopen panels, load training bundles, or contact a model.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from submission_candidates.text_core.evidence import EvidenceError, load_corpus
from .runtime import build_view, apply_scenario

COMPONENT_VERSION = 'scenario-core-v1'


def _context(unit):
    asof = unit.asof
    if not isinstance(asof, str) or date.fromisoformat(asof).isoformat() != asof:
        raise EvidenceError('invalid_unit_cutoff')
    declared = (unit.card.get('forecast', {}).get('asof')
                or unit.card.get('provenance', {}).get('data_cutoff'))
    if declared != asof:
        raise EvidenceError('unit_cutoff_mismatch')
    target = unit.card['targets']
    for key in ('asset_ids', 'horizons', 'target_type', 'value_unit', 'target_frequency'):
        spec = unit.spec.get('targets', {})
        if key in spec and key in target and spec[key] != target[key]:
            raise EvidenceError('unit_target_mismatch')
    if list(unit.assets) != ['UST_2Y']:
        raise EvidenceError('unsupported_asset_grid')
    frequency = target.get('target_frequency', unit.card.get('metadata', {}).get('target_frequency', 'daily'))
    if unit.target_type != 'level' or frequency != 'daily':
        raise EvidenceError('unsupported_target_clock')
    # Never treat a decimal yield, return or basis-point value as a percentage.
    if target.get('value_unit') not in {'percent_per_annum', 'percentage_points', 'percent'}:
        raise EvidenceError('unsupported_or_missing_value_unit')
    horizons = list(unit.horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h < 1 for h in horizons):
        raise EvidenceError('invalid_horizons')
    history = unit.histories.get('UST_2Y')
    if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
        raise EvidenceError('invalid_anchor_history')
    if history.index.tz is not None or history.index.hasnans:
        raise EvidenceError('invalid_anchor_dates')
    # Future values are discarded before type conversion or finite checks.
    cutoff = pd.Timestamp(asof)
    prefix = history.loc[history.index <= cutoff].copy().sort_index()
    if prefix.empty or not prefix.index.is_unique or not prefix.index.equals(prefix.index.normalize()):
        raise EvidenceError('missing_or_ambiguous_current_anchor')
    prefix = prefix.astype(float)
    if not np.isfinite(prefix.to_numpy()).all():
        raise EvidenceError('nonfinite_current_anchor_history')
    anchor = {'asset': 'UST_2Y', 'value': float(prefix.iloc[-1]),
        'observed_at': str(prefix.index[-1].date()), 'available_at': asof, 'unit': 'percentage_points'}
    # available_at above is an upper bound guaranteed by the trusted Unit loader,
    # not a fabricated first-publication timestamp for the selected observation.
    return asof, horizons, anchor, len(history) - len(prefix)


def apply_optional(samples, unit, textdir=None, seed=0):
    """Return (samples, audit, source_indices); absent support is exact identity.

    Only the fixed 'scenario' route is exposed here. Research beta-zero/shuffled
    controls remain in the byte-identical reference copy but are never selected.
    No House/NVIDIA request, learned artifact, future outcome or global corpus
    is accepted. The current input's text directory defaults to unit.root/text.
    """
    baseline = np.asarray(samples)
    if (baseline.ndim != 3 or not 200 <= len(baseline) <= 20000
            or baseline.dtype.kind != 'f' or not np.isfinite(baseline).all()):
        raise ValueError('Numerical input must already be a valid finite official sample cube')
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    if baseline.shape[1:] != (len(unit.assets), len(unit.horizons)):
        raise ValueError('Numerical input and unit grid differ')
    view = {'ready': False, 'reason': 'not_prepared'}
    preparation = {'source': 'current_unit_supplied_corpus_only', 'documents_loaded': 0,
        'future_documents_skipped': 0, 'future_history_rows_discarded': 0,
        'anchor_availability': 'asof is the trusted Unit loader availability upper bound, not first release',
        'intraday_order_established': False, 'training_artifact_loaded': False,
        'api_calls': 0, 'forecast_accuracy_claimed': False}
    try:
        asof, horizons, anchor, discarded = _context(unit)
        preparation['future_history_rows_discarded'] = discarded
        root = Path(unit.root).resolve()
        directory = Path(textdir).resolve() if textdir is not None else root / 'text'
        if not directory.is_relative_to(root):
            raise EvidenceError('text_directory_outside_current_unit')
        documents, skipped = load_corpus(directory, asof)
        preparation.update(documents_loaded=len(documents),
            future_documents_skipped=len(skipped),
            source_sha256={d['doc_id']: d['sha256'] for d in documents})
        view = build_view(documents=documents, asof=asof, assets=list(unit.assets),
            horizons=horizons, anchor=anchor, target_type='level', frequency='daily')
    except (EvidenceError, OSError, KeyError, TypeError, ValueError, OverflowError, UnicodeError) as exc:
        reason = str(exc) if isinstance(exc, EvidenceError) else 'preparation_failed:' + type(exc).__name__
        view = {'ready': False, 'reason': reason}
    output, audit, indices = apply_scenario(baseline, view, seed=seed, control='scenario')
    audit.update(component_version=COMPONENT_VERSION, preparation=preparation,
        view=view, control_selected_by_wrapper='scenario',
        evaluation_boundary='Exploratory fixed-N1 historical proxy: 14 eligible units, only 2 real interventions; no general/official efficacy claim')
    return output, audit, indices
