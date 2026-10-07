"""Production numerical adapter; no research runners, scores, or outcome files.

Input is the V8 Unit contract after its official loader has selected eligible
vintages. Output is [draw, asset, horizon] in the declared native target units.
The adapter bounds histories again before inspecting any supplied values.
"""
from dataclasses import dataclass, replace, field
from copy import deepcopy
from datetime import date
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .v6.config import Config
from .v6.numeric import fit_numeric
from .v6.hybrid import sample_hybrid
from .v6.fallback import diagonal_fit, recoverable
from .v6.sampler import sample_paths
from .v6.targets import target_layout
from .postprocess import rate_sources, rate_combination, tail_sharpen, width_scale

CANDIDATES = ('N1', 'N2', 'N3')


class WidthIntegrationError(ValueError):
    """Missing or invalid configured fitter is a configuration failure."""


@dataclass(frozen=True)
class NumericOptions:
    candidate: str = 'N1'
    draws: int = 20000
    seed: int = 0
    # Width fitter is an explicit integration point for the migrated objective.
    # Without one, alpha=1 is a named fixed candidate, not a claimed fitted N.
    fit_single_fx_width: bool = False
    tail_strength: float = .15

    def validate(self):
        if self.candidate not in CANDIDATES:
            raise ValueError('Unknown numerical candidate')
        if type(self.draws) is not int or not 200 <= self.draws <= 20000:
            raise ValueError('Draw count outside official limits')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Seed must be a nonnegative integer')
        if type(self.fit_single_fx_width) is not bool:
            raise ValueError('Width fit switch must be boolean')
        if self.tail_strength != .15:
            raise ValueError('Tail strength is a predeclared fixed configuration')
        return self


@dataclass(frozen=True)
class WidthFit:
    """Return type for a trusted cutoff-local width fitter.

    Origins and label availability dates are required for any learned alpha.
    The runtime does not itself certify first-release vintages: the official
    loader and fitter must respect available_at, not just observation dates.
    """
    alpha: float
    origins: tuple[str, ...]
    label_available_at: tuple[str, ...]
    method: str
    actual_fit: bool = True
    diagnostics: dict = field(default_factory=dict)


def bounded_unit(unit):
    declared_cutoff = (unit.card.get('forecast', {}).get('asof')
                       or unit.card.get('provenance', {}).get('data_cutoff'))
    if declared_cutoff != unit.asof:
        raise ValueError('Explicit cutoff differs from trusted card')
    cutoff = pd.Timestamp(unit.asof)
    if pd.isna(cutoff) or cutoff.tz is not None or cutoff != cutoff.normalize():
        raise ValueError('Expected a normalized explicit cutoff date')
    if date.fromisoformat(unit.asof).isoformat() != unit.asof:
        raise ValueError('Expected ISO cutoff date')
    assets, horizons = list(unit.assets), list(unit.horizons)
    if not assets or len(set(assets)) != len(assets) or not all(isinstance(a, str) and a for a in assets):
        raise ValueError('Invalid asset grid')
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError('Invalid horizon grid')
    if unit.target_type not in {'level', 'yield', 'log_return'}:
        raise ValueError('Unsupported native target type')
    for key in ('asset_ids', 'horizons', 'target_type'):
        declared = unit.spec.get('targets', {})
        if key in declared and declared[key] != unit.card['targets'].get(key):
            raise ValueError('Card/spec target mismatch')
    selected, dropped = {}, {}
    for asset in assets:
        history = unit.histories.get(asset)
        if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError('Each asset requires a dated series')
        if history.index.tz is not None or history.index.hasnans:
            raise ValueError('History dates must be finite and timezone-naive')
        # Truncate before value conversion/duplicates/estimation. Future poison
        # values and duplicates therefore cannot alter a current forecast.
        bounded = history.loc[history.index <= cutoff].copy().sort_index()
        dropped[asset] = len(history) - len(bounded)
        if bounded.empty or not bounded.index.is_unique or not bounded.index.equals(bounded.index.normalize()):
            raise ValueError('Need a unique normalized bounded history and real anchor')
        bounded = bounded.astype(float)
        if not np.isfinite(bounded.to_numpy()).all():
            raise ValueError('Nonfinite bounded history')
        if asset not in unit.panel_ids:
            raise ValueError('Missing panel identity')
        selected[asset] = bounded
    result = replace(unit, histories=selected, card=deepcopy(unit.card), spec=deepcopy(unit.spec),
                     input_diagnostics=deepcopy(unit.input_diagnostics), panel_ids=dict(unit.panel_ids))
    target_layout(result)  # Resolve explicit monthly periods; reject ambiguity.
    return result, dropped


def _identity(unit):
    # Do not include task id/title, family name or scoring/outcome metadata.
    layout = target_layout(unit)
    payload = {'assets': list(unit.assets), 'horizons': list(unit.horizons),
               'asof': unit.asof, 'target_type': unit.target_type,
               'frequency': layout.frequency, 'transforms': list(layout.transforms),
               'panel_ids': {asset: unit.panel_ids[asset] for asset in unit.assets},
               'layout': layout.endpoints.tolist()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for asset in unit.assets:
        history = unit.histories[asset]
        digest.update(asset.encode())
        digest.update(np.asarray(history.index.asi8, dtype='<i8').tobytes())
        digest.update(np.asarray(history.to_numpy(), dtype='<f8').tobytes())
    return digest.hexdigest()


def _base(unit, options):
    config = Config(draws=options.draws, seed=options.seed, v7_method='V6').validate()
    try:
        fit = fit_numeric(unit, config)
        cube, detail = sample_hybrid(unit, fit, config)
        return cube, fit.layout, {'implementation_path': 'extracted_v6', 'numeric': fit.diagnostics, 'hybrid': detail}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        if not recoverable(exc):
            raise
        fit = diagonal_fit(unit, config, exc)
        cube = sample_paths(fit, replace(config, model='B0', innovation_family='gaussian', text_decay_periods=0))
        return cube, fit.layout, {'implementation_path': 'extracted_v6_diagonal_recovery', 'numeric': fit.diagnostics}


def _validated_alpha(result, cutoff):
    if not isinstance(result, WidthFit):
        raise ValueError('Width fitter must return WidthFit')
    if not np.isfinite(result.alpha) or not .5 <= result.alpha <= 1.5:
        raise ValueError('Width alpha outside fixed bounds')
    if not result.actual_fit:
        if result.alpha != 1. or not result.method or not result.diagnostics.get('reason'):
            raise ValueError('An untrained width fallback must be explicit alpha1 with a reason')
        return 1.
    if not result.actual_fit or not isinstance(result.method, str) or not result.method:
        raise ValueError('Width result must identify an actual fitting method')
    if len(result.origins) < 24 or len(set(result.origins)) != len(result.origins):
        raise ValueError('Need at least 24 distinct mature training origins')
    if len(result.origins) != len(result.label_available_at):
        raise ValueError('Training origins and maturity dates must align')
    end = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
    for origin, available in zip(result.origins, result.label_available_at):
        left, right = pd.Timestamp(origin), pd.Timestamp(available)
        if pd.isna(left) or pd.isna(right) or left.tz is not None or right.tz is not None or not left < right < end:
            raise ValueError('Immature, missing, or invalid training label availability')
    return float(result.alpha)


def _postprocess(unit, cube, layout, options, width_fitter):
    is_rate = (layout.frequency == 'daily' and unit.target_type in {'level', 'yield'}
               and all(a.startswith('UST_') for a in unit.assets) and cube.shape[1] * cube.shape[2] >= 2)
    if is_rate:
        anchors = [float(unit.histories[a].iloc[-1]) for a in unit.assets]
        soft, location, sources = rate_sources(cube, anchors, options.seed)
        out, combine = rate_combination(cube, soft, location, options.candidate, options.seed)
        return out, {'branch': 'pure_ust_multicell', 'sources': sources, 'combination': combine,
                     'actual_fit': False, 'research_formula_parity_scope': 'daily native UST grid with matching anchors'}
    single_fx = (layout.frequency == 'daily' and unit.target_type in {'level', 'yield'}
                 and cube.shape[1:] == (1, 1) and layout.transforms == ['log_level'])
    if single_fx:
        alpha, fit_detail = 1., {'actual_fit': False, 'reason': 'fixed_alpha1_pending_migrated_width_objective'}
        if options.fit_single_fx_width:
            if width_fitter is None:
                raise WidthIntegrationError('Requested width learning needs an explicit migrated cutoff-local fitter')
            result = width_fitter(unit=unit, base_cube=cube.copy(), seed=options.seed)
            try:
                alpha = _validated_alpha(result, unit.asof)
            except ValueError as exc:
                raise WidthIntegrationError(str(exc)) from exc
            fit_detail = {'actual_fit': result.actual_fit, 'method': result.method, 'training_origins': len(result.origins),
                          'origin_min': min(result.origins, default=None), 'origin_max': max(result.origins, default=None),
                          'label_available_max': max(result.label_available_at, default=None),
                          'fit_diagnostics': deepcopy(result.diagnostics)}
        return width_scale(cube, layout.transforms, .85 * alpha), {
            'branch': 'single_fx_fixed_width' if not fit_detail['actual_fit'] else 'single_fx_learned_width',
            'base_width': .85, 'alpha': alpha, **fit_detail}
    # Same central-tail operation generalized to the target's declared modeling
    # coordinates. Monthly and return scope has contract tests, no uplift claim.
    out, detail = tail_sharpen(cube, layout.transforms, options.tail_strength)
    return out, {'branch': 'other_shape_central_tail', **detail,
                 'scope': 'daily levels, log_return sums, or explicit low-frequency periods; performance pending'}


def forecast(unit, options=None, *, width_fitter=None):
    options = (options or NumericOptions()).validate()
    bounded, dropped = bounded_unit(unit)
    with threadpool_limits(limits=1):
        base, layout, base_detail = _base(bounded, options)
        try:
            cube, post = _postprocess(bounded, base, layout, options, width_fitter)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            # A requested learning stage must not masquerade as a trained model.
            if isinstance(exc, WidthIntegrationError):
                raise
            cube = base.copy()
            post = {'branch': 'retain_base_after_transform_failure', 'actual_fit': False,
                    'reason': f'{type(exc).__name__}: {exc}'}
    expected = (options.draws, len(bounded.assets), len(bounded.horizons))
    if cube.shape != expected or not np.isfinite(cube).all():
        raise ValueError('Numerical result violates finite official cube contract')
    return cube, {'candidate': options.candidate, 'draws': options.draws, 'seed': options.seed,
        'asof': bounded.asof, 'assets': list(bounded.assets), 'horizons': list(bounded.horizons),
        'target_type': bounded.target_type, 'frequency': layout.frequency,
        'transforms': list(layout.transforms), 'steps': (layout.endpoints - layout.anchor_ticks[:, None]).tolist(),
        'history_sha256': _identity(bounded), 'future_rows_discarded': dropped,
        'thread_limit': 1, 'numerical': base_detail, 'postprocess': post,
        'routing_uses_task_id_or_F_family': False, 'current_outcome_or_ref_scale_used': False,
        'training_data_scope': 'Current cutoff-bounded unit only; loader must select available vintages first',
        'deployment_difference': 'Log-return short branch uses official log1p steps; singleton FX defaults alpha1 until migrated fitter supplied',
        'performance_status': 'Fixed candidate; deployment extension is not a new-score performance claim'}


def forecast_all(unit, *, draws=20000, seed=0, fit_single_fx_width=False, width_fitter=None):
    """Convenience interface. Root integration normally requests one candidate."""
    cubes, details = {}, {}
    for name in CANDIDATES:
        cubes[name], details[name] = forecast(unit, NumericOptions(name, draws, seed, fit_single_fx_width), width_fitter=width_fitter)
    return cubes, details
