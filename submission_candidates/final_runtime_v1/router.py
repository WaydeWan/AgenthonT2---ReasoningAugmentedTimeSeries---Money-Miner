"""Five numerical policies, with exact same-call raw singleton FX samples."""
from dataclasses import dataclass, asdict
import hashlib
import math
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from ..numeric_core import runtime as fixed
from ..adaptive_core_v1 import runtime as adaptive
from .plugins import disabled_plugin

POLICIES = ('fixedN1', 'fixedN2', 'fixedN3', 'conservative', 'shrink')
DEFAULT_POLICY = 'fixedN1'


@dataclass(frozen=True)
class FinalOptions:
    policy: str = DEFAULT_POLICY
    draws: int = 20000
    seed: int = 0
    adaptation_seconds: float = adaptive.DEFAULT_ADAPTATION_SECONDS

    def validate(self):
        if self.policy not in POLICIES:
            raise ValueError('Unknown final numerical policy')
        fixed.NumericOptions(draws=self.draws, seed=self.seed, fit_single_fx_width=False).validate()
        if (type(self.adaptation_seconds) not in (int, float)
                or not math.isfinite(self.adaptation_seconds) or self.adaptation_seconds <= 0):
            raise ValueError('Invalid adaptation budget')
        return self


def array_hash(cube):
    array = np.ascontiguousarray(cube, dtype='<f8')
    h = hashlib.sha256(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def singleton_fx(unit, cube, layout):
    """Exactly the old .85 branch; do not silently broaden to all FX grids."""
    return (layout.frequency == 'daily' and unit.target_type in {'level', 'yield'}
            and cube.shape[1:] == (1, 1) and layout.transforms == ['log_level'])


def fixed_forecast(unit, options):
    """Reuse the frozen base once, then keep raw FX or original N1/N2/N3 route."""
    bounded, dropped = fixed.bounded_unit(unit)
    numerical = fixed.NumericOptions(candidate=options.policy[-2:], draws=options.draws,
                                     seed=options.seed, fit_single_fx_width=False).validate()
    with threadpool_limits(limits=1):
        base, layout, base_detail = fixed._base(bounded, numerical)
        base_hash = array_hash(base)
        if singleton_fx(bounded, base, layout):
            samples = base.copy()
            post = {'branch': 'single_fx_raw_same_call_V6', 'actual_fit': False,
                    'legacy_fixed_width_085_applied': False, 'width_fitter_called': False,
                    'retains_raw_joint_sample_bytes': True}
        else:
            try:
                samples, post = fixed._postprocess(bounded, base, layout, numerical, None)
            except (ValueError, FloatingPointError, OverflowError) as error:
                if isinstance(error, fixed.WidthIntegrationError):
                    raise
                samples = base.copy()
                post = {'branch': 'retain_base_after_transform_failure', 'actual_fit': False,
                        'reason_type': type(error).__name__}
    output_hash = array_hash(samples)
    if post['branch'] == 'single_fx_raw_same_call_V6' and (base_hash != output_hash or not np.array_equal(base, samples)):
        raise ValueError('Raw singleton FX samples changed')
    return samples, {'route': post['branch'], 'fixed_candidate': numerical.candidate,
        'asof': bounded.asof, 'assets': list(bounded.assets), 'horizons': list(bounded.horizons),
        'target_type': bounded.target_type, 'frequency': layout.frequency, 'transforms': list(layout.transforms),
        'history_sha256': fixed._identity(bounded), 'future_rows_discarded': dropped,
        'base_samples_sha256': base_hash, 'output_samples_sha256': output_hash,
        'output_identical_to_same_call_raw_V6': base_hash == output_hash,
        'numerical': base_detail, 'postprocess': post, 'base_calls': 1,
        'adaptation_attempted': False, 'width_fitter_called': False,
        'fallback_reason': post.get('reason_type') if post['branch'] == 'retain_base_after_transform_failure' else None}


def adaptive_route(audit, mode):
    if audit.get('fallback_reason'):
        return 'adaptive_fallback_raw_V6', audit['fallback_reason']
    application = audit.get('application', {}).get(mode, {})
    if application.get('reason') == 'confirmation_gate_retains_V6':
        return 'adaptive_confirmation_retains_raw_V6', application['reason']
    if audit.get('fit_completed'):
        return 'adaptive_selected_policy', None
    raise ValueError('Adaptive output has neither completed fit nor explicit fallback')


def forecast(unit, options=None, *, plugin=None, clock=monotonic):
    """Pure numerical forecast. No credentials, text model or external endpoint.

    Fixed policies preserve the frozen postprocessing outside its singleton FX
    branch. Adaptive policies retain their own unsupported-grid raw-V6 fallback;
    no fixed .85/.15 transform is added after adaptation or fallback.
    """
    options = (options or FinalOptions()).validate()
    if not callable(clock):
        raise ValueError('Clock must be callable')
    plugin_audit = disabled_plugin(plugin)
    started = float(clock())
    if not math.isfinite(started):
        raise ValueError('Invalid clock')
    if options.policy.startswith('fixed'):
        samples, numeric = fixed_forecast(unit, options)
        route, fallback = numeric['route'], numeric['fallback_reason']
    else:
        samples, numeric = adaptive.forecast(unit, mode=options.policy, draws=options.draws, seed=options.seed,
                                            adaptation_seconds=options.adaptation_seconds, clock=clock)
        route, fallback = adaptive_route(numeric, options.policy)
    elapsed = float(clock())-started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid elapsed clock')
    if samples.shape != (options.draws, len(unit.assets), len(unit.horizons)) or not np.isfinite(samples).all():
        raise ValueError('Final policy violates finite declared joint-grid contract')
    return samples, {'version': 'final-runtime-v1', 'method': options.policy+' pure numerical joint forecast',
        'policy': options.policy, 'route': route, 'options': asdict(options), 'numeric': numeric,
        'fallback_reason': fallback, 'numeric_elapsed_seconds': round(elapsed, 6),
        'numeric_elapsed_includes_output_writing': False,
        'adaptive_budget_scope': ('Cooperative checks after usable raw V6 exists; excludes base generation/output writing and cannot interrupt a blocking call or OS kill'
                                 if not options.policy.startswith('fixed') else 'Not used by fixed policy'),
        'house_calls': 0, 'text_used': False, 'optional_plugin': plugin_audit,
        'current_future_outcome_used': False, 'routing_uses_task_id_title_or_F_family': False,
        'output_samples_sha256': array_hash(samples), 'disclosure_status': 'Pending release-owner decision; not certified model-free or policy-compliant',
        'performance_status': 'No new accuracy claim from this integration wrapper'}
