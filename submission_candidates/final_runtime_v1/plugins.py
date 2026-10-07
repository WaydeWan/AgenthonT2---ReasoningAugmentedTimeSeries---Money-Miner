"""Explicit off/SEP-only production hook; no arbitrary plugin or LLM route."""
from typing import Protocol
import numpy as np

TEXT_MODES = ('off', 'sep_v1')


class ScenarioPlugin(Protocol):
    """Documented shape of a scenario hook, with no future-label argument.

    An implementation would return a complete native joint sample cube and an
    inspectable source/selection record. This interface alone is not approval
    or calibration. Arbitrary plugin injection remains disabled: the explicit
    sep_v1 route below imports only the fixed production adapter.
    """
    def __call__(self, samples: np.ndarray, *, unit, seed: int) -> tuple[np.ndarray, dict]: ...


def disabled_plugin(plugin):
    if plugin is not None:
        raise ValueError('Optional SEP plugin is reserved and disabled in final_runtime_v1')
    return {'enabled': False, 'implementation': None, 'house_calls': 0,
            'status': 'Numerical stage only; optional SEP is a separate explicit final stage'}


def _production_apply():
    from ..scenario_core_v1 import apply_optional
    return apply_optional


def apply_text(samples, *, unit, textdir, seed, mode='off'):
    """Return cube and audit; unsupported SEP sources retain exact numeric rows.

    The production adapter handles declared text/input-support failures by
    returning identity. Missing code or violated output invariants are release
    errors, not silently labelled successful text fallbacks.
    """
    if mode not in TEXT_MODES:
        raise ValueError('Unknown final text mode')
    if mode == 'off':
        return samples, {'mode': mode, 'enabled': False, 'applied': False, 'house_calls': 0,
                         'reason': 'explicit_text_off', 'changed_rows': 0, 'whole_rows_preserved': True}
    original = np.asarray(samples)
    output, audit, indices = _production_apply()(original.copy(), unit, textdir=textdir, seed=seed)
    output, indices = np.asarray(output), np.asarray(indices)
    if (output.shape != original.shape or not np.isfinite(output).all()
            or indices.shape != (len(original),) or indices.dtype.kind not in 'iu'
            or (indices < 0).any() or (indices >= len(original)).any()
            or not np.array_equal(output, original[indices])):
        raise ValueError('SEP production plugin violated complete joint-row preservation')
    if not isinstance(audit, dict) or type(audit.get('applied')) is not bool:
        raise ValueError('SEP production plugin has no explicit application audit')
    if not audit['applied'] and not np.array_equal(output, original):
        raise ValueError('SEP identity fallback changed numerical samples')
    changed = int(np.count_nonzero(indices != np.arange(len(original))))
    if audit.get('changed_rows') != changed:
        raise ValueError('SEP change count differs from actual source indices')
    return output, {**audit, 'mode': mode, 'enabled': True, 'house_calls': 0,
        'wrapper_verified_whole_joint_rows': True,
        'boundary': 'Exploratory SEP scenario; two prior real interventions support no broad or official prediction claim'}
