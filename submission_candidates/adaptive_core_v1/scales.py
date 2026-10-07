"""Official 60509df M0 expected-error divisors; no realized-outcome inputs.

Raw candidate metrics remain the previously pinned official CRPS/variogram/
pinball values. This module changes only their divisors and the 8.0 clip.
"""
from __future__ import annotations

import math
from numbers import Real
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, hyp1f1, ndtr
from scipy.stats import norm

COMPONENTS = ('marginal', 'joint', 'tail')
TAIL_LEVELS = (.01, .05, .95, .99)
CAP = 8.0
OFFICIAL_COMMIT = '60509df4ad0756443f4af8dc9500aed99a995694'


def pair_variance(delta, variance):
    """Var(|Normal(delta, variance)|**.5), nonnegative and symmetric.

The official hyp1f1 expression loses precision by subtracting two large
numbers for extreme noncentrality. Beyond |delta|/sd=100 use independent
centered quadrature, with a rationalized sqrt difference, for that pair.
No ensemble draws, seeds or outcomes enter either branch.
"""
    delta, variance = float(delta), float(variance)
    if not math.isfinite(delta) or not math.isfinite(variance) or variance < 0:
        raise ValueError('Invalid normal difference moments')
    if variance == 0:
        return 0.0, 'degenerate'
    sd = math.sqrt(variance)
    r = abs(delta) / sd
    if r > 100:
        root = math.sqrt(r)
        def centered(z):
            # r>100 and |z|<=12, so r+z is positive. Keep numerator z
            # directly: (r+z)-r would round to zero when r is enormous.
            return z/(math.sqrt(r+z)+root)
        # The omitted |z|>12 Gaussian mass is <4e-33. The cusp at -r is
        # outside this range in this branch; integration stays smooth.
        first = quad(lambda z: centered(z)*norm.pdf(z), -12., 12.,
                     epsabs=1e-15, epsrel=2e-12)[0]
        second = quad(lambda z: centered(z)**2*norm.pdf(z), -12., 12.,
                      epsabs=1e-15, epsrel=2e-12)[0]
        value = sd * (second-first*first)
        method = 'centered_quadrature_large_noncentrality'
    else:
        e_abs = math.sqrt(2/math.pi)*math.exp(-.5*r*r) + r*(1-2*ndtr(-r))
        e_root = 2**.25 * gamma(.75)/math.sqrt(math.pi) * hyp1f1(-.25, .5, -.5*r*r)
        value = sd * (e_abs-e_root*e_root)
        method = 'hyp1f1'
    if not math.isfinite(value) or value < 0:
        raise ValueError('Nonfinite or negative variogram expected error')
    return float(value), method


def expected_divisors(mean, covariance):
    """Exact Gaussian M0 moments, after jitter/fallback, in one declared grid."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    d = mean.size
    if (mean.ndim != 1 or not d or covariance.shape != (d, d)
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)):
        raise ValueError('Expected finite mean and symmetric covariance on one grid')
    scale = max(float(np.max(np.abs(covariance))), 1e-300)
    if np.linalg.eigvalsh(covariance).min() < -1e-12*scale:
        raise ValueError('Covariance is not positive semidefinite')
    if np.diag(covariance).min() < 0:
        raise ValueError('Negative marginal variance')
    sd = np.sqrt(np.diag(covariance))
    joint = 0.
    methods = {}
    for i in range(d):
        for j in range(i+1, d):
            variance = float(covariance[i, i]+covariance[j, j]-2*covariance[i, j])
            # Roundoff in an exactly degenerate PSD pair, never an epsilon divisor.
            if variance < 0 and variance >= -1e-12*scale:
                variance = 0.
            value, method = pair_variance(mean[i]-mean[j], variance)
            joint += 2*value  # the scorer counts both ordered off-diagonal pairs
            methods[method] = methods.get(method, 0)+1
    raw = {'marginal': float(sd.mean()/math.sqrt(math.pi)), 'joint': float(joint),
           'tail': float(norm.pdf(norm.ppf(TAIL_LEVELS)).mean()*sd.mean())}
    return {'raw': raw, 'effective': {k: v if v != 0 else 1. for k, v in raw.items()},
            'zero_components_replaced_by_one': [k for k, v in raw.items() if v == 0],
            'cell_count': d, 'pair_methods': methods,
            'scale_rule': 'official_M0_analytic_expected_error', 'outcome_used': False}


def effective_weights(cell_count):
    if not isinstance(cell_count, int) or cell_count < 1:
        raise ValueError('Positive integer cell count required')
    return ({'marginal': .5/.7, 'joint': 0., 'tail': .2/.7} if cell_count == 1
            else {'marginal': .5, 'joint': .3, 'tail': .2})


def canonical_weights(value):
    """Official `_score` serializes marginal/joint/tail as a three-float list.

    Accept that exact ordered format or our named internal representation.
    This is a representation repair, not a tolerance or scoring-rule change.
    """
    if isinstance(value, dict):
        if set(value) != set(COMPONENTS):
            raise ValueError('Weight mapping must name precisely the three components')
        values = [value[k] for k in COMPONENTS]
    elif isinstance(value, (list, tuple)) and len(value) == len(COMPONENTS):
        values = list(value)
    else:
        raise ValueError('Official weights require exactly three ordered numeric values')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           or not math.isfinite(v) for v in values):
        raise ValueError('Official weights must be finite numeric values')
    return dict(zip(COMPONENTS, map(float, values)))


def rescore_raw(old, divisors):
    """Use original raw component values, never old normalized scores."""
    if old.get('failed'):
        return {'failed': True, 'loss_clipped': CAP,
                'reason': old.get('reason', 'original_prediction_or_scoring_failure')}
    raw = old.get('raw')
    if not isinstance(raw, dict) or set(COMPONENTS)-set(raw):
        raise ValueError('Original raw components required; normalized loss is insufficient')
    raw = {key: float(raw[key]) for key in COMPONENTS}
    if not all(math.isfinite(v) for v in raw.values()):
        raise ValueError('Nonfinite original raw component')
    count = divisors['cell_count']
    if int(old.get('cell_count', count)) != count:
        raise ValueError('Raw component grid differs from reference distribution')
    weights = effective_weights(count)
    if 'effective_weights' in old and canonical_weights(old['effective_weights']) != weights:
        raise ValueError('Original effective weights differ from declared grid')
    scales = divisors['effective']
    if not all(math.isfinite(scales[k]) and scales[k] > 0 for k in COMPONENTS):
        raise ValueError('Invalid expected-error scales')
    normalized = {k: 0. if weights[k] == 0 else raw[k]/scales[k] for k in COMPONENTS}
    loss = sum(weights[k]*normalized[k] for k in COMPONENTS)
    return {'failed': False, 'raw': raw, 'effective_weights': weights,
            'normalized_components': normalized, 'normalized_unclipped': loss,
            'loss_clipped': float(np.clip(loss, 0., CAP)), 'cell_count': count,
            'n_draws': old.get('n_draws'), 'rankable': False}
