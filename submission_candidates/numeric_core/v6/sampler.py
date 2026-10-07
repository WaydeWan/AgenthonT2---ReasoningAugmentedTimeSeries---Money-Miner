"""Sample a shared calendar path with covariance-standardized elliptical innovations."""

from __future__ import annotations

import numpy as np
from .config import Config
from .numeric import Fit


def innovations(
    rng: np.random.Generator, draws: int, covariance: np.ndarray, df: float | None
) -> np.ndarray:
    normal = (
        rng.standard_normal((draws, len(covariance))) @ np.linalg.cholesky(covariance).T
    )
    if df is None:
        return normal
    if df <= 2:
        raise ValueError("Student-t covariance requires df > 2")
    # One radial shock per draw and time, shared across every asset. The covariance
    # equals covariance, while the corresponding t scale matrix is (df-2)/df times it.
    return normal * np.sqrt((df - 2) / rng.chisquare(df, size=draws))[:, None]


def sample_paths(
    fit: Fit, config: Config, multipliers: np.ndarray | None = None, *, path_gain: np.ndarray | None = None
) -> np.ndarray:
    multipliers = (
        np.ones(len(fit.anchor))
        if multipliers is None
        else np.asarray(multipliers, dtype=float)
    )
    if (
        multipliers.shape != fit.anchor.shape
        or not np.isfinite(multipliers).all()
        or np.any(multipliers <= 0)
    ):
        raise ValueError("Invalid positive scale multiplier")
    covariance = fit.covariance * np.outer(multipliers, multipliers)
    rng = np.random.default_rng(config.seed)
    family = "gaussian" if config.model == "B0" else config.innovation_family
    path_scale = (
        np.sqrt((config.student_df - 2) / rng.chisquare(config.student_df, size=config.draws))[:, None]
        if family == "path_student" else 1.0
    )
    values = np.tile(fit.anchor, (config.draws, 1))
    previous = np.tile(fit.previous_step, (config.draws, 1))
    result = np.empty((config.draws, *fit.layout.endpoints.shape))
    first = int(fit.layout.anchor_ticks.min()) + 1
    last = int(fit.layout.endpoints.max())
    if last - first > 10000:
        raise ValueError(
            "Required path exceeds V1 bound; stale/missing anchor metadata"
        )
    for period in range(first, last + 1):
        current_covariance = covariance
        shock_multiplier = 1.0
        if config.text_decay_periods > 0:
            # A positive diagonal D(t) preserves covariance PSD and common paths.
            elapsed = np.maximum(0, period - fit.layout.anchor_ticks - 1)
            current_multiplier = np.exp(np.log(multipliers) * 2 ** (-elapsed / config.text_decay_periods))
            # D L is the Cholesky factor of D Sigma D; row scaling avoids
            # recomputation and preserves unexposed assets bit-for-bit.
            current_covariance = fit.covariance
            shock_multiplier = current_multiplier
        shock = innovations(
            rng,
            config.draws,
            current_covariance,
            config.student_df if family == "student" else None,
        ) * path_scale * shock_multiplier
        if path_gain is not None:
            if path_gain.shape != (config.draws, len(fit.anchor)) or not np.isfinite(path_gain).all() or np.any(path_gain <= 0):
                raise ValueError("Invalid experimental path gains")
            shock *= path_gain
        active = period > fit.layout.anchor_ticks
        increment = fit.drift + fit.phi * previous + shock
        values[:, active] += increment[:, active]
        previous[:, active] = increment[:, active]
        for ai, hi in np.argwhere(fit.layout.endpoints == period):
            result[:, ai, hi] = (
                np.exp(values[:, ai])
                if fit.layout.transforms[ai] == "log_level"
                else values[:, ai]
            )
    if not np.isfinite(result).all():
        raise ValueError("Non-finite joint forecast")
    return result
