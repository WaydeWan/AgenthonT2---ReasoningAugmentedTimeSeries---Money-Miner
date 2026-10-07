"""Fit B0/B1 and the B2 inflation-trend correction on current-unit history."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import numpy as np
import pandas as pd
from .config import Config
from .io import Unit
from .targets import TargetLayout, modeling_values, target_layout, tick


@dataclass
class Fit:
    layout: TargetLayout
    anchor: np.ndarray
    drift: np.ndarray
    phi: np.ndarray
    previous_step: np.ndarray
    covariance: np.ndarray
    diagnostics: dict[str, Any]


def repair_covariance(cov: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Repair a correlation matrix and rescale to preserve intended marginal variances."""
    cov = (cov + cov.T) / 2
    sd = np.sqrt(np.maximum(np.diag(cov), floor**2))
    corr = np.nan_to_num(cov / np.outer(sd, sd), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)
    corr = (v * np.maximum(w, 1e-8)) @ v.T
    normalizer = np.sqrt(np.diag(corr))
    corr /= np.outer(normalizer, normalizer)
    return corr * np.outer(sd, sd)


def ewma_covariance(
    residuals: pd.DataFrame, config: Config, floor: np.ndarray
) -> tuple[np.ndarray, list[list[int]]]:
    """Estimate pairwise centered covariance without forward/backward filling missing data."""
    d = residuals.shape[1]
    cov = np.zeros((d, d))
    count = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i, d):
            mask = residuals.iloc[:, i].notna() & residuals.iloc[:, j].notna()
            pair = residuals.loc[mask].iloc[:, [i, j]].to_numpy()
            count[i, j] = count[j, i] = len(pair)
            if len(pair) < (2 if i == j else config.min_pair):
                value = floor[i] ** 2 if i == j else 0.0
            else:
                # Weight the aligned observation clock, retaining missing-position age.
                positions = np.flatnonzero(mask.to_numpy())
                weights = (
                    np.ones(len(pair))
                    if config.model == "B0"
                    else config.ewma_decay ** (len(residuals) - 1 - positions)
                )
                if weights.sum() < np.finfo(float).tiny:
                    weights = np.ones(len(pair))
                weights = weights / weights.sum()
                centered = pair - (weights[:, None] * pair).sum(axis=0)
                denom = 1 - (weights**2).sum()
                value = float(
                    np.sum(weights * centered[:, 0] * centered[:, 1])
                    / max(denom, 1e-12)
                )
            cov[i, j] = cov[j, i] = value
    cov = (
        1 - config.covariance_shrinkage
    ) * cov + config.covariance_shrinkage * np.diag(np.diag(cov))
    return repair_covariance(cov, floor), count.tolist()


def fit_numeric(unit: Unit, config: Config) -> Fit:
    config.validate()
    unit = replace(
        unit,
        histories={
            a: s.loc[
                s.index < pd.Timestamp(unit.asof) + pd.Timedelta(days=1)
            ].sort_index()
            for a, s in unit.histories.items()
        },
    )
    layout = target_layout(unit)
    if config.model == "B2" and layout.frequency != "daily":
        # CPI indices have multiplicative inflation, not a mean-reverting price level.
        layout.transforms[:] = [
            "log_level" if asset.upper() in {"CPI_ALL", "CPI_CORE", "CPIAUCSL", "CPILFESL"} and (unit.histories[asset] > 0).all() else transform
            for asset, transform in zip(unit.assets, layout.transforms)
        ]
    residuals: dict[str, pd.Series] = {}
    anchors, drifts, phis, previous, floors = [], [], [], [], []
    stats: dict[str, Any] = {}
    employment_updates: dict[int, tuple[float, float]] = {}
    for i, asset in enumerate(unit.assets):
        source = unit.histories[asset]
        # Unit is also a public Python API; enforce the cutoff here as well as in the loader.
        source = source.loc[
            source.index < pd.Timestamp(unit.asof) + pd.Timedelta(days=1)
        ].sort_index()
        if source.empty:
            raise ValueError(f"Missing as-of anchor: {asset}")
        transform = layout.transforms[i]
        values = modeling_values(source, transform)
        ticks = tick(pd.DatetimeIndex(values.index), layout.frequency)
        gaps = np.r_[0, np.diff(ticks)]
        if len(set(ticks)) != len(ticks):
            raise ValueError(f"Duplicate observations in native periods: {asset}")
        steps = (
            values.copy()
            if transform == "simple_to_log_return"
            else values.diff().where(gaps == 1)
        )
        usable = steps.dropna().tail(
            config.window_daily
            if layout.frequency == "daily"
            else config.window_low_frequency
        )
        fallbacks = []
        if len(usable) < 2:
            raise ValueError(
                f"Insufficient observed increments for {asset}; no invented absolute price"
            )
        if len(usable) < config.min_pair:
            fallbacks.append("Small sample: off-diagonal covariance may be zero")
        anchor = 0.0 if transform == "simple_to_log_return" else float(values.iloc[-1])
        floor = max(
            config.absolute_sd_floor, config.relative_sd_floor * max(abs(anchor), 1.0)
        )
        sd = float(usable.std())
        if sd < floor:
            fallbacks.append(
                "Constant/nearly constant series: engineering variance floor"
            )
        mean = float(usable.mean())
        drift = (
            0.0
            if config.model == "B0"
            else float(
                np.clip(
                    mean * config.drift_shrinkage,
                    -config.drift_cap_sd * sd,
                    config.drift_cap_sd * sd,
                )
            )
        )
        phi = 0.0
        if config.model == "B2" and asset.upper().startswith("CPI") and transform == "log_level":
            recent_mean = float(usable.tail(config.inflation_recent_periods).mean())
            weight = config.inflation_recent_weight
            drift = weight * recent_mean + (1 - weight) * mean
        elif layout.frequency != "daily":
            # A simple AR(1) on native-frequency increments, with a shrunk intercept.
            pairs = (
                pd.DataFrame({"lag": steps.shift(1), "step": steps})
                .dropna()
                .tail(config.window_low_frequency)
            )
            if len(pairs) >= config.min_ar and float(pairs.lag.var()) > floor**2:
                slope = float(
                    np.cov(pairs.lag, pairs.step, ddof=1)[0, 1] / pairs.lag.var()
                )
                phi = float(np.clip(slope, -0.8, 0.8) * config.ar_shrinkage)
                drift = mean * config.drift_shrinkage * (1 - phi)
            else:
                fallbacks.append(
                    "Low-frequency AR sample inadequate: random-walk fallback"
                )
        if (config.model == "B2" and layout.frequency == "monthly" and asset == "NFP"
                and transform == "level" and config.employment_trend != "shrunk"):
            gate = 1.0 if config.employment_trend == "full" else (
                float(np.clip(usable.tail(6).mean() / max(mean, sd / np.sqrt(len(usable))), 0, 1))
                if mean > 0 and len(usable) >= 6 else 0.0)
            full_intercept = mean * (1 - phi)
            uncapped_reference = config.drift_shrinkage * mean * (1 - phi)
            if gate == 0:
                # A closed gate must retain the actual reference, including its cap.
                intercept = drift
            elif drift == uncapped_reference:
                # Preserve the existing arithmetic for the established AR path.
                intercept = (config.drift_shrinkage + (1 - config.drift_shrinkage) * gate) * mean * (1 - phi)
            else:
                intercept = drift + gate * (full_intercept - drift)
            employment_updates[i] = (intercept, gate)
        # Keep the innovation estimate identical to the shrunk reference model.
        resid = steps - drift - phi * steps.shift(1) if phi else steps - drift
        resid = resid.reindex(usable.index).dropna()
        # Calendar-period alignment is necessary for monthly series with different date labels.
        resid.index = tick(pd.DatetimeIndex(resid.index), layout.frequency)
        residuals[asset] = resid
        last_step = float(steps.iloc[-1]) if pd.notna(steps.iloc[-1]) else 0.0
        if pd.isna(steps.iloc[-1]) and len(steps) > 1:
            fallbacks.append(
                "Current anchor is separated from history; no gap difference used"
            )
        anchors.append(anchor)
        drifts.append(drift)
        phis.append(phi)
        previous.append(last_step)
        floors.append(floor)
        stats[asset] = {
            "panel_id": unit.panel_ids[asset],
            "transform": transform,
            "anchor_observation": str(source.index[-1].date()),
            "anchor_native": float(source.iloc[-1]),
            "anchor_model": anchor,
            "fit_start": str(usable.index[0].date()),
            "fit_end": str(usable.index[-1].date()),
            "n_increments": len(usable),
            "gaps_removed": int((gaps > 1).sum()),
            "large_gaps_removed": int((gaps > config.gap_business_days).sum()),
            "sample_mean_step": mean,
            "sample_sd_step": sd,
            "drift_intercept": drift,
            "ar_phi": phi,
            "previous_step": last_step,
            "variance_floor_sd": floor,
            "steps_by_horizon": {
                str(h): int(layout.endpoints[i, j] - layout.anchor_ticks[i])
                for j, h in enumerate(unit.horizons)
            },
            "fallbacks": fallbacks,
        }
    matrix = pd.DataFrame(residuals).sort_index()
    covariance, counts = ewma_covariance(matrix, config, np.asarray(floors))
    for index, (intercept, gate) in employment_updates.items():
        asset_stats = stats[unit.assets[index]]
        asset_stats["employment_trend"] = {"mode": config.employment_trend, "gate": gate,
                                            "reference_intercept": drifts[index], "applied_intercept": intercept,
                                            "recent_periods": 6, "event_date_rules": False}
        drifts[index] = intercept
        asset_stats["drift_intercept"] = intercept
    diagnostics = {
        "frequency": layout.frequency,
        "assets": stats,
        "warnings": layout.warnings,
        "pair_observation_counts": counts,
        "covariance": covariance.tolist(),
        "covariance_min_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        "covariance_shrinkage": config.covariance_shrinkage,
        "ewma_decay": config.ewma_decay if config.model != "B0" else None,
        "student_df": config.student_df if config.model != "B0" and config.innovation_family != "gaussian" else None,
        "innovation_family": "gaussian" if config.model == "B0" else config.innovation_family,
        "parameter_origin": "Fixed engineering defaults; B2 trend design assessed on retrospective development windows; fitted statistics use this unit only",
        "auxiliary_policy": "Own target history and current anchor preferred; no cross-unit data or synthetic anchor",
    }
    return Fit(
        layout,
        np.asarray(anchors),
        np.asarray(drifts),
        np.asarray(phis),
        np.asarray(previous),
        covariance,
        diagnostics,
    )
