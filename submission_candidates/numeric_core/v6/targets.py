"""Respect native target units and explicit low-frequency observation periods."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from qfbench2_track_forecasting.horizons import monthly_horizon_steps
from qfbench2_track_forecasting.targets import log_return_steps
from .io import Unit


@dataclass
class TargetLayout:
    frequency: str
    anchor_ticks: np.ndarray
    endpoints: np.ndarray
    transforms: list[str]
    warnings: list[str]


def cadence(series: pd.Series) -> str:
    if len(series) < 3:
        return "unknown"
    dates = pd.DatetimeIndex(series.index)
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    median = float(np.median(gaps))
    months = dates.to_period("M").asi8
    # First-day and last-day labels can denote the same monthly clock.
    if len(np.unique(months)) == len(months) and np.median(np.diff(months)) == 1:
        return "monthly"
    quarters = dates.to_period("Q").asi8
    if len(np.unique(quarters)) == len(quarters) and np.median(np.diff(months)) == 3:
        return "quarterly"
    if 0 < median <= 4:
        return "daily"
    return "unknown"


def tick(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "daily":
        return np.busday_count(
            np.datetime64("1970-01-01"), index.values.astype("datetime64[D]")
        )
    return index.to_period("M" if frequency == "monthly" else "Q").asi8


def target_layout(unit: Unit) -> TargetLayout:
    histories = [unit.histories[a] for a in unit.assets]
    detected = [cadence(s) for s in histories]
    declared = unit.card["targets"].get(
        "target_frequency",
        unit.card.get("metadata", {}).get("target_frequency", "daily"),
    )
    warnings: list[str] = []
    freq = declared
    explicit = any(
        "observation_periods" in src.get("targets", {})
        or any("observation_period" in q for q in src.get("questions", []))
        for src in (unit.card, unit.spec)
    )
    if declared == "monthly" and all(x == "daily" for x in detected) and not explicit:
        freq = "daily"
        warnings.append(
            "Legacy monthly declaration with dense daily targets: official daily compatibility fallback"
        )
    if freq not in {"daily", "monthly", "quarterly"} or any(
        d not in {freq, "unknown"} for d in detected
    ):
        raise ValueError(
            f"Target cadence inconsistent with metadata: {declared}, {detected}"
        )
    if freq != "daily" and any(d == "unknown" for d in detected):
        raise ValueError(
            "Insufficient native-frequency observations to resolve low-frequency cadence"
        )
    anchors = np.array(
        [tick(pd.DatetimeIndex([s.index[-1]]), freq)[0] for s in histories]
    )
    if freq == "daily":
        asof_tick = tick(pd.DatetimeIndex([unit.asof]), freq)[0]
        endpoints = np.tile(asof_tick + np.asarray(unit.horizons), (len(histories), 1))
        if unit.target_type == "log_return":
            anchors[:] = asof_tick
    else:
        if unit.target_type not in {"level", "yield"}:
            raise ValueError("Low-frequency cumulative-return semantics unavailable")
        last = {a: str(unit.histories[a].index[-1].date()) for a in unit.assets}
        if freq == "monthly":
            steps = monthly_horizon_steps(
                unit.assets,
                unit.horizons,
                last,
                asof=unit.asof,
                card=unit.card,
                forecast_spec=unit.spec,
            ).astype(int)
            endpoints = anchors[:, None] + steps
        else:
            # Quarter labels use the same explicit calendar-month contract, never h / 63.
            steps_month = monthly_horizon_steps(
                unit.assets,
                unit.horizons,
                last,
                asof=unit.asof,
                card=unit.card,
                forecast_spec=unit.spec,
            ).astype(int)
            base_month = np.array(
                [pd.Timestamp(last[a]).to_period("M").ordinal for a in unit.assets]
            )
            endpoints = (base_month[:, None] + steps_month) // 3
            if np.any(endpoints <= anchors[:, None]):
                raise ValueError(
                    "Quarterly target must follow the last available observation"
                )
    transforms = []
    for a in unit.assets:
        panel = unit.panel_ids[a].lower()
        fx_unit = (
            "per_usd" in unit.card["targets"].get("value_unit", "").lower()
            or "fx" in panel
            or "em_transfer" in panel
        )
        transform = (
            "simple_to_log_return"
            if unit.target_type == "log_return"
            else "log_level"
            if fx_unit
            else "level"
        )
        if transform == "log_level" and (unit.histories[a] <= 0).any():
            raise ValueError(f"FX log modeling requires positive spot levels: {a}")
        transforms.append(transform)
    return TargetLayout(freq, anchors, endpoints, transforms, warnings)


def modeling_values(series: pd.Series, transform: str) -> pd.Series:
    if transform == "log_level":
        return np.log(series)
    if transform == "simple_to_log_return":
        return pd.Series(log_return_steps(series.to_numpy()), index=series.index)
    return series.copy()
