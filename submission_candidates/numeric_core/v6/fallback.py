"""Conservative numeric recovery after the input contract has been validated."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .numeric import Fit
from .targets import modeling_values, target_layout, tick


def recoverable(error: Exception) -> bool:
    """Never reinterpret invalid cutoffs, target grids, anchors or source records."""
    if isinstance(error, (np.linalg.LinAlgError, FloatingPointError, OverflowError)):
        return True
    return isinstance(error, ValueError) and str(error).startswith((
        "Insufficient observed increments for ", "Non-finite joint forecast"))


def diagonal_fit(unit, config, error: Exception) -> Fit:
    """Zero drift and independent Gaussian increments; use real current anchors."""
    unit = replace(unit, histories={a: s.loc[s.index < pd.Timestamp(unit.asof) + pd.Timedelta(days=1)].sort_index()
                                   for a, s in unit.histories.items()})
    layout = target_layout(unit)
    if config.model == "B2" and layout.frequency != "daily":
        layout.transforms[:] = ["log_level" if a.upper() in {"CPI_ALL", "CPI_CORE", "CPIAUCSL", "CPILFESL"}
                                and (unit.histories[a] > 0).all() else t
                                for a, t in zip(unit.assets, layout.transforms)]
    anchors, scales, statistics = [], [], {}
    for asset, transform in zip(unit.assets, layout.transforms):
        values = modeling_values(unit.histories[asset], transform)
        clock = tick(pd.DatetimeIndex(values.index), layout.frequency)
        step = values if transform == "simple_to_log_return" else values.diff().where(np.r_[False, np.diff(clock) == 1])
        step = step[np.isfinite(step)].tail(config.window_daily if layout.frequency == "daily" else config.window_low_frequency)
        anchor = 0.0 if transform == "simple_to_log_return" else float(values.iloc[-1])
        floor = max(config.absolute_sd_floor, config.relative_sd_floor * max(abs(anchor), 1.0))
        if len(step) >= 2:
            scale = float(np.median(np.abs(step - np.median(step))) * 1.4826)
            if scale <= floor:
                scale = float(np.max(np.abs(step - np.median(step))))
        elif len(step) == 1:
            scale = abs(float(step.iloc[0]))
        else:
            scale = floor
        scale = max(scale, floor)
        if not np.isfinite(anchor) or not np.isfinite(scale) or scale > np.sqrt(np.finfo(float).max):
            raise ValueError("Fallback cannot represent the supplied native anchor or variance")
        anchors.append(anchor)
        scales.append(scale)
        statistics[asset] = {"anchor_observation": str(values.index[-1].date()), "anchor_native": float(unit.histories[asset].iloc[-1]),
                             "transform": transform, "usable_increments": len(step), "fallback_step_sd": scale}
    count = len(anchors)
    covariance = np.diag(np.square(scales))
    return Fit(layout, np.asarray(anchors), np.zeros(count), np.zeros(count), np.zeros(count), covariance,
               {"frequency": layout.frequency, "assets": statistics, "covariance": covariance.tolist(),
                "fallback": {"active": True, "method": "B0_DIAGONAL_GAUSSIAN", "trigger_type": type(error).__name__,
                             "trigger": str(error), "policy": "Validated current anchors; zero drift; no invented input records"}})
