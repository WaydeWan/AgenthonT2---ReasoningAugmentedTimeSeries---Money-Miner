"""Current-unit raw vintages, with separate feature and label cutoffs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v6.targets import modeling_values, target_layout


class Unsupported(ValueError):
    """A predeclared native coverage failure; it must remain in the roster."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None):
        self.reason = reason
        self.detail = dict(detail or {})
        super().__init__(reason)


def end_of_day(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize() + pd.Timedelta(days=1)


@dataclass
class AssetTape:
    asset: str
    transform: str
    rows: pd.DataFrame
    vintage_policy: str

    def snapshot(self, cutoff) -> pd.DataFrame:
        end = end_of_day(cutoff)
        selected = self.rows.loc[(self.rows.date < end) & (self.rows.available_at < end)]
        selected = selected.sort_values(["date", "available_at"], kind="stable")
        if selected.duplicated(["date", "available_at"]).any():
            if selected.groupby(["date", "available_at"]).value.nunique().gt(1).any():
                raise Unsupported("conflicting_same_vintage", {"asset": self.asset})
        return selected.drop_duplicates("date", keep="last").reset_index(drop=True)

    def values(self, rows: pd.DataFrame) -> np.ndarray:
        try:
            result = modeling_values(pd.Series(rows.value.to_numpy()), self.transform).to_numpy()
        except ValueError as exc:
            raise Unsupported("invalid_modeling_transform", {"asset": self.asset, "error": str(exc)}) from exc
        if not np.isfinite(result).all():
            raise Unsupported("nonfinite_modeling_values", {"asset": self.asset})
        return result


def load_tapes(unit, return_lag_days: int = 30):
    """Reload raw panel versions from this unit, never its sibling directories.

    A history-only synthetic adapter is supported. It is not accepted when input
    diagnostics indicate that vintage rows have already been discarded.
    """
    bounded_histories = {a: s.loc[pd.DatetimeIndex(s.index) < end_of_day(unit.asof)]
                         for a, s in unit.histories.items()}
    if any(not len(bounded_histories.get(a, [])) for a in unit.assets):
        raise Unsupported("asset_has_no_cutoff_history")
    try:
        layout = target_layout(replace(unit, histories=bounded_histories))
    except ValueError as exc:
        raise Unsupported("target_layout_unresolved", {"error": str(exc)}) from exc
    if layout.frequency != "daily":
        raise Unsupported("native_frequency_not_implemented", {"frequency": layout.frequency})
    root = Path(unit.root).resolve()
    files = sorted((root / "panels").glob("*.parquet")) or sorted(root.glob("*.parquet"))
    pieces = []
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise Unsupported("panel_path_outside_current_unit", {"file": path.name})
        frame = pd.read_parquet(path)
        if "asset_id" in frame and "asset" not in frame:
            frame = frame.rename(columns={"asset_id": "asset"})
        if not {"date", "asset", "value"} <= set(frame):
            raise Unsupported("raw_panel_schema", {"file": path.name})
        frame = frame[frame.asset.isin(unit.assets)].copy()
        frame["date"] = pd.to_datetime(frame.date, utc=True).dt.tz_localize(None).dt.normalize()
        frame = frame[frame.date < end_of_day(unit.asof)].copy()
        frame["available_at"] = (pd.to_datetime(frame.available_at, utc=True).dt.tz_localize(None)
                                  if "available_at" in frame else pd.NaT)
        pieces.append(frame[["date", "asset", "value", "available_at"]])
    source = "raw_current_unit_panels"
    if pieces:
        frame = pd.concat(pieces, ignore_index=True)
    else:
        if unit.input_diagnostics.get("superseded_vintages_removed", 0):
            raise Unsupported("raw_vintages_unavailable")
        source = "history_only_no_vintage_metadata"
        frame = pd.concat([pd.DataFrame({"date": s.index, "value": s.to_numpy(), "asset": a,
                                        "available_at": pd.NaT})
                           for a, s in unit.histories.items() if a in unit.assets], ignore_index=True)
        frame["date"] = pd.to_datetime(frame.date).dt.tz_localize(None).dt.normalize()
        frame = frame[frame.date < end_of_day(unit.asof)].copy()
    tapes = []
    for asset, transform in zip(unit.assets, layout.transforms):
        rows = frame[frame.asset == asset].copy()
        if rows.empty:
            raise Unsupported("asset_has_no_raw_history", {"asset": asset})
        missing = rows.available_at.isna()
        lag = return_lag_days if transform == "simple_to_log_return" else 0
        rows.loc[missing, "available_at"] = rows.loc[missing, "date"] + pd.Timedelta(days=lag)
        policy = (f"{source}; missing_available_at_proxy={lag}_calendar_days; "
                  "features_use_each_origin_vintage; labels_use_latest_outer_cutoff_vintage")
        rows = rows[rows.available_at < end_of_day(unit.asof)].copy()
        rows["value"] = pd.to_numeric(rows.value, errors="raise")
        # Missing data remain missing; neither forward nor future backfill is used.
        rows = rows.dropna(subset=["value"])
        if np.isinf(rows.value).any():
            raise Unsupported("infinite_raw_value", {"asset": asset})
        declared = unit.spec.get("targets", {}).get("as_of_level", {}).get(asset)
        if declared is not None and transform != "simple_to_log_return":
            at_asof = rows[rows.date == pd.Timestamp(unit.asof)]
            if at_asof.empty:
                rows = pd.concat([rows, pd.DataFrame({"asset": [asset], "date": [pd.Timestamp(unit.asof)],
                      "available_at": [pd.Timestamp(unit.asof)], "value": [float(declared)]})], ignore_index=True)
        rows = rows.sort_values(["date", "available_at", "value"], kind="stable").reset_index(drop=True)
        tapes.append(AssetTape(asset, transform, rows, policy))
    return layout, tapes
