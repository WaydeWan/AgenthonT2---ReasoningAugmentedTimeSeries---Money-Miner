"""Load one unit, truncate before estimating, and write the official output contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
import json
import tomllib

import jsonschema
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from qfbench2_common.taskcard import schema_path
from qfbench2_track_forecasting.cutoff import trusted_asof, bind_metadata
from qfbench2_track_forecasting.grid import (
    grid_from_card,
    check_declared_grid,
    build_sample_matrix,
)
from qfbench2_track_forecasting.limits import DEFAULT_LIMITS, inspect_parquet


@dataclass
class Unit:
    root: Path
    card: dict[str, Any]
    spec: dict[str, Any]
    asof: str
    histories: dict[str, pd.Series]
    panel_ids: dict[str, str]
    input_diagnostics: dict[str, Any]

    @property
    def assets(self) -> list[str]:
        return self.card["targets"]["asset_ids"]

    @property
    def horizons(self) -> list[int]:
        return self.card["targets"]["horizons"]

    @property
    def target_type(self) -> str:
        return self.card["targets"].get("target_type", "level")

    @property
    def unit_id(self) -> str:
        return self.card["task"]["id"]


def read_unit(panels: Path, asof: str) -> Unit:
    if date.fromisoformat(asof).isoformat() != asof:
        raise ValueError("asof must be an ISO date")
    card_path = next(
        (p for p in (panels / "card.toml", panels.parent / "card.toml") if p.is_file()),
        None,
    )
    if card_path is None:
        raise ValueError("No card.toml in the current input unit")
    if card_path.is_symlink():
        raise ValueError("Symlinked card is outside the input file contract")
    root = card_path.parent.resolve()
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    if trusted_asof(card) != asof:
        raise ValueError("CLI cutoff differs from the trusted card")
    spec_path = root / "forecast_spec.json"
    if spec_path.is_symlink() or not spec_path.resolve().is_relative_to(root):
        raise ValueError("Forecast spec path escapes current unit")
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    if spec and spec.get("card_id", card["task"]["id"]) != card["task"]["id"]:
        raise ValueError("Card/spec unit mismatch")
    targets = card["targets"]
    for key in ("asset_ids", "horizons", "target_type"):
        if key in spec.get("targets", {}) and spec["targets"][key] != targets.get(key):
            raise ValueError(f"Card/spec target mismatch: {key}")
    if not targets["asset_ids"] or len(set(targets["asset_ids"])) != len(
        targets["asset_ids"]
    ):
        raise ValueError("Target assets must be unique and nonempty")
    if (
        not targets["horizons"]
        or any(type(h) is not int or h <= 0 for h in targets["horizons"])
        or len(set(targets["horizons"])) != len(targets["horizons"])
    ):
        raise ValueError("Invalid horizon grid")
    if targets.get("target_type") not in {"level", "log_return", "yield"}:
        raise ValueError("Unsupported target type")
    files = sorted(panels.glob("*.parquet"))
    if not files:
        files = sorted(root.glob("*.parquet"))
    if not files:
        raise ValueError("No panel files within this unit")
    histories: dict[str, pd.Series] = {}
    panel_ids: dict[str, str] = {}
    audit: dict[str, Any] = {
        "files": [],
        "future_rows_removed": 0,
        "unavailable_rows_removed": 0,
        "superseded_vintages_removed": 0,
        "duplicates_removed": 0,
        "missing_values_removed": 0,
    }
    pieces = []
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Panel path escapes current unit")
        df = pd.read_parquet(path)
        if "asset_id" in df and "asset" not in df:
            df = df.rename(columns={"asset_id": "asset"})
        if not {"date", "asset", "value"} <= set(df):
            raise ValueError(f"Unsupported panel schema in {path.name}")
        df = df.loc[df["asset"].isin(targets["asset_ids"])].copy()
        dates = pd.to_datetime(df["date"], utc=True, errors="raise")
        mask = dates < pd.Timestamp(asof, tz="UTC") + pd.Timedelta(days=1)
        audit["future_rows_removed"] += int((~mask).sum())
        df = df.loc[mask].copy()
        df["date"] = dates.loc[mask].dt.tz_localize(None).dt.normalize()
        # No deduplication, cadence estimate or numeric conversion precedes the cutoff.
        if "available_at" in df:
            available = pd.to_datetime(df.available_at, utc=True, errors="raise")
            known = available < pd.Timestamp(asof, tz="UTC") + pd.Timedelta(days=1)
            audit["unavailable_rows_removed"] += int((~known).sum())
            df = df.loc[known].copy()
            df["available_at"] = available.loc[known]
        else:
            df["available_at"] = pd.NaT
        df["value"] = pd.to_numeric(df["value"], errors="raise")
        if np.isinf(df.value).any():
            raise ValueError("Infinite panel value")
        audit["missing_values_removed"] += int(df.value.isna().sum())
        df = df.dropna(subset=["value"])
        df["panel_id"] = df["panel_id"].astype(str) if "panel_id" in df else path.stem
        pieces.append(df[["date", "asset", "value", "panel_id", "available_at"]])
        audit["files"].append(path.name)
    frame = pd.concat(pieces, ignore_index=True)
    for asset, rows in frame.groupby("asset", sort=True):
        if rows["available_at"].notna().any():
            selected = []
            for _, versions in rows.groupby("date", sort=False):
                if versions["available_at"].notna().all():
                    if versions.groupby("available_at").value.nunique().gt(1).any():
                        raise ValueError(f"Conflicting same-vintage observations: {asset}")
                    current = versions["available_at"].max()
                    keep = versions["available_at"] == current
                    audit["superseded_vintages_removed"] += int((~keep).sum())
                    versions = versions.loc[keep]
                selected.append(versions)
            rows = pd.concat(selected, ignore_index=True)
        if rows.groupby("date").value.nunique().gt(1).any():
            raise ValueError(
                f"Conflicting same-date observations: {asset}; vintage selection unavailable"
            )
        rows = rows.sort_values("date", kind="stable")
        before = len(rows)
        rows = rows.drop_duplicates("date", keep="last")
        audit["duplicates_removed"] += before - len(rows)
        histories[str(asset)] = rows.set_index("date").value.astype(float)
        panel_ids[str(asset)] = str(rows.panel_id.iloc[-1])
    missing = set(targets["asset_ids"]) - set(histories)
    if missing:
        raise ValueError(f"Missing necessary target anchor/history: {sorted(missing)}")
    declared_anchors = spec.get("targets", {}).get("as_of_level", {})
    for asset in targets["asset_ids"]:
        history = histories[asset]
        if asset in declared_anchors:
            anchor = float(declared_anchors[asset])
            if not np.isfinite(anchor):
                raise ValueError("Non-finite declared as-of anchor")
            if history.index[-1] == pd.Timestamp(asof):
                if not np.isclose(history.iloc[-1], anchor, rtol=1e-8, atol=1e-10):
                    raise ValueError("Panel/spec as-of anchor conflict")
            else:
                histories[asset] = pd.concat(
                    [history, pd.Series([anchor], index=[pd.Timestamp(asof)])]
                )
                audit.setdefault("explicit_spec_anchors", []).append(asset)
        elif (
            "transfer" in panel_ids[asset]
            and (pd.Timestamp(asof) - history.index[-1]).days > 10
        ):
            raise ValueError(f"Transfer series {asset} has no current absolute anchor")
    return Unit(root, card, spec, asof, histories, panel_ids, audit)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def required_draws(unit: Unit) -> int:
    """Resolve the supplied lower bound before allocating a forecast."""
    floors = [DEFAULT_LIMITS.min_draws,
              unit.card.get("scoring", {}).get("params", {}).get("n_draws_min", 200),
              unit.spec.get("n_draws_min", 200)]
    if any(type(n) is not int or n <= 0 for n in floors):
        raise ValueError("Invalid declared draw floor")
    floor = max(floors)
    if floor > DEFAULT_LIMITS.max_draws:
        raise ValueError("Declared draw floor exceeds the official ceiling")
    return floor


def validate_output(unit: Unit, output: Path) -> None:
    meta = json.loads((output.parent / "forecast_meta.json").read_text(encoding="utf-8"))
    schema = json.loads(schema_path("forecast.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(meta)
    # Do not use unlisted sidecar keys even though this schema revision permits them.
    if set(meta) - set(schema["properties"]):
        raise ValueError("Unlisted forecast metadata key")
    bind_metadata(meta, unit.card, unit_handle=unit.unit_id)
    check_declared_grid(meta, grid_from_card(unit.card))
    floor = required_draws(unit)
    if not floor <= meta["n_draws"] <= DEFAULT_LIMITS.max_draws:
        raise ValueError("Draw count violates card/contract")
    facts = inspect_parquet(output, what="forecast.parquet")
    build_sample_matrix(
        output, facts, grid_from_card(unit.card), meta["n_draws"], limits=DEFAULT_LIMITS
    )
    types = pq.read_schema(output)
    if types != pa.schema(
        [
            ("draw", pa.int32()),
            ("asset", pa.string()),
            ("horizon", pa.int32()),
            ("value", pa.float64()),
        ]
    ):
        raise ValueError("Forecast physical dtypes differ from the contract")
    if not (output.parent / "forecast_rationale.md").read_text(encoding="utf-8").strip():
        raise ValueError("Empty rationale")


def write_output(
    unit: Unit, samples: np.ndarray, output: Path, diagnostics: dict[str, Any]
) -> None:
    n, a, h = samples.shape
    if (a, h) != (len(unit.assets), len(unit.horizons)) or not np.isfinite(
        samples
    ).all():
        raise ValueError("Invalid forecast sample tensor")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "draw": pa.array(np.repeat(np.arange(n), a * h), type=pa.int32()),
            "asset": pa.array(np.tile(np.repeat(unit.assets, h), n), type=pa.string()),
            "horizon": pa.array(np.tile(unit.horizons, n * a), type=pa.int32()),
            "value": pa.array(samples.ravel(), type=pa.float64()),
        }
    )
    pq.write_table(table, output, compression="zstd")
    dump_json(
        output.parent / "forecast_meta.json",
        {
            "unit_id": unit.unit_id,
            "asof": unit.asof,
            "representation": "samples",
            "asset_ids": unit.assets,
            "horizons": unit.horizons,
            "n_draws": n,
            "target": unit.target_type,
            "units": unit.card["targets"].get("value_unit", "as stated in card"),
            "rationale": {
                "file": "forecast_rationale.md",
                "method": diagnostics["method"],
            },
        },
    )
    rationale = [
        f"# Forecast calculation record: {unit.unit_id}",
        f"Cutoff {unit.asof}; {n} joint sample rows; native target {unit.target_type}.",
        "Histories are bounded to the trusted input cutoff before estimation. Each sample row preserves the declared asset/horizon grid.",
        "The record below states the actual numerical branch, text features, fitting scope and fallback. Engineering validity is not predictive accuracy.",
        "```json", json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False), "```",
    ]
    (output.parent / "forecast_rationale.md").write_text("\n\n".join(rationale) + "\n", encoding="utf-8")
    validate_output(unit, output)
