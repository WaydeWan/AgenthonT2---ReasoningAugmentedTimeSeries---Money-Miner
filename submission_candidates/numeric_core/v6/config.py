"""Validated defaults; historical development evidence is not an unseen test."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import math
import json
import yaml


@dataclass(frozen=True)
class Config:
    model: str = "B2"
    mode: str = "numeric"
    draws: int = 20000
    seed: int = 0
    window_daily: int = 756
    window_low_frequency: int = 120
    ewma_decay: float = 0.995
    covariance_shrinkage: float = 0.20
    drift_shrinkage: float = 0.10
    drift_cap_sd: float = 0.10
    student_df: float = 7.0
    min_pair: int = 12
    min_ar: int = 24
    ar_shrinkage: float = 0.5
    gap_business_days: int = 5
    relative_sd_floor: float = 1e-6
    absolute_sd_floor: float = 1e-8
    beta_vol: float = 0.25
    multiplier_min: float = 0.8
    multiplier_max: float = 1.25
    min_relevance: float = 0.35
    min_evidence_quality: float = 0.5
    freshness_half_life_days: float = 30.0
    max_text_age_days: int = 120
    retrieval_days: int = 730
    max_documents: int = 6
    max_document_chars: int = 6000
    max_context_chars: int = 24000
    house_max_calls: int = 25
    house_max_output_tokens: int = 4000
    house_max_request_bytes: int = 48000
    house_max_response_bytes: int = 100000
    house_total_seconds: float = 180.0
    house_request_seconds: float = 30.0
    house_temperature: float = 0.3
    ablation: str = "T2"
    innovation_family: str = "path_student"
    inflation_recent_periods: int = 12
    inflation_recent_weight: float = 0.5
    employment_trend: str = "adaptive"
    text_version: str = "v4"
    text_decay_periods: float = 21.0
    numeric_variant: str = "M0_T7_MIX75"

    short_student_df: float = 7.0
    drift_uq_strength: float = 0.0
    research_variant: str = "B5"
    adaptive_origins: int = 12
    adaptive_draws: int = 500
    adaptive_seconds: float = 240.0
    text_location_enabled: bool = False
    text_location_coefficient: float = 0.05
    text_consensus_votes: int = 3
    text_verifier: bool = True
    text_per_asset: bool = False
    v7_method: str = "V6"
    v8_method: str = "OFF"

    def validate(self) -> Config:
        if self.v8_method != "OFF":
            raise ValueError('Use the fixed V8 release CLI; experimental v8_method is unavailable')
        if self.v7_method not in {"V6", "SDF4_SINGLE", "SDF4_SCALE", "SDF4_DRIFT_SCALE", "SDF4_NS", "SDF4_SCALE_NS", "SDF4_DRIFT_SCALE_NS"}:
            raise ValueError("Unknown V7 integration method")
        if self.v7_method != "V6" and (self.mode != "numeric" or self.model != "B2" or self.research_variant != "B5" or self.numeric_variant != "M0_T7_MIX75" or self.ewma_decay != .995 or self.student_df != 7 or self.short_student_df != 7 or self.innovation_family != "path_student"):
            raise ValueError("V7 integration requires the frozen numeric V6 parent")
        if self.short_student_df <= 2 or self.drift_uq_strength < 0:
            raise ValueError("Invalid short-branch uncertainty")
        if not 2 <= self.adaptive_origins <= 32 or not 200 <= self.adaptive_draws <= 2000 or self.adaptive_seconds <= 0:
            raise ValueError("Invalid within-unit selection budget")
        if not 1 <= self.text_consensus_votes <= 12 or not 0 <= self.text_location_coefficient <= 0.15:
            raise ValueError("Invalid text ensemble controls")
        if not 0 <= self.house_temperature <= 2:
            raise ValueError("Invalid House temperature")
        if self.numeric_variant not in {"V4", "M0_T7_MIX75", "M0_T4"}:
            raise ValueError("Unknown numeric runtime variant")
        if self.model not in {"B0", "B1", "B2"} or self.mode not in {"numeric", "text_vol"}:
            raise ValueError("Unknown model or mode")
        if self.innovation_family not in {"gaussian", "student", "path_student"}:
            raise ValueError("Unknown innovation family")
        if self.text_version not in {"v1", "v2", "v4"}:
            raise ValueError("Unknown text version")
        if self.employment_trend not in {"shrunk", "full", "adaptive"}:
            raise ValueError("Unknown employment trend policy")
        if self.inflation_recent_periods < 3 or not 0 <= self.inflation_recent_weight <= 1 or self.text_decay_periods < 0:
            raise ValueError("Invalid V2 trend or text decay")
        if self.ablation not in {"T1", "T2"}:
            raise ValueError("Unknown text ablation")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite config: {name}")
        if not 200 <= self.draws <= 20000 or self.seed < 0 or self.student_df <= 2:
            raise ValueError("Invalid draws, seed or Student-t degrees of freedom")
        for name in (
            "ewma_decay",
            "covariance_shrinkage",
            "drift_shrinkage",
            "ar_shrinkage",
            "min_relevance",
            "min_evidence_quality",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if (
            not 0 < self.ewma_decay < 1
            or not 0 < self.multiplier_min <= 1 <= self.multiplier_max
        ):
            raise ValueError("Invalid decay or adjustment bounds")
        if (
            not 1 <= self.house_max_calls <= 25
            or not 1 <= self.house_max_output_tokens <= 4000
        ):
            raise ValueError("House budget exceeds this release or official bounds")
        for name in (
            "window_daily",
            "window_low_frequency",
            "min_pair",
            "min_ar",
            "gap_business_days",
            "relative_sd_floor",
            "absolute_sd_floor",
            "freshness_half_life_days",
            "max_text_age_days",
            "retrieval_days",
            "max_documents",
            "max_document_chars",
            "max_context_chars",
            "house_total_seconds",
            "house_request_seconds",
            "house_max_request_bytes",
            "house_max_response_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.beta_vol < 0 or self.drift_cap_sd < 0:
            raise ValueError("Negative scale configuration")
        return self


def load_config(path: Path | None = None, *, defaults: dict[str, Any] | None = None, **overrides: Any) -> Config:
    data = (json.loads(path.read_text()) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text())) if path else {}
    if data is None:
        data = {}
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(Config)}:
        raise ValueError("Configuration contains unknown keys")
    data = {**(defaults or {}), **data, **{k: v for k, v in overrides.items() if v is not None}}
    return Config(**data).validate()
