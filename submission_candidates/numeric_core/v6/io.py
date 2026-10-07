"""In-memory contract only; production file reading remains in the V8 interface.

The runtime accepts this dataclass or the structurally identical t2_rase.io.Unit.
No raw panels, labels, cached forecasts, or reference scales are packaged here.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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
    def assets(self):
        return self.card['targets']['asset_ids']

    @property
    def horizons(self):
        return self.card['targets']['horizons']

    @property
    def target_type(self):
        return self.card['targets'].get('target_type', 'level')

    @property
    def unit_id(self):
        return self.card['task']['id']
