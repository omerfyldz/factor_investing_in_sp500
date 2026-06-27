"""Load/save portfolio state between monthly rebalances."""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from live.config import STATE_PATH


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"last_rebalance": None, "target_weights": {}}


def save_state(target_weights: dict[str, float]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_rebalance": date.today().isoformat(),
        "target_weights": {k: round(float(v), 6) for k, v in target_weights.items()},
    }
    with STATE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
