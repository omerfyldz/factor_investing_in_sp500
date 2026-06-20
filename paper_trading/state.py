"""Load and save portfolio state between monthly rebalances."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from config import STATE_PATH


def load_state() -> dict[str, Any]:
    """Return the persisted state dict, or an empty default if it doesn't exist."""
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "last_rebalance": None,
        "improved_4": {"holdings": []},
    }


def save_state(state: dict[str, Any]) -> None:
    """Persist state dict to JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)


def update_holdings(state: dict, strategy: str, holdings: list[str]) -> dict:
    """Return updated state with new holdings for a given strategy."""
    state = dict(state)
    state[strategy] = {"holdings": holdings}
    state["last_rebalance"] = date.today().isoformat()
    return state
