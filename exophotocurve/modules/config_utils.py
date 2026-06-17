"""JSON configuration utilities."""

from __future__ import annotations

import json
from typing import Dict

from .constants import CONFIG_KEYS
from .sg_loader import sg


def save_config(path: str, values: Dict[str, object]) -> None:
    """Save GUI settings to a JSON file."""
    config = {key: values.get(key) for key in CONFIG_KEYS}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def load_config(path: str) -> Dict[str, object]:
    """Load GUI settings from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_config(window: sg.Window, config: Dict[str, object]) -> None:
    """Apply loaded settings to the GUI."""
    for key, value in config.items():
        if value is None:
            continue
        if key in window.AllKeysDict:
            try:
                window[key].update(value=value)
            except Exception:
                pass
