"""Small persistent user-preferences helper for ExoPhotoCurve.

This module stores only stable user preferences. It is intentionally not a
session/recipe system: files, dynamic columns, masks, fitted models and other
analysis products should never be persisted here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

APP_NAME = "ExoPhotoCurve"
PREFERENCES_VERSION = 1


def user_config_dir() -> Path:
    """Return the per-user configuration directory for ExoPhotoCurve."""
    if os.name == "nt":
        root = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / APP_NAME
        return Path.home() / "AppData" / "Roaming" / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    root = os.environ.get("XDG_CONFIG_HOME")
    if root:
        return Path(root) / APP_NAME
    return Path.home() / ".config" / APP_NAME


def preferences_path() -> Path:
    """Return the JSON preferences path."""
    return user_config_dir() / "user_preferences.json"


def load_preferences() -> dict[str, Any]:
    """Load the full preferences dictionary, returning an empty structure on failure."""
    path = preferences_path()
    if not path.exists():
        return {"version": PREFERENCES_VERSION, "sections": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": PREFERENCES_VERSION, "sections": {}}
    if not isinstance(data, dict):
        return {"version": PREFERENCES_VERSION, "sections": {}}
    sections = data.get("sections")
    if not isinstance(sections, dict):
        # Accept very old/experimental layouts where sections were top-level.
        sections = {k: v for k, v in data.items() if isinstance(v, dict) and k != "version"}
    return {"version": int(data.get("version", PREFERENCES_VERSION) or PREFERENCES_VERSION), "sections": sections}


def save_preferences(data: Mapping[str, Any]) -> Path:
    """Write the full preferences dictionary and return the saved path."""
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["version"] = PREFERENCES_VERSION
    payload.setdefault("sections", {})
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def reset_preferences() -> bool:
    """Delete the user-preferences file if it exists."""
    path = preferences_path()
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def get_section(section: str) -> dict[str, Any]:
    """Return one preferences section."""
    data = load_preferences()
    value = data.get("sections", {}).get(section, {})
    return dict(value) if isinstance(value, dict) else {}


def update_section(section: str, values: Mapping[str, Any]) -> Path:
    """Update one preferences section with JSON-safe values."""
    data = load_preferences()
    sections = data.setdefault("sections", {})
    current = sections.get(section, {})
    if not isinstance(current, dict):
        current = {}
    clean: dict[str, Any] = {}
    for key, value in values.items():
        safe = _json_safe_value(value)
        if safe is not None:
            clean[str(key)] = safe
    current.update(clean)
    sections[section] = current
    return save_preferences(data)


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-safe scalar/list value, or None if it should not be stored."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            safe = _json_safe_value(item)
            if safe is not None:
                out.append(safe)
        return out
    return str(value)


def _window_value(window: Any, key: str) -> Any:
    try:
        return window[key].get()
    except Exception:
        return None


def collect_window_preferences(window: Any, keys: Iterable[str]) -> dict[str, Any]:
    """Collect whitelisted values from a PySimpleGUI window."""
    out: dict[str, Any] = {}
    for key in keys:
        value = _window_value(window, key)
        if value is not None:
            out[str(key)] = value
    return out


def save_window_preferences(window: Any, section: str, keys: Iterable[str]) -> Path:
    """Collect and save whitelisted values from a PySimpleGUI window."""
    return update_section(section, collect_window_preferences(window, keys))


def apply_preferences_to_window(window: Any, section: str, keys: Iterable[str]) -> dict[str, Any]:
    """Apply a whitelisted section to an existing PySimpleGUI window.

    Invalid or obsolete keys are ignored, so old preference files do not break
    new versions of the GUI.
    """
    allowed = set(keys)
    prefs = get_section(section)
    applied: dict[str, Any] = {}
    for key, value in prefs.items():
        if key not in allowed:
            continue
        try:
            window[key].update(value=value)
            applied[key] = value
        except Exception:
            try:
                window[key].update(values=value)
                applied[key] = value
            except Exception:
                continue
    return applied
