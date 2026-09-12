"""config_loader — YAML config as the single source of truth.

All tunable parameters (calibration matrices, solver knobs, mode defaults,
limits, PID gains, serial settings) live in ``config.yaml`` at the repo root.
GUI widgets bind to dotted paths (``solver.lambda``, ``modes.mode_a.freq_default``)
and edits write back through this loader.

Migration: on first run, if ``config.yaml`` doesn't exist but the legacy
``calibration.json`` does, the loader seeds a fresh config from the example
template and folds the JSON's ``coil_gains`` + ``channel_map`` into it.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any, Callable

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")
EXAMPLE_CONFIG_PATH = os.path.join(REPO_ROOT, "config_example.yaml")
LEGACY_CALIBRATION_PATH = os.path.join(REPO_ROOT, "calibration.json")


def _deep_get(data: dict, path: str, default: Any = None) -> Any:
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _deep_set(data: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


class Config:
    """Mutable in-memory config with dotted-path access and change callbacks.

    Callers register with ``on_change(path_prefix, callback)``. Any set() on
    a matching path fires the callback with (path, new_value). Empty prefix
    receives every change.
    """

    def __init__(self, path: str = DEFAULT_CONFIG_PATH):
        self.path = path
        self._data: dict = {}
        self._listeners: list[tuple[str, Callable[[str, Any], None]]] = []
        self._dirty = False

    # ---- I/O --------------------------------------------------------

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self._data = yaml.safe_load(f) or {}
        elif os.path.exists(EXAMPLE_CONFIG_PATH):
            with open(EXAMPLE_CONFIG_PATH, "r") as f:
                self._data = yaml.safe_load(f) or {}
            self._migrate_legacy_calibration()
            self.save()
        else:
            self._data = {}
        self._dirty = False

    def save(self) -> None:
        """Atomic write via temp file + rename. Keeps prior version as .bak."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".config-", suffix=".yaml.tmp", dir=os.path.dirname(self.path) or "."
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                yaml.safe_dump(self._data, f, sort_keys=False, default_flow_style=None)
            if os.path.exists(self.path):
                shutil.copy2(self.path, self.path + ".bak")
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        self._dirty = False

    def _migrate_legacy_calibration(self) -> None:
        if not os.path.exists(LEGACY_CALIBRATION_PATH):
            return
        try:
            with open(LEGACY_CALIBRATION_PATH, "r") as f:
                legacy = json.load(f)
        except (OSError, ValueError):
            return
        gains = legacy.get("coil_gains")
        if isinstance(gains, list) and len(gains) == 6:
            _deep_set(self._data, "calibration.per_coil_gains",
                      [float(x) for x in gains])
        cmap = legacy.get("channel_map")
        if isinstance(cmap, list) and len(cmap) == 6:
            _deep_set(self._data, "calibration.channel_map",
                      [int(x) for x in cmap])
        _deep_set(self._data, "calibration.date",
                  time.strftime("%Y-%m-%d") + " (migrated)")

    # ---- access -----------------------------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        return _deep_get(self._data, path, default)

    def set(self, path: str, value: Any) -> None:
        old = _deep_get(self._data, path, None)
        if old == value:
            return
        _deep_set(self._data, path, value)
        self._dirty = True
        for prefix, cb in list(self._listeners):
            if path == prefix or path.startswith(prefix + ".") or prefix == "":
                try:
                    cb(path, value)
                except Exception:
                    pass

    def on_change(self, path_prefix: str, callback: Callable[[str, Any], None]) -> None:
        self._listeners.append((path_prefix, callback))

    def is_dirty(self) -> bool:
        return self._dirty

    def as_dict(self) -> dict:
        return self._data

    # ---- convenience ------------------------------------------------

    def get_matrix(self, path: str) -> "np.ndarray":
        """Return a numeric list-of-lists as a numpy array. Raises if missing."""
        import numpy as np
        m = self.get(path)
        if m is None:
            raise KeyError(f"config path {path!r} is missing")
        return np.asarray(m, dtype=float)


# Module-level singleton — GUI code and solver both talk to the same instance.
_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
        _CONFIG.load()
    return _CONFIG


def reset_for_tests(path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Force a fresh singleton pointed at ``path``. Only used by tests."""
    global _CONFIG
    _CONFIG = Config(path)
    _CONFIG.load()
    return _CONFIG
