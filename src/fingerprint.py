"""Config fingerprinting: a normalized, machine-independent view of a Config.

Checkpoints store one of these so a resume can verify the config has not drifted. Kept separate
from the training loop because it changes for its own reasons (what counts as "the same config"),
and so a future analysis script can compare fingerprints without importing torch or the loop.

Config is imported under TYPE_CHECKING only -- the type hint must not drag config.py (and with it
torch/numpy/yaml) into a module that is otherwise pure dict manipulation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config import Config


class _Absent:
    """Sentinel for a key present in one fingerprint but not the other."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<absent>"


_ABSENT = _Absent()

# Cap on how many differing keys are spelled out before the diff is truncated.
_MAX_DIFF_ENTRIES = 12


def _config_fingerprint(config: "Config") -> dict:
    """Normalized, machine-independent view of a Config, for storing in checkpoints.

    Runs move between Kaggle, a rented Vast.ai box and a local machine, so a naive
    dataclasses.asdict() comparison would report false drift. Two normalizations:
      * source_path is dropped entirely (machine-specific provenance; already compare=False).
      * data.path is reduced to its basename -- config.py anchors it to an absolute repo-root
        path, so the absolute form differs per machine while the dataset is identical.
    Everything else is left intact. The same helper builds both the stored and the current
    value, so the two are always constructed identically.
    """
    fingerprint = dataclasses.asdict(config)
    fingerprint.pop("source_path", None)
    data = fingerprint.get("data")
    if isinstance(data, dict) and "path" in data:
        data["path"] = Path(data["path"]).name
    return fingerprint


def _flatten(mapping: dict, prefix: str = "") -> Dict[str, object]:
    """Flatten nested dicts to dotted paths, e.g. train.optimizer.name -> value."""
    flat: Dict[str, object] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _fingerprint_diff(stored: dict, current: dict):
    """Differing keys as [(dotted_path, stored_value, current_value), ...], sorted by path."""
    flat_stored, flat_current = _flatten(stored), _flatten(current)
    diff = []
    for key in sorted(set(flat_stored) | set(flat_current)):
        old = flat_stored.get(key, _ABSENT)
        new = flat_current.get(key, _ABSENT)
        if old != new:
            diff.append((key, old, new))
    return diff


def _format_fingerprint_diff(diff) -> str:
    """Render a diff as `  dotted.key: old -> new` lines, truncated if very long."""
    lines = [f"  {key}: {old!r} -> {new!r}" for key, old, new in diff[:_MAX_DIFF_ENTRIES]]
    if len(diff) > _MAX_DIFF_ENTRIES:
        lines.append(f"  ... and {len(diff) - _MAX_DIFF_ENTRIES} more")
    return "\n".join(lines)
