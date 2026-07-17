"""Inspect the RadioML 2018.01A HDF5 file and report its real structure.

Callable both ways with identical output:

    # notebook
    from scripts.inspect_dataset import inspect_dataset
    summary = inspect_dataset(config="baseline")     # config name or path
    summary = inspect_dataset(path="data/GOLD_XYZ_OSC.0001_1024.hdf5")

    # console
    python scripts/inspect_dataset.py --config baseline
    python scripts/inspect_dataset.py --path data/GOLD_XYZ_OSC.0001_1024.hdf5
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_data_path
from src.data import (
    KEY_X,
    KEY_Y,
    KEY_Z,
    MODULATION_CLASSES
)

_DEFAULT_PATH = "data/GOLD_XYZ_OSC.0001_1024.hdf5"

def inspect_dataset(
    config: str | Path | None = None,
    path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Open the HDF5, print its real structure, and verify the (class, SNR) grid.

    The file is chosen from `config` (a config name/path -> its data.path) or from an
    explicit `path`; if neither is given, the default dataset is used.

    Prints actual values read from the file; nothing is hardcoded from assumptions.
    Returns a summary dict for programmatic use.
    """
    data_path, config_path = resolve_data_path(config, path, _DEFAULT_PATH)
    if verbose and config_path is not None:
        print(f"Using config at the path: {config_path}")
    path = Path(data_path)
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        shapes = {k: tuple(f[k].shape) for k in keys}
        dtypes = {k: str(f[k].dtype) for k in keys}

        y = f[KEY_Y][:]
        z = f[KEY_Z][:].reshape(-1)
        n_classes = int(f[KEY_Y].shape[1])
        frame_len = int(f[KEY_X].shape[1])

    class_idx = y.argmax(axis=1)
    is_one_hot = bool(np.all(y.sum(axis=1) == 1) and np.all(y.max(axis=1) == 1))
    class_counts = np.bincount(class_idx, minlength=n_classes)

    uniq_snr, snr_counts = np.unique(z, return_counts=True)
    snr_steps = np.unique(np.diff(uniq_snr)).tolist()

    # Grid: frames per (class, SNR) cell.
    snr_pos = {int(v): i for i, v in enumerate(uniq_snr)}
    snr_index = np.array([snr_pos[int(v)] for v in z], dtype=np.int64)
    grid = np.zeros((n_classes, len(uniq_snr)), dtype=np.int64)
    np.add.at(grid, (class_idx, snr_index), 1)
    grid_uniform = bool(grid.min() == grid.max())

    names_ok = len(MODULATION_CLASSES) == n_classes

    summary = {
        "path": str(path),
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "n_frames": int(y.shape[0]),
        "n_classes": n_classes,
        "frame_length": frame_len,
        "one_hot": is_one_hot,
        "snr_values": uniq_snr.tolist(),
        "snr_step": snr_steps,
        "frames_per_cell": int(grid.max()),
        "grid_uniform": grid_uniform,
        "grid_shape": grid.shape,
    }

    if verbose:
        print("=== FILE ===")
        print(f"path: {path}")
        print(f"keys: {keys}")
        for k in keys:
            print(f"  {k}: shape={shapes[k]} dtype={dtypes[k]}")
        print("\n=== LABELS ===")
        print(f"n_frames   : {summary['n_frames']}")
        print(f"n_classes  : {n_classes}")
        print(f"frame_length: {frame_len}")
        print(f"strict one-hot Y: {is_one_hot}")
        print("class counts (column index -> canonical name -> count):")
        for i in range(n_classes):
            name = MODULATION_CLASSES[i] if names_ok else "?"
            print(f"  {i:2d}  {name:<10s}  {int(class_counts[i])}")
        print("\n=== SNR ===")
        print(f"unique SNR values ({len(uniq_snr)}): {uniq_snr.tolist()}")
        print(f"SNR step(s): {snr_steps}")
        print(f"count per SNR level: {snr_counts.tolist()}")
        print("\n=== GRID (class x SNR) ===")
        print(f"grid shape           : {grid.shape}")
        print(f"frames per cell (min): {int(grid.min())}")
        print(f"frames per cell (max): {int(grid.max())}")
        print(f"grid uniform         : {grid_uniform}")
        expected = n_classes * len(uniq_snr) * int(grid.max())
        print(f"total = {n_classes} x {len(uniq_snr)} x {int(grid.max())} = {expected} "
              f"(matches n_frames: {expected == summary['n_frames']})")
        if not names_ok:
            print(f"\nWARNING: {len(MODULATION_CLASSES)} canonical names but file has "
                  f"{n_classes} classes -- mapping cannot be trusted.")

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the RadioML HDF5 file.")
    parser.add_argument("--path", default=None, help="Path to the HDF5 file.")
    parser.add_argument("--config", default=None,
                        help="Config name (e.g. 'baseline') or path; its data.path is used.")
    args = parser.parse_args()

    inspect_dataset(config=args.config, path=args.path, verbose=True)


if __name__ == "__main__":
    main()
