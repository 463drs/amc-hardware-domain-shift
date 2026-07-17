"""Create a compact standalone HDF5 containing only the stratified subset.

The output name is auto-generated from the parameters that determine the file's contents:
    subset_fpp{frames_per_pair}_seed{subset_seed}_snr{snr_min}_{snr_max}.hdf5
(+ "_gz" when --compress). Those four values plus the source file fully define the subset
(the split / normalization are applied at load time, not baked in), so the name is a
complete signature and re-running is idempotent -- an existing matching file is left alone
unless --overwrite is given.

Usage:
    python scripts/make_subset.py --config configs/baseline.yaml           # auto name in data/
    python scripts/make_subset.py --config configs/baseline.yaml --out data/my_subset.hdf5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, resolve_config_path
from src.data import (
    KEY_X,
    KEY_Y,
    KEY_Z,
    MODULATION_CLASSES,
    read_labels_and_snr,
    select_subset_indices,
)

# Rows copied per HDF5 read/write batch (bounds memory; each frame is ~8 KB).
_COPY_BATCH = 4096


def default_output_name(cfg, compress: bool) -> str:
    """Signature filename fully describing the subset's contents."""
    suffix = "_gz" if compress else ""
    return (f"subset_fpp{cfg.frames_per_pair}_seed{cfg.subset_seed}"
            f"_snr{cfg.snr_min}_{cfg.snr_max}{suffix}.hdf5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact stratified subset HDF5.")
    parser.add_argument("--config", required=True,
                        help="Config name (e.g. 'baseline') or path. Bare names resolve in configs/.")
    parser.add_argument("--out", default=None,
                        help="Explicit output HDF5 path (default: auto signature name in --out-dir).")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the auto-named output (default: the source file's directory).")
    parser.add_argument("--compress", action="store_true",
                        help="gzip-compress X (smaller upload, slower reads).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite even if the target file already exists.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = Config.from_yaml(config_path)
    cfg = config.data
    src_path = Path(cfg.path)

    # Resolve the output path: explicit --out wins; otherwise auto signature name in --out-dir.
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_dir = Path(args.out_dir) if args.out_dir is not None else src_path.parent
        out_path = out_dir / default_output_name(cfg, args.compress)

    if out_path.resolve() == src_path.resolve():
        raise ValueError("Output path must differ from the source dataset.")

    if out_path.exists() and not args.overwrite:
        print(f"[make_subset] {out_path} already exists; skipping (use --overwrite to rewrite).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[make_subset] source: {src_path}")
    print(f"[make_subset] output: {out_path}")
    print(f"[make_subset] reading labels/SNR ...")
    class_idx, snr, frame_len, n_classes = read_labels_and_snr(src_path)

    indices = select_subset_indices(
        class_idx=class_idx,
        snr=snr,
        frames_per_pair=cfg.frames_per_pair,
        subset_seed=cfg.subset_seed,
        snr_min=cfg.snr_min,
        snr_max=cfg.snr_max,
    )
    n_out = int(indices.size)
    print(f"[make_subset] selecting {n_out} frames "
          f"(frames_per_pair={cfg.frames_per_pair}, snr=[{cfg.snr_min},{cfg.snr_max}])")

    compression = "gzip" if args.compress else None

    with h5py.File(src_path, "r") as fin, h5py.File(out_path, "w") as fout:
        x_in = fin[KEY_X]
        x_dtype = x_in.dtype

        x_out = fout.create_dataset(
            KEY_X, shape=(n_out, frame_len, 2), dtype=x_dtype, compression=compression,
        )
        # Y and Z are small; copy directly. Sorted indices satisfy h5py's increasing-order rule.
        idx_list = indices.tolist()
        fout.create_dataset(KEY_Y, data=fin[KEY_Y][idx_list], compression=compression)
        fout.create_dataset(KEY_Z, data=fin[KEY_Z][idx_list], compression=compression)

        # Copy X in batches to bound memory.
        for start in range(0, n_out, _COPY_BATCH):
            stop = min(start + _COPY_BATCH, n_out)
            batch = indices[start:stop].tolist()   # ascending within the batch
            x_out[start:stop] = x_in[batch]
            print(f"\r[make_subset] copying X: {stop}/{n_out}", end="", flush=True)
        print()

        # Provenance so the file is self-describing.
        fout.attrs["frames_per_pair"] = cfg.frames_per_pair
        fout.attrs["subset_seed"] = cfg.subset_seed
        fout.attrs["snr_min"] = cfg.snr_min
        fout.attrs["snr_max"] = cfg.snr_max
        fout.attrs["split"] = list(cfg.split)
        fout.attrs["split_seed"] = cfg.split_seed
        fout.attrs["normalization"] = cfg.normalization
        fout.attrs["source_file"] = src_path.name
        fout.attrs["source_n_frames"] = int(class_idx.size)
        fout.attrs["n_frames"] = n_out
        fout.attrs["n_classes"] = n_classes
        fout.attrs["frame_length"] = frame_len
        fout.attrs["class_order"] = json.dumps(list(MODULATION_CLASSES))
        fout.attrs["source_config"] = str(config.source_path)
        fout.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    size_gb = out_path.stat().st_size / 1e9
    print(f"[make_subset] wrote {out_path} ({n_out} frames, {size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
