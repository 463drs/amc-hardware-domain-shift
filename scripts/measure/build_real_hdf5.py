"""Cut the RTL-SDR captures into a RadioML-format HDF5. Offline -- reads files, drives nothing.

Front end per capture, identical to check_capture.py because it IS check_capture.py: drop
SKIP_S, uint8 -> complex about mid-scale, shift back by f_shift (whole bins), frame, notch the
three spurs. The notch sits where the spurs were FOUND (median over captures.json), not where
they are predicted: the two radios' reference oscillators disagree by ~4 kHz here, e.g. the LO
leakage lands at +328 kHz in every capture against a predicted +324. NOTCH_HALF_BINS = 3 (7
bins) covers the 1-2 kHz scatter around that median. Label = class from the filename; Z = the 2 dB bin (check_bins.bin_centre)
of the noise-bw SNR measured for that file. Only bins check_bins calls usable are kept, and
each is truncated to the same frame count for every class, split across that class's captures
in proportion to their length so a bin fed by two rungs keeps both.

Keys, dtypes and trailing shapes are copied from the synthetic reference file, and frames get
the reference config's normalizer from src.data (re-applied at load time, where it is a no-op).

  python scripts/measure/build_real_hdf5.py --captures captures --config baseline_100
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_bins import (BIN_WIDTH_DB, MIN_FRAMES, REF_BW_KHZ, bin_centre, classify_bins,
                        frames_in, occupancy, snr_of)
from check_capture import (MID_SCALE, SKIP_S, apply_shift, load_capture, notch_frames,
                           shift_bins, spur_targets)
from make_tx import FRAME_LEN, FS_RX, to_frames
from src.config import Config, resolve_config_path
from src.data import BATCHED_NORMALIZERS, KEY_X, KEY_Y, KEY_Z, MODULATION_CLASSES

SNR_REF = "noise-bw"
NOTCH_HALF_BINS = 3        # 7 bins per spur: the 1-2 kHz scatter of the found positions
_DEFAULT_OUT = "data/real_captures_2db_noise-bw.hdf5"


def front_end(path: Path, n_frames: int, k_shift: int, spur_hz,
              half_bins: int = NOTCH_HALF_BINS) -> tuple[np.ndarray, float]:
    """One capture -> (first n_frames notched complex frames, clipped fraction)."""
    x, clipped = load_capture(path)
    frames = to_frames(apply_shift(x, k_shift), n_frames)
    if frames.shape[0] < n_frames:
        raise ValueError(f"{path}: {frames.shape[0]} frames, {n_frames} requested")
    return notch_frames(frames, spur_hz, half_bins)[0], clipped


def found_spurs(records: list[dict]) -> dict[str, float]:
    """Median found position of each spur over every capture, in Hz, whole bins."""
    names = [s["name"] for s in records[0]["spurs"]]
    bin_hz = FS_RX / FRAME_LEN
    return {n: float(bin_hz * round(np.median([s["found_khz"] for r in records
                                                for s in r["spurs"] if s["name"] == n])
                                     * 1e3 / bin_hz)) for n in names}


def notch_bins(spur_hz: dict[str, float], half_bins: int = NOTCH_HALF_BINS) -> list[int]:
    """Signed bin indices notch_frames zeroes for these centres (bin k = k * fs/1024 Hz)."""
    bin_hz = FS_RX / FRAME_LEN
    return sorted({int(round(f / bin_hz)) + d for f in spur_hz.values()
                   for d in range(-half_bins, half_bins + 1)})


def to_layout(frames: np.ndarray, normalization: str, dtype) -> np.ndarray:
    """Complex (N, T) -> normalized (N, T, 2), via the same batched normalizer as training."""
    x = np.stack((frames.real, frames.imag), axis=1).astype(dtype)        # (N, 2, T)
    x = BATCHED_NORMALIZERS[normalization](torch.from_numpy(x))
    return np.ascontiguousarray(x.permute(0, 2, 1).numpy())


def split_quota(total: int, sizes: list[int]) -> list[int]:
    """Split `total` over captures in proportion to their sizes (largest remainder)."""
    exact = np.array(sizes, dtype=float) * total / sum(sizes)
    share = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - share))[: total - share.sum()]:
        share[i] += 1
    return share.tolist()


def plan(records: list[dict], root: Path, usable: list[int]) -> tuple[list[dict], dict]:
    """Rows to write, one entry per contributing capture, ordered bin -> class -> SNR."""
    by_cell: dict[tuple[int, str], list[dict]] = {}
    for r in records:
        if Path(r["file"].replace("\\", "/")).stem != r["class"]:
            raise ValueError(f"{r['file']} is recorded as class {r['class']}")
        b = bin_centre(snr_of(r, SNR_REF), BIN_WIDTH_DB)
        if b in usable:
            by_cell.setdefault((b, r["class"]), []).append(r)
    n_min = {b: min(sum(frames_in(root / r["file"]) for r in by_cell[(b, c)])
                    for c in MODULATION_CLASSES) for b in usable}

    rows, start = [], 0
    for b in usable:
        for c in MODULATION_CLASSES:
            caps = sorted(by_cell[(b, c)], key=lambda r: (r["snr_db"], r["file"]))
            sizes = [frames_in(root / r["file"]) for r in caps]
            # Kept even when their share is 0, so the table shows every capture in the bin.
            for r, n, avail in zip(caps, split_quota(n_min[b], sizes), sizes):
                rows.append({"file": r["file"].replace("\\", "/"), "class": c, "bin": b,
                             "snr_inband_db": r["snr_db"],
                             "snr_noise_bw_db": round(snr_of(r, SNR_REF), 2),
                             "row_start": start, "n_frames": n, "frames_available": avail})
                start += n
    return rows, n_min


def main() -> int:
    p = argparse.ArgumentParser(description="Build a RadioML-format HDF5 from the captures.")
    p.add_argument("--captures", default="captures", help="Directory holding captures.json.")
    p.add_argument("--tx-dir", default="tx", help="Directory holding manifest.json (f_off).")
    p.add_argument("--config", default="baseline_100",
                   help="Config whose data.path is the synthetic reference (keys, dtypes, "
                        "shapes) and whose data.normalization is applied.")
    p.add_argument("--out", default=_DEFAULT_OUT, help="Output HDF5 path.")
    p.add_argument("--overwrite", action="store_true", help="Rewrite an existing output.")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to rewrite it")
        return 1

    root = Path(args.captures)
    doc = json.loads((root / "captures.json").read_text(encoding="utf-8"))
    records = doc["captures"]
    manifest = json.loads((Path(args.tx_dir) / "manifest.json").read_text(encoding="utf-8"))
    config_path = resolve_config_path(args.config)
    cfg = Config.from_yaml(config_path).data
    if doc["fs_rx_hz"] != FS_RX:
        raise ValueError(f"captures at {doc['fs_rx_hz']} Hz, front end expects {FS_RX}")
    if not doc["notched"]:
        raise ValueError("captures.json was measured without the notch; SNRs would not match")

    f_off, f_shift = float(manifest["f_off_hz"]), float(doc["f_shift_hz"])
    k_shift = shift_bins(f_shift)
    predicted = spur_targets(f_off, f_shift)
    for r in records:
        recorded = {s["name"]: s["expected_khz"] for s in r["spurs"]}
        if any(abs(recorded[k] - v / 1e3) > 1e-6 for k, v in predicted.items()):
            raise ValueError(f"{r['file']}: predicted spur positions differ from this setup's")
    spur_hz = found_spurs(records)
    # Moving the notch leaves captures.json's SNRs valid only while it stays out of band.
    zeroed = notch_bins(spur_hz)
    bin_khz = FS_RX / FRAME_LEN / 1e3
    band_lo = min(r["band_lo_khz"] for r in records)
    band_hi = max(r["band_hi_khz"] for r in records)
    if any(band_lo <= k * bin_khz <= band_hi for k in zeroed):
        raise ValueError(f"notch reaches into the occupied band {band_lo}..{band_hi} kHz")
    print("notch at found positions: " + ", ".join(
        f"{k} {predicted[k] / 1e3:+.0f}k -> {v / 1e3:+.0f}k" for k, v in spur_hz.items())
        + f"; {len(zeroed)} bins, widest occupied band {band_lo:.0f}..{band_hi:.0f} kHz")

    with h5py.File(cfg.path, "r") as ref:
        ref_keys = sorted(ref.keys())
        spec = {k: (ref[k].dtype, ref[k].shape[1:]) for k in (KEY_X, KEY_Y, KEY_Z)}
    if ref_keys != sorted((KEY_X, KEY_Y, KEY_Z)):
        raise ValueError(f"reference {cfg.path} has keys {ref_keys}, expected X/Y/Z only")
    if spec[KEY_X][1] != (FRAME_LEN, 2) or spec[KEY_Y][1] != (len(MODULATION_CLASSES),):
        raise ValueError(f"reference layout {spec} does not match the front end")

    classes = list(MODULATION_CLASSES)
    table = occupancy(records, root, SNR_REF, BIN_WIDTH_DB)
    ok, under, incomplete = classify_bins(table, classes, MIN_FRAMES)
    if not ok:
        raise ValueError("no usable bin")
    rows, n_min = plan(records, root, ok)
    n_total = sum(r["n_frames"] for r in rows)
    print(f"{len(ok)} usable bins {ok}; excluded {under + incomplete}")
    print(f"{n_total} frames = {len(classes)} classes x "
          + ", ".join(f"{b:+d}:{n_min[b]}" for b in ok)
          + f"  (~{n_total * FRAME_LEN * 2 * spec[KEY_X][0].itemsize / 1e9:.1f} GB)")

    meta = {
        "rf_setup": {k: doc[k] for k in ("tx_centre_hz", "rx_centre_hz", "f_shift_hz",
                                          "fs_rx_hz", "rx_gain_db", "seconds",
                                          "equalize_papr")}
                    | {"f_off_hz": f_off, "tx": "HackRF, tx/<class>.bin looped",
                       "rx": "RTL-SDR, uint8 I/Q, AGC off"},
        "front_end": {"skip_s": SKIP_S, "mid_scale": MID_SCALE, "k_shift_bins": k_shift,
                      "frame_len": FRAME_LEN, "normalization": cfg.normalization,
                      "x_storage": f"X is stored ALREADY normalized ({cfg.normalization}), "
                                   "unlike the synthetic files which store raw frames; the "
                                   "loader re-applies it, which is idempotent"},
        "notch": {"source": f"median found_khz over {len(records)} captures.json records",
                  "half_bins": NOTCH_HALF_BINS, "bins_per_spur": 2 * NOTCH_HALF_BINS + 1,
                  "centres_hz": spur_hz, "predicted_hz": predicted,
                  "n_bins": len(zeroed), "bins_khz": [k * bin_khz for k in zeroed],
                  "widest_occupied_band_khz": [band_lo, band_hi]},
        "snr": {"definition": f"{SNR_REF}: in-band SNR measured by check_capture.py "
                              f"(signal power over zero-signal capture, notched, in the TX "
                              f"file's 99% band) + 10*log10(band width / {REF_BW_KHZ:g} kHz)",
                "ref_bw_khz": REF_BW_KHZ, "bin_width_db": BIN_WIDTH_DB,
                "bin_rule": "centre = width * floor((snr + width/2) / width); Z holds the centre",
                "bin_edges_db": {str(b): [b - BIN_WIDTH_DB / 2, b + BIN_WIDTH_DB / 2]
                                 for b in ok},
                "usable_bins": ok, "min_frames": MIN_FRAMES,
                "excluded_bins": {"underpopulated": under,
                                  "incomplete": {str(b): [c for c in classes
                                                          if not table[b].get(c)]
                                                 for b in incomplete}}},
        "frame_counts": {"per_class_per_bin": {str(b): n_min[b] for b in ok},
                         "available_before_equalizing": {str(b): table[b] for b in ok},
                         "total": n_total},
        "papr_db": {c["class"]: c["papr_db"] for c in manifest["classes"]},
        "captures": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial")
    with h5py.File(tmp, "w") as f:
        x = f.create_dataset(KEY_X, shape=(n_total, *spec[KEY_X][1]), dtype=spec[KEY_X][0])
        y = f.create_dataset(KEY_Y, shape=(n_total, *spec[KEY_Y][1]), dtype=spec[KEY_Y][0])
        z = f.create_dataset(KEY_Z, shape=(n_total, *spec[KEY_Z][1]), dtype=spec[KEY_Z][0])
        for i, r in enumerate(rows):
            if not r["n_frames"]:
                continue
            frames, clipped = front_end(root / r["file"], r["n_frames"], k_shift,
                                        spur_hz.values())
            r["clipped_frac"] = clipped
            a, b = r["row_start"], r["row_start"] + r["n_frames"]
            x[a:b] = to_layout(frames, cfg.normalization, spec[KEY_X][0])
            onehot = np.zeros((r["n_frames"], len(classes)), dtype=spec[KEY_Y][0])
            onehot[:, classes.index(Path(r["file"]).stem)] = 1
            y[a:b] = onehot
            z[a:b] = r["bin"]
            print(f"\r  {i + 1}/{len(rows)} captures, {b}/{n_total} frames", end="", flush=True)
        print()

        f.attrs["metadata"] = json.dumps(meta, indent=1)
        f.attrs["n_frames"] = n_total
        f.attrs["n_classes"] = len(classes)
        f.attrs["frame_length"] = FRAME_LEN
        f.attrs["normalization"] = cfg.normalization
        f.attrs["class_order"] = json.dumps(classes)
        f.attrs["reference_file"] = Path(cfg.path).name
        f.attrs["source_config"] = str(config_path)
        f.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    tmp.replace(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
