"""Compare the generated TX files against RadioML 2018.01A, offline -- no hardware involved.

Per class: remove f_off, decimate 2.048 -> 1.024 MS/s, cut 1024-sample frames, then overlay
the averaged PSD on RadioML's at SNR = 30 dB and sample the constellation after matched
filtering. Mismatches are REPORTED, not fixed: the numbers say which SPEC entry to revisit.

  python scripts/measure/validate_tx.py --tx-dir tx
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_tx import (FRAME_LEN, FS_RX, OCC_FRAC, RRC_SPAN, SPEC, SPS_RX, avg_psd,
                     decimate_to_rx, frame_papr_db, occupied_mask, rrc_taps, to_frames)
from src.config import resolve_data_path
from src.data import KEY_X, MODULATION_CLASSES, read_labels_and_snr

_DEFAULT_DATA = "data/subset_fpp384_seed1234_snr-20_30.hdf5"
_DEFAULT_OUT_DIR = "outputs/tx_validation"


# Loading



def load_tx(path: Path, k_off: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one int8 I/Q loop -> (raw complex at 2.048 MS/s, frames at 1.024 MS/s)."""
    raw = np.fromfile(path, dtype=np.int8).astype(np.float64)
    x = raw[0::2] + 1j * raw[1::2]
    return x, to_frames(decimate_to_rx(x, k_off), n_frames)


def load_radioml(path: str, name: str, snr: int, n_frames: int,
                 labels: tuple, offset: int = 0) -> np.ndarray:
    """Frames of one class at one SNR, as (n, 1024) complex. `offset` skips that many frames
    first, so a different slice of the same cell can be drawn to test measurement stability."""
    class_idx, snr_all = labels
    rows = np.nonzero((class_idx == MODULATION_CLASSES.index(name)) & (snr_all == snr))[0]
    if rows.size == 0:
        raise ValueError(f"{name} at SNR {snr} dB is not in {path}")
    rows = rows[offset:offset + n_frames]
    if rows.size == 0:
        raise ValueError(f"{name} at SNR {snr} dB has no frames past offset {offset}")
    with h5py.File(path, "r") as f:
        frames = f[KEY_X][rows.tolist()]
    return frames[..., 0] + 1j * frames[..., 1]


# Measurements



def symbol_samples(frames: np.ndarray, alpha: float, matched: bool) -> np.ndarray:
    """Matched-filter each frame and sample at whichever of the SPS_RX phases peaks in power."""
    taps = rrc_taps(alpha, SPS_RX, RRC_SPAN)
    out = []
    for frame in frames:
        y = np.convolve(frame, taps, mode="same") if matched else frame
        power = [np.mean(np.abs(y[o::SPS_RX]) ** 2) for o in range(SPS_RX)]
        out.append(y[int(np.argmax(power))::SPS_RX])
    return np.concatenate(out)




def loop_step_ratio(x: np.ndarray) -> float:
    """Jump across the loop joint relative to a typical sample-to-sample step. ~1 = seamless."""
    return float(np.abs(x[0] - x[-1]) / np.median(np.abs(np.diff(x))))




def compare_psd(tx_psd: np.ndarray, rml_psd: np.ndarray) -> dict:
    """dB statistics over the band RadioML occupies, plus each side's own occupied width.

    `bias` is the in-band minus out-of-band power split (both PSDs carry unit total power,
    so it mostly reports floor and skirt differences); `rms`/`max` are taken after removing
    it and are therefore pure shape mismatch."""
    freq = np.fft.fftshift(np.fft.fftfreq(FRAME_LEN, 1 / FS_RX)) / 1e3        # kHz
    tx_db, rml_db = 10 * np.log10(tx_psd + 1e-20), 10 * np.log10(rml_psd + 1e-20)
    band = occupied_mask(rml_psd)
    diff = tx_db[band] - rml_db[band]
    bias = float(diff.mean())
    bin_khz = FS_RX / FRAME_LEN / 1e3
    return {"band_lo_khz": round(float(freq[band].min()), 1),
            "band_hi_khz": round(float(freq[band].max()), 1),
            "bw_rml_khz": round(float(band.sum() * bin_khz), 1),
            "bw_tx_khz": round(float(occupied_mask(tx_psd).sum() * bin_khz), 1),
            "psd_bias_db": round(bias, 2),
            "psd_rms_db": round(float(np.sqrt(np.mean((diff - bias) ** 2))), 2),
            "psd_max_db": round(float(np.abs(diff - bias).max()), 2)}


# Figure

def plot_class(name: str, x_raw: np.ndarray, tx_frames: np.ndarray, rml_frames: np.ndarray,
               alpha: float, stats: dict, out_path: Path) -> None:
    """Four panels: PSD overlay, PSD difference, constellation / envelope, loop joint."""
    freq = np.fft.fftshift(np.fft.fftfreq(FRAME_LEN, 1 / FS_RX)) / 1e3
    tx_db = 10 * np.log10(avg_psd(tx_frames) + 1e-20)
    rml_db = 10 * np.log10(avg_psd(rml_frames) + 1e-20)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(f"{name}  --  TX (2.048 MS/s -> 1.024 MS/s) vs RadioML 2018.01A @ 30 dB SNR")

    ax[0, 0].plot(freq, rml_db, lw=1.0, label="RadioML")
    ax[0, 0].plot(freq, tx_db, lw=1.0, label="TX file")
    ax[0, 0].axvspan(stats["band_lo_khz"], stats["band_hi_khz"], color="0.85", zorder=0)
    ax[0, 0].set(xlabel="kHz", ylabel="dB (unit total power)",
                 title=f"averaged PSD  ({OCC_FRAC:.0%} band shaded)")
    ax[0, 0].legend(fontsize=8)

    band = occupied_mask(avg_psd(rml_frames))
    ax[0, 1].plot(freq[band], (tx_db - rml_db - stats["psd_bias_db"])[band], lw=1.0)
    ax[0, 1].axhline(0, color="0.6", lw=0.8)
    ax[0, 1].set(xlabel="kHz", ylabel="TX - RadioML - bias, dB",
                 title=f"shape difference over the occupied band  "
                       f"(rms {stats['psd_rms_db']} dB, max {stats['psd_max_db']} dB)")

    spec = SPEC[name]
    if "mod" in spec:
        ax[1, 0].hist(np.abs(tx_frames).ravel(), bins=120, histtype="step", density=True,
                      label="TX")
        ax[1, 0].hist(np.abs(rml_frames).ravel() / np.abs(rml_frames).std(), bins=120,
                      histtype="step", density=True, label="RadioML (scaled)")
        ax[1, 0].set(xlabel="|x|", title="envelope distribution (analog class)")
        ax[1, 0].legend(fontsize=8)
    else:
        s = symbol_samples(tx_frames[:16], alpha, spec["shape"] != "gmsk")
        lim = 1.15 * np.abs(s).max()
        ax[1, 0].plot(s.real, s.imag, ".", ms=1.5)
        ax[1, 0].set(xlabel="I", ylabel="Q", xlim=(-lim, lim), ylim=(-lim, lim),
                     title="TX constellation after matched filter")
        ax[1, 0].set_aspect("equal")

    joint = np.concatenate((x_raw[-96:], x_raw[:96]))
    ax[1, 1].plot(np.arange(-96, 96), joint.real, lw=0.8, label="I")
    ax[1, 1].plot(np.arange(-96, 96), joint.imag, lw=0.8, label="Q")
    ax[1, 1].axvline(-0.5, color="r", lw=0.8)
    ax[1, 1].set(xlabel="samples around the loop joint", ylabel="LSB",
                 title=f"loop joint (step / typical step = {stats['loop_step_ratio']})")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate generated TX files against RadioML.")
    p.add_argument("--tx-dir", default="tx", help="Directory holding <class>.bin + manifest.json.")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="Where figures and summary.csv go.")
    p.add_argument("--config", default=None, help="Config name whose data.path supplies the HDF5.")
    p.add_argument("--path", default=None, help="Explicit RadioML HDF5 path.")
    p.add_argument("--snr", type=int, default=30, help="RadioML SNR slice used as the reference.")
    p.add_argument("--n-frames", type=int, default=128, help="Frames averaged on each side.")
    p.add_argument("--classes", nargs="*", default=None, help="Subset of classes (default: all).")
    args = p.parse_args()

    tx_dir = Path(args.tx_dir)
    manifest = json.loads((tx_dir / "manifest.json").read_text(encoding="utf-8"))
    k_off, alpha = manifest["f_off_cycles"], manifest["rrc_alpha"]
    entry = {c["class"]: c for c in manifest["classes"]}

    data_path = resolve_data_path(args.config, args.path, _DEFAULT_DATA)[0]
    class_idx, snr_all, _, _ = read_labels_and_snr(data_path)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = list(args.classes) if args.classes else [c["class"] for c in manifest["classes"]]
    rows = []
    print(f"{'class':<11}{'bw tx':>8}{'bw rml':>8}{'bias':>7}{'rms':>7}{'max':>7}"
          f"{'PAPR tx':>9}{'PAPR rml':>10}{'joint':>7}")
    for name in names:
        x_raw, tx_frames = load_tx(tx_dir / f"{name}.bin", k_off, args.n_frames)
        rml_frames = load_radioml(data_path, name, args.snr, args.n_frames,
                                  (class_idx, snr_all))
        stats = compare_psd(avg_psd(tx_frames), avg_psd(rml_frames))
        # papr_tx_db is the whole-file peak (what sets the DAC backoff); papr_tx_frame_db
        # repeats RadioML's per-frame measure so the two columns are comparable.
        stats.update(**{"class": name, "digital": entry[name]["digital"],
                        "snr_tx_ceiling_db": entry[name]["snr_tx_ceiling_db"],
                        "papr_tx_db": entry[name]["papr_db"],
                        "papr_tx_frame_db": round(frame_papr_db(tx_frames), 2),
                        "papr_rml_db": round(frame_papr_db(rml_frames), 2),
                        "loop_step_ratio": round(loop_step_ratio(x_raw), 2)})
        plot_class(name, x_raw, tx_frames, rml_frames, alpha, stats, out / f"tx_{name}.png")
        rows.append(stats)
        print(f"{name:<11}{stats['bw_tx_khz']:>8.1f}{stats['bw_rml_khz']:>8.1f}"
              f"{stats['psd_bias_db']:>7.2f}{stats['psd_rms_db']:>7.2f}{stats['psd_max_db']:>7.2f}"
              f"{stats['papr_tx_frame_db']:>9.2f}{stats['papr_rml_db']:>10.2f}"
              f"{stats['loop_step_ratio']:>7.2f}")

    cols = ["class", "digital", "band_lo_khz", "band_hi_khz", "bw_tx_khz", "bw_rml_khz",
            "psd_bias_db", "psd_rms_db", "psd_max_db", "papr_tx_db", "papr_tx_frame_db",
            "papr_rml_db", "loop_step_ratio", "snr_tx_ceiling_db"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows({k: r[k] for k in cols} for r in rows)

    print(f"\n{len(rows)} figures + summary.csv -> {out}/")
    # Reported with and without the 5 analog classes: those carry a documented deviation
    # from RadioML, so they would otherwise dominate any aggregate.
    for label, sel in (("all classes", rows), ("digital only", [r for r in rows if r["digital"]])):
        if sel:
            rms = [r["psd_rms_db"] for r in sel]
            papr = [r["papr_tx_frame_db"] - r["papr_rml_db"] for r in sel]
            ceil = [r["snr_tx_ceiling_db"] for r in sel]
            print(f"  {label:<13} n={len(sel):>2}  PSD rms mean {np.mean(rms):5.2f} / max "
                  f"{max(rms):5.2f} dB   PAPR err mean {np.mean(papr):+5.2f} dB   "
                  f"TX SNR ceiling {min(ceil):.1f}..{max(ceil):.1f} dB")
    worst = sorted(rows, key=lambda r: -r["psd_rms_db"])[:5]
    print("  worst PSD match: " + ", ".join(f"{r['class']} ({r['psd_rms_db']} dB)" for r in worst))


if __name__ == "__main__":
    main()
