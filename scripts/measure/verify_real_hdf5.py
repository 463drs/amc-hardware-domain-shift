"""Verify the real-capture HDF5 before anything trains or evaluates on it. Offline.

Checks, each a hard failure unless noted:
  layout      keys, dtypes and trailing shapes equal to the synthetic reference file
  classes     all 24 present, every Y row exactly one-hot
  labels      each capture's rows carry the class of its filename and the bin of its SNR
  balance     per bin, the same frame count for every class; only usable bins present
  finite      no NaN/Inf; per-frame power 1 under the recorded normalization
  rebuild     a few captures re-run through the front end reproduce their rows exactly
  PSD         QPSK at the top bin vs tx/QPSK.bin through the same front end (reported, and
              flagged above PSD_RMS_LIMIT_DB), plus spur residue at each notch
  limiting    per bin, which class sets the equalized count and whether a short file is why
  synthetic   every synthetic SNR label maps to its own bin under check_bins.bin_centre

  python scripts/measure/verify_real_hdf5.py --path data/real_captures_2db_noise-bw.hdf5
"""

from __future__ import annotations

import argparse
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

from build_real_hdf5 import _DEFAULT_OUT, front_end, to_layout
from check_bins import bin_centre
from check_capture import RML_FRAMES, notch_frames
from make_tx import FRAME_LEN, FS_RX, avg_psd
from src.config import Config
from src.data import KEY_X, KEY_Y, KEY_Z, MODULATION_CLASSES
from src.metrics import snr_bucket
from validate_tx import compare_psd, load_tx

_BLOCK = 16384
N_REBUILD = 4
PSD_RMS_LIMIT_DB = 1.5     # in-band shape mismatch above this is flagged, not failed
POWER_TOL = 1e-3


class Report:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            self.failed.append(name)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify the real-capture HDF5.")
    p.add_argument("--path", default=_DEFAULT_OUT, help="HDF5 written by build_real_hdf5.py.")
    p.add_argument("--captures", default="captures", help="Directory the captures came from.")
    p.add_argument("--tx-dir", default="tx", help="Directory holding tx/<class>.bin.")
    p.add_argument("--out-dir", default="outputs/real_hdf5", help="Where the figure goes.")
    p.add_argument("--seed", type=int, default=0, help="Picks the captures to rebuild.")
    args = p.parse_args()

    rep = Report()
    root = Path(args.captures)
    classes = list(MODULATION_CLASSES)
    with h5py.File(args.path, "r") as f:
        meta = json.loads(f.attrs["metadata"])
        source_config = Config.from_yaml(f.attrs["source_config"])
        norm = str(f.attrs["normalization"])
        x_ds = f[KEY_X]
        y = f[KEY_Y][:]
        z = f[KEY_Z][:].reshape(-1)
        n = x_ds.shape[0]
        print(f"{args.path}: {n} frames")

        print("layout")
        with h5py.File(source_config.data.path, "r") as ref:
            same_keys = sorted(f.keys()) == sorted(ref.keys())
            rep.check("keys", same_keys, f"{sorted(f.keys())} vs {sorted(ref.keys())}")
            for k in (KEY_X, KEY_Y, KEY_Z):
                got, want = (f[k].dtype, f[k].shape[1:]), (ref[k].dtype, ref[k].shape[1:])
                rep.check(f"{k} dtype/shape", got == want, f"{got} vs reference {want}")
        rep.check("normalization matches the reference config",
                  norm == source_config.data.normalization, norm)

        print("classes")
        onehot = np.all((y == 0) | (y == 1), axis=1) & (y.sum(axis=1) == 1)
        rep.check("Y rows one-hot", bool(onehot.all()), f"{int((~onehot).sum())} bad rows")
        cls = y.argmax(axis=1)
        present = np.unique(cls)
        rep.check("24 classes present", present.size == len(classes),
                  f"{present.size} present")

        print("labels")
        caps = meta["captures"]
        covered = np.zeros(n, dtype=bool)
        bad = []
        for c in caps:
            a, b = c["row_start"], c["row_start"] + c["n_frames"]
            want = classes.index(Path(c["file"]).stem)
            if (cls[a:b] != want).any() or (z[a:b] != c["bin"]).any() or covered[a:b].any():
                bad.append(c["file"])
            covered[a:b] = True
        rep.check("rows match filename class and SNR bin", not bad, ", ".join(bad[:5]))
        rep.check("capture table covers every row once", bool(covered.all()))

        print("balance / SNR distribution per class (frames, rows = bins)")
        usable = meta["snr"]["usable_bins"]
        bins = sorted(np.unique(z).tolist())
        rep.check("only usable bins present", bins == usable, f"{bins}")
        grid = np.array([[np.count_nonzero((z == b) & (cls == k)) for k in range(len(classes))]
                         for b in bins])
        print(f"      {'bin':>5} {'min':>6} {'max':>6}   measured noise-bw SNR range")
        for b, row in zip(bins, grid):
            snrs = [c["snr_noise_bw_db"] for c in caps if c["bin"] == b]
            print(f"      {b:>+5d} {row.min():>6d} {row.max():>6d}   "
                  f"{min(snrs):+6.2f} .. {max(snrs):+6.2f} dB")
        rep.check("equal frames per class within each bin",
                  bool((grid.min(axis=1) == grid.max(axis=1)).all()))
        excl = meta["snr"]["excluded_bins"]
        print(f"      excluded: {excl['underpopulated'] + [int(k) for k in excl['incomplete']]}")

        print("finite / normalization")
        nonfinite, worst_power = 0, 0.0
        for a in range(0, n, _BLOCK):
            blk = x_ds[a:a + _BLOCK]
            nonfinite += int((~np.isfinite(blk)).sum())
            power = (blk.astype(np.float64) ** 2).sum(axis=2).mean(axis=1)
            if norm == "unit_power":
                worst_power = max(worst_power, float(np.abs(power - 1).max()))
        rep.check("no NaN/Inf", nonfinite == 0, f"{nonfinite} non-finite values")
        if norm == "unit_power":
            rep.check("per-frame power 1", worst_power < POWER_TOL, f"max |P-1| {worst_power:.2e}")

        print("rebuild")
        fe = meta["front_end"]
        spur_hz = meta["notch"]["centres_hz"].values()
        half = meta["notch"]["half_bins"]
        used = [c for c in caps if c["n_frames"]]
        rng = np.random.default_rng(args.seed)
        for i in sorted(rng.choice(len(used), size=min(N_REBUILD, len(used)), replace=False)):
            c = used[int(i)]
            frames, _ = front_end(root / c["file"], c["n_frames"], fe["k_shift_bins"], spur_hz,
                                  half)
            again = to_layout(frames, norm, x_ds.dtype)
            stored = x_ds[c["row_start"]:c["row_start"] + c["n_frames"]]
            err = float(np.abs(again - stored).max())
            rep.check(f"{c['file']} rows reproduce", err == 0.0, f"max abs diff {err:.2e}")

        print("PSD spot-check: QPSK at the top bin vs tx/QPSK.bin through the same front end")
        top = max(bins)
        rows = np.nonzero((z == top) & (cls == classes.index("QPSK")))[0]
        xq = x_ds[rows[0]:rows[-1] + 1]
    real = xq[..., 0] + 1j * xq[..., 1]

    manifest = json.loads((Path(args.tx_dir) / "manifest.json").read_text(encoding="utf-8"))
    _, tx_frames = load_tx(Path(args.tx_dir) / "QPSK.bin", manifest["f_off_cycles"], RML_FRAMES)
    tx_frames, _ = notch_frames(tx_frames, spur_hz, half)
    real_psd, tx_psd = avg_psd(real), avg_psd(tx_frames)
    stats = compare_psd(real_psd, tx_psd)
    one = compare_psd(avg_psd(real[:1]), tx_psd)
    print(f"      {len(real)} frames averaged at {top:+d} dB: in-band shape rms "
          f"{stats['psd_rms_db']:.2f} dB, max {stats['psd_max_db']:.2f} dB, bias "
          f"{stats['psd_bias_db']:+.2f} dB; occupied {stats['bw_tx_khz']:.0f} kHz (capture) "
          f"vs {stats['bw_rml_khz']:.0f} kHz (tx)")
    print(f"      single frame: rms {one['psd_rms_db']:.2f} dB (periodogram variance)")
    shape_ok = stats["psd_rms_db"] <= PSD_RMS_LIMIT_DB
    print(f"  [{'PASS' if shape_ok else 'FLAG'}] averaged in-band shape within "
          f"{PSD_RMS_LIMIT_DB} dB rms")

    # Spur residue: highest bin within +-8 kHz of each notch, against the out-of-band floor.
    freq = np.fft.fftshift(np.fft.fftfreq(FRAME_LEN, 1 / FS_RX)) / 1e3
    floor = float(np.median(real_psd[np.abs(freq) > 150]))
    print(f"      notch: {meta['notch']['n_bins']} bins, {meta['notch']['bins_per_spur']} per "
          f"spur ({meta['notch']['source']})")
    print("      spur residue near each notch (QPSK, top bin), dB over the out-of-band median:")
    for name, f_hz in meta["notch"]["centres_hz"].items():
        near = np.abs(freq - f_hz / 1e3) <= 8
        k = int(np.argmax(np.where(near, real_psd, 0)))
        print(f"        {name:<18} predicted {meta['notch']['predicted_hz'][name] / 1e3:+5.0f}k"
              f"  notch {f_hz / 1e3:+5.0f}k   peak {freq[k]:+5.0f}k "
              f"{10 * np.log10(real_psd[k] / floor):+5.1f} dB")

    print("\nlimiting class per bin (frames available before equalizing)")
    papr = meta["papr_db"]
    avail = meta["frame_counts"]["available_before_equalizing"]
    file_frames = max(c["frames_available"] for c in caps)
    print(f"      full-length capture = {file_frames} frames")
    for b in usable:
        counts = avail[str(b)]
        lo = min(counts.values())
        limiting = [k for k in classes if counts[k] == lo]
        n_caps = {k: sum(c["bin"] == b and c["class"] == k for c in caps) for k in classes}
        short = [c["file"] for c in caps if c["bin"] == b and c["class"] in limiting
                 and c["frames_available"] < file_frames]
        why = (f"SHORT RECORDING {short}" if short else
               "rung placement: fewer of its rungs fall in this bin")
        print(f"      {b:+4d}: {lo} frames = {', '.join(limiting)}  "
              f"({n_caps[limiting[0]]} capture(s); classes span {min(n_caps.values())}.."
              f"{max(n_caps.values())} captures)  -> {why}")
    print("      per class, frames available in each bin:")
    print("      " + f"{'class':<11}" + "".join(f"{b:>+6d}" for b in usable) + "   PAPR  bw corr")
    for k in classes:
        corr = next(c["snr_noise_bw_db"] - c["snr_inband_db"] for c in caps if c["class"] == k)
        print("      " + f"{k:<11}" + "".join(f"{avail[str(b)][k]:>6d}" for b in usable)
              + f"{papr[k]:>7.1f}{corr:>+8.1f}")

    print("\nsynthetic labels under the check_bins rule")
    with h5py.File(source_config.data.path, "r") as ref:
        z_syn = np.unique(ref[KEY_Z][:].reshape(-1))
    moved = [int(s) for s in z_syn if bin_centre(float(s)) != s]
    other = [int(s) for s in z_syn if snr_bucket(float(s)) != bin_centre(float(s))]
    rep.check("synthetic SNR labels stay in their own bin (bin_centre(s) == s)", not moved,
              f"{len(z_syn)} labels {int(z_syn.min())}..{int(z_syn.max())}, moved {moved}")
    rep.check("bin_centre and metrics.snr_bucket agree on every synthetic label", not other,
              f"differ at {other}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(freq, 10 * np.log10(tx_psd + 1e-20), lw=0.9, label="tx/QPSK.bin, notched")
    ax.plot(freq, 10 * np.log10(real_psd + 1e-20), lw=0.9, label=f"real QPSK @ {top:+d} dB")
    ax.set(xlabel="kHz", ylabel="dB (unit total power)",
           title=f"QPSK PSD, {len(real)} frames  --  in-band rms {stats['psd_rms_db']:.2f} dB")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "qpsk_psd_check.png", dpi=130)
    plt.close(fig)
    print(f"      figure -> {out / 'qpsk_psd_check.png'}")

    print("\n" + ("ALL CHECKS PASS" if not rep.failed else f"FAILED: {rep.failed}"))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
