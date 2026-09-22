"""Bin occupancy of the captured SNR axis. Offline -- reads captures.json, drives nothing.

Because the TX files are peak-normalized, a fixed -x puts each class at a different SNR (a
12.3 dB spread across the PAPR range), so the real-domain SNR axis is NOT the rung index and
is not uniform across classes. This script bins the MEASURED SNR on RadioML's own grid and
reports which bins can carry an accuracy-vs-SNR point.

Three outcomes per bin, and only the first is usable for the curve:
  ok             every class present with at least MIN_FRAMES frames
  underpopulated every class present, but some below MIN_FRAMES -- confidence intervals are
                 not comparable across classes, so the point is flagged, not silently plotted
  incomplete     at least one class absent; the bin is EXCLUDED from the curve and flagged

Exclusion is local: a bad bin drops one point, it never fails the run. The primary metric --
balanced accuracy over all frames at SNR >= 0 -- does not depend on binning at all and is
reported separately; only that metric being uncomputable is treated as an error.

  python scripts/measure/check_bins.py --captures captures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_capture import SKIP_S
from make_tx import FRAME_LEN, FS_RX
from src.data import MODULATION_CLASSES

_DEFAULT_OUT_DIR = "outputs/tx_capture"
MIN_FRAMES = 30            # per class per bin; below this the CIs are not comparable
BIN_WIDTH_DB = 2           # RadioML's own SNR step, so the two curves overlay bin for bin
# Noise bandwidth all classes are referred to under --snr-ref noise-bw: the symbol rate, which
# is the same 128 kBd for every digital class and is what Es/N0 is defined against.
REF_BW_KHZ = 128.0


def bin_centre(snr_db: float, width: int = BIN_WIDTH_DB) -> int:
    """Centre of the bin holding snr_db. Centres stay on even SNRs whatever the width, so a
    coarser binning still lines up with RadioML's grid rather than straddling it."""
    return int(width * math.floor((snr_db + width / 2) / width))


def snr_of(rec: dict, mode: str) -> float:
    """Measured SNR under the chosen reference.

    `inband` is what was captured: signal power over the noise inside the class's OWN occupied
    band. That flatters a narrowband class -- FM occupies 7 kHz against 141 kHz for the digital
    classes, so it reads ~13 dB higher for the same drive. `noise-bw` refers every class to one
    noise bandwidth instead, which is what makes the axis comparable across classes."""
    if mode == "inband":
        return rec["snr_db"]
    bw = rec["band_hi_khz"] - rec["band_lo_khz"]
    return rec["snr_db"] + 10 * math.log10(bw / REF_BW_KHZ)


def frames_in(path: Path) -> int:
    """Usable 1024-sample frames in a uint8 I/Q capture, after the start-up skip."""
    usable = path.stat().st_size // 2 - int(SKIP_S * FS_RX)
    return max(usable // FRAME_LEN, 0)


def occupancy(records: list[dict], root: Path, mode: str,
              width: int = BIN_WIDTH_DB) -> dict[int, dict[str, int]]:
    """Frames per class per bin, {bin centre: {class: frames}}."""
    table: dict[int, dict[str, int]] = {}
    for r in records:
        b = bin_centre(snr_of(r, mode), width)
        table.setdefault(b, {}).setdefault(r["class"], 0)
        table[b][r["class"]] += frames_in(root / r["file"])
    return table


def classify_bins(table: dict[int, dict[str, int]], classes: list[str],
                  min_frames: int = MIN_FRAMES) -> tuple[list[int], list[int], list[int]]:
    """(ok, underpopulated, incomplete) bin centres, each sorted; see the module docstring."""
    ok, under, incomplete = [], [], []
    for b in sorted(table):
        counts = [table[b].get(c, 0) for c in classes]
        if min(counts) == 0:
            incomplete.append(b)
        elif min(counts) < min_frames:
            under.append(b)
        else:
            ok.append(b)
    return ok, under, incomplete


def main() -> int:
    p = argparse.ArgumentParser(description="Bin occupancy of the captured SNR axis.")
    p.add_argument("--captures", default="captures", help="Directory holding captures.json.")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="Where the table and PNG go.")
    p.add_argument("--bin-width", type=int, default=BIN_WIDTH_DB,
                   help="Bin width in dB. Centres stay on even SNRs, so a wider bin still "
                        "aligns with RadioML's grid; widen it when the per-class rung step "
                        "is too coarse to fill every bin.")
    p.add_argument("--snr-ref", choices=("inband", "noise-bw"), default="inband",
                   help="SNR definition: 'inband' as captured, or 'noise-bw' referred to a "
                        f"common {REF_BW_KHZ:.0f} kHz noise bandwidth so narrowband classes "
                        "lose their bandwidth advantage.")
    p.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                   help="Frames per class per bin below which a bin is underpopulated.")
    args = p.parse_args()

    root = Path(args.captures)
    doc = json.loads((root / "captures.json").read_text(encoding="utf-8"))
    records = doc["captures"]
    if not records:
        raise ValueError(f"{root / 'captures.json'} holds no captures")

    classes = [c for c in MODULATION_CLASSES if any(r["class"] == c for r in records)]
    table = occupancy(records, root, args.snr_ref, args.bin_width)
    bins = sorted(table)
    ok, under, incomplete = classify_bins(table, classes, args.min_frames)

    print(f"{len(records)} captures, {len(classes)} classes, {len(bins)} occupied bins "
          f"({args.bin_width} dB wide, centred on even SNR as in RadioML)")
    print(f"  usable for the curve : {len(ok)} bins  {ok}")
    if under:
        print(f"  underpopulated (<{args.min_frames} frames for some class): {under}")
    if incomplete:
        missing = {b: [c for c in classes if not table[b].get(c)] for b in incomplete}
        print(f"  incomplete, EXCLUDED : {len(incomplete)} bins")
        for b, miss in missing.items():
            shown = ", ".join(miss[:6]) + (f" +{len(miss) - 6} more" if len(miss) > 6 else "")
            print(f"      {b:>+4d} dB  missing {len(miss):>2d}/{len(classes)}: {shown}")

    # Primary metric: pooled over every frame at SNR >= 0, so it does not depend on the bins.
    pooled = {c: sum(frames_in(root / r["file"]) for r in records
                     if r["class"] == c and snr_of(r, args.snr_ref) >= 0) for c in classes}
    starved = [c for c, n in pooled.items() if n < args.min_frames]
    print(f"\n  primary metric (balanced accuracy, SNR >= 0, binning-independent):")
    print(f"    {min(pooled.values())} .. {max(pooled.values())} frames per class")
    if starved:
        print(f"    !! {len(starved)} class(es) below {args.min_frames} frames: {starved}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"bin_occupancy_{args.bin_width}db_{args.snr_ref}"
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["snr_bin_db", *classes, "min", "status"])
        for b in bins:
            row = [table[b].get(c, 0) for c in classes]
            status = ("incomplete" if b in incomplete else
                      "underpopulated" if b in under else "ok")
            w.writerow([b, *row, min(row), status])

    grid = np.array([[table[b].get(c, 0) for b in bins] for c in classes], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(bins) + 3), 0.32 * len(classes) + 2.2))
    ax.imshow(np.where(grid == 0, np.nan, grid), aspect="auto", cmap="viridis",
              interpolation="nearest")
    ax.set(xticks=range(len(bins)), yticks=range(len(classes)),
           xticklabels=[f"{b:+d}" for b in bins], yticklabels=classes,
           xlabel=f"measured SNR bin, dB ({args.bin_width} dB, even centres, RadioML grid)",
           title=f"frames per class per bin  --  {len(ok)} of {len(bins)} bins usable "
                 f"(blank = absent, red = <{args.min_frames})")
    for i in range(len(classes)):
        for j, b in enumerate(bins):
            n = int(grid[i, j])
            ax.text(j, i, "-" if n == 0 else f"{n}", ha="center", va="center", fontsize=6,
                    color="red" if 0 < n < args.min_frames else "w" if n else "0.5")
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=140)
    plt.close(fig)
    print(f"\n  {out / f'{stem}.csv'}  and  {out / f'{stem}.png'}")
    return 1 if starved else 0


if __name__ == "__main__":
    sys.exit(main())
