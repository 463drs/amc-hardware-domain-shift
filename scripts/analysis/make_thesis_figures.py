"""Thesis figures for the domain comparison: fixed classes and bins -> PNGs, a long-form summary CSV,
the per-class aggregate table and the ADC-loading table, all under outputs/figures/. Reads data only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import domain_compare as dc     # noqa: E402

CLASSES = ("BPSK", "QPSK", "16QAM", "FM")
BINS = (0, 16)
TABLE_BINS = (16,)         # the aggregate table: every class at the top real bin
OUT_DIR = "outputs/figures"

# The one style block: DejaVu ships with matplotlib, so fonts render identically everywhere.
STYLE = {
    "figure.figsize": (6.3, 2.6),         # A4 text width (16 cm) at a third of a page
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "figure.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "legend.frameon": False, "lines.linewidth": 1.0,
    "axes.linewidth": 0.6, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
}


def main() -> int:
    p = argparse.ArgumentParser(description="Write the domain-comparison thesis figures.")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--classes", nargs="+", default=list(CLASSES))
    p.add_argument("--bins", nargs="+", type=int, default=list(BINS))
    p.add_argument("--table-bins", nargs="+", type=int, default=list(TABLE_BINS))
    p.add_argument("--n-frames", type=int, default=dc.N_FRAMES)
    p.add_argument("--skip-figures", action="store_true", help="Only the aggregate table.")
    args = p.parse_args()

    out = dc.ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(STYLE)

    summary = []
    if not args.skip_figures:
        for cls in args.classes:
            for b in args.bins:
                for name, fn in dc.FIGURES.items():
                    kw = {} if name == "spectrogram" else {"n_frames": args.n_frames}
                    fig, nums = fn(cls, b, title=False, **kw)
                    png = out / f"{name}_{cls}_{b:+d}dB.png"
                    fig.savefig(png)
                    summary += [{"figure": name, "class": cls, "snr_bin": b, "metric": k,
                                 "value": v} for k, v in nums.items()]
                    print(f"  {png.relative_to(dc.ROOT)}")
        print(f"summary -> {dc.write_csv(summary, out / 'domain_summary.csv')}")

    rows = dc.aggregate_table(args.table_bins, n_frames=args.n_frames)
    print("\n" + dc.format_table(rows))
    print(f"\naggregate -> {dc.write_csv(rows, out / 'domain_aggregate.csv')}")

    adc = dc.adc_loading_table(n_frames=args.n_frames)      # every class x usable real bin
    print("\n" + dc.format_adc_table(adc))
    print(f"\nADC loading -> {dc.write_csv(adc, out / 'adc_loading.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
