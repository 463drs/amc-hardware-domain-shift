"""Sanity check + acceptance criterion for the data module.

Plots I-vs-Q constellations for one high-SNR frame of every class and prints the
class-name -> index mapping taken from the dataset ordering (NOT from memory).

Callable both ways with identical console output:

    # notebook -- saves a PNG (dynamic name) AND displays it inline; returns the Figure
    from scripts.sanity_check import plot_constellations
    fig = plot_constellations(config="baseline")     # config name or path
    fig = plot_constellations(path="data/GOLD_XYZ_OSC.0001_1024.hdf5", snr=30)

    # console -- saves a PNG headlessly (no window), same printed report
    python scripts/sanity_check.py --config baseline
    python scripts/sanity_check.py --path data/GOLD_XYZ_OSC.0001_1024.hdf5 --snr 30 --sps 8

Expected shapes if labels are aligned:
    BPSK   -> 2 points
    QPSK   -> 4 points
    16QAM  -> 4x4 grid
    256QAM -> dense 16x16 grid
If these look wrong, the column->name mapping is misaligned and nothing downstream is valid.

IMPORTANT -- RadioML 2018.01A is oversampled (~8 samples/symbol) with pulse shaping.
Scattering all 1024 raw samples shows the inter-symbol transition trajectory, which smears
the discrete constellation into a cloud. We therefore downsample to the symbol rate
(every --sps-th sample) for the acceptance figure; that is a VISUALIZATION aid only and
does not touch the training data (the model consumes the full 1024-sample frames). Use
--sps 1 to see the raw, smeared frame.

We plot a SINGLE frame per class: each frame is phase-coherent, so its symbols land on the
true constellation. (Overlaying several frames would NOT help -- each frame has an
independent random carrier phase, so stacking them just rotates and smears the points.)
A single 1024-sample frame yields ~1024/sps symbols (~128 at sps=8), so BPSK/QPSK/16QAM
resolve cleanly, while 64/128/256QAM appear as a dense filled square rather than a fully
resolved lattice -- that is a symbol-count limit of one frame, not a labelling problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_data_path
from src.data import (
    KEY_X,
    MODULATION_CLASSES,
    NORMALIZERS,
    read_labels_and_snr,
)

_DEFAULT_PATH = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
_DEFAULT_OUT_DIR = "outputs"
# RadioML 2018.01A oversampling factor. Used only to downsample for the constellation view.
_DEFAULT_SPS = 8
# Classes whose constellation shape is unambiguous -> used as the printed acceptance check.
_KEY_CHECKS = {"BPSK": "2 points", "QPSK": "4 points",
               "16QAM": "4x4 grid", "256QAM": "dense 16x16 grid"}


def _symbol_points(
    x_ds: h5py.Dataset, row: int, normalize, sps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the symbol-rate I/Q constellation points from a single frame."""
    raw = x_ds[int(row)]                                  # (T, 2) = (time, [I, Q])
    iq = np.ascontiguousarray(raw.T, dtype=np.float32)    # (2, T) channels-first
    iq = normalize(torch.from_numpy(iq)).numpy()
    return iq[0, ::sps], iq[1, ::sps]                     # symbol-rate downsample of I, Q


def _figure_name(tag: str, snr: int, sps: int, normalization: str) -> str:
    """Dynamic filename encoding what the figure was generated from."""
    return f"constellations_{tag}_snr{snr}_sps{sps}_{normalization}.png"


def _running_in_notebook() -> bool:
    """True when executing inside a Jupyter/IPython notebook kernel (so we should display)."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def print_class_mapping() -> None:
    """Print the column-index -> modulation mapping (canonical DeepSig order).

    The classes with an unambiguous constellation shape are flagged as the acceptance
    checks. Independent of any data file -- it reports the fixed ordering the code assigns
    to the one-hot Y columns.
    """
    print("=== class index -> modulation (canonical DeepSig order) ===")
    for i, name in enumerate(MODULATION_CLASSES):
        flag = f"   <-- expect {_KEY_CHECKS[name]}" if name in _KEY_CHECKS else ""
        print(f"  {i:2d}  {name}{flag}")


def plot_constellations(
    config: str | Path | None = None,
    path: str | Path | None = None,
    snr: int = 30,
    sps: int = _DEFAULT_SPS,
    normalization: str = "unit_power",
    out: str | Path | None = None,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    show: bool | None = None,
    verbose: bool = True,
) -> "matplotlib.figure.Figure":
    """Render the per-class constellation grid; save it and (in a notebook) display it.

    Parameters
    ----------
    config       : config name/path; its data.path selects the HDF5 file.
    path         : explicit HDF5 path, used only when `config` is None.
    snr, sps     : SNR (dB) of the frames to plot, and symbol-rate downsample factor.
    normalization: per-frame normalization method (a key of src.data.NORMALIZERS).
    out          : explicit output PNG path. If None, a dynamic name is written into
                   `out_dir`, encoding the config/dataset tag + snr + sps + normalization,
                   so the filename tells you how it was generated.
    show         : force display on/off. Default None auto-detects: display when running in
                   a notebook kernel, skip on the console.
    verbose      : print progress (config / rendering / saved). The class-index mapping is
                   printed separately by print_class_mapping().

    Returns the matplotlib Figure (handy in notebooks for further tweaking).
    """
    if normalization not in NORMALIZERS:
        raise ValueError(f"Unknown normalization {normalization!r}. Available: {sorted(NORMALIZERS)}.")

    data_path, config_path = resolve_data_path(config, path, _DEFAULT_PATH)
    normalize = NORMALIZERS[normalization]
    # Tag the output after the config (if used) else the dataset file, for a descriptive name.
    tag = config_path.stem if config_path is not None else Path(data_path).stem

    class_idx, snr_all, _, n_classes = read_labels_and_snr(data_path)
    if n_classes != len(MODULATION_CLASSES):
        raise RuntimeError(
            f"File has {n_classes} classes but {len(MODULATION_CLASSES)} canonical names."
        )
    if snr not in np.unique(snr_all):
        raise ValueError(f"SNR {snr} not in dataset. Available: {np.unique(snr_all).tolist()}")

    if verbose:
        if config_path is not None:
            print(f"Using config at the path: {config_path}")
        print(f"=== rendering constellations at SNR = {snr} dB "
              f"(sps={sps}, normalization={normalization}) ===")

    # --- render one panel per class ---
    n_cols = 6
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    axes = axes.ravel()

    with h5py.File(data_path, "r") as f:
        x_ds = f[KEY_X]
        for c in range(n_classes):
            candidates = np.nonzero((class_idx == c) & (snr_all == snr))[0]
            if candidates.size == 0:
                raise RuntimeError(f"No frame for class {c} at SNR {snr}.")
            pi, pq = _symbol_points(x_ds, int(candidates[0]), normalize, sps)
            ax = axes[c]
            ax.scatter(pi, pq, s=4, alpha=0.5)
            ax.set_title(f"{c}: {MODULATION_CLASSES[c]}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    subtitle = "raw samples" if sps == 1 else f"symbol-rate (every {sps}th sample)"
    fig.suptitle(f"RadioML 2018.01A constellations @ {snr} dB -- {subtitle}\n"
                 f"{tag}  (norm={normalization})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # --- save (dynamic name unless an explicit path is given) ---
    if out is not None:
        out_path = Path(out)
    else:
        out_path = Path(out_dir) / _figure_name(tag, snr, sps, normalization)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    if verbose:
        print(f"\nsaved: {out_path.resolve()}")
        print("ACCEPTANCE: verify BPSK=2 pts, QPSK=4 pts, 16QAM=4x4, 256QAM~16x16 in the figure.")

    # --- display in a notebook; stay headless on the console ---
    if show is None:
        show = _running_in_notebook()
    if show:
        plt.show()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Constellation sanity check.")
    parser.add_argument("--path", default=None, help="HDF5 path.")
    parser.add_argument("--config", default=None,
                        help="Config name (e.g. 'baseline') or path; its data.path is used.")
    parser.add_argument("--snr", type=int, default=30, help="SNR (dB) of the frames to plot.")
    parser.add_argument("--sps", type=int, default=_DEFAULT_SPS,
                        help="Samples per symbol to downsample for the constellation (1 = raw).")
    parser.add_argument("--normalization", default="unit_power", choices=sorted(NORMALIZERS))
    parser.add_argument("--out", default=None,
                        help="Explicit output PNG path (default: dynamic name in --out-dir).")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR,
                        help="Directory for the auto-named PNG.")
    args = parser.parse_args()

    # Console is headless: render to a file, never try to open a window.
    matplotlib.use("Agg")
    print_class_mapping()
    plot_constellations(
        config=args.config, path=args.path, snr=args.snr, sps=args.sps,
        normalization=args.normalization, out=args.out, out_dir=args.out_dir,
        show=False, verbose=True,
    )


if __name__ == "__main__":
    main()
