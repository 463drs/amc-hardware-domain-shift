"""All five conditions at once: each impairment's paired cost against the clean baseline, and the
5x5 cross-domain matrix. Inference only -- every number comes from predictions on disk."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_condition import (
    _HIGH_SNR_MIN as HIGH_SNR_MIN,
    DatasetFacts,
    GuardFailure,
    RunPredictions,
    balanced_accuracy,
    dataset_facts,
    per_class_recall,
    read_predictions,
)
from src.config import Config, resolve_config_path
from src.data import MODULATION_CLASSES
from src.predict import (discover_cells, expected_cells, predictions_name, run_cross_domain,
                         verify_cells)

_DEFAULT_CONDITIONS: Tuple[str, ...] = ("baseline_100", "phase_noise_100", "iq_imbalance_100",
                                        "quantization_100", "all_100")
_DEFAULT_BASELINE = "baseline_100"
_DEFAULT_OUT_DIR = "notebooks/outputs"
_DEFAULT_RUNS_ROOT = "runs"

# Degeneration shows as a GAP in the PSK family: a run either holds these classes or loses them
# outright. QAM recall is continuous, so no threshold separates degeneration from spread there.
_PSK_FAMILY: Tuple[str, ...] = ("8PSK", "16PSK", "32PSK")
_DEGENERATE_THRESHOLD = 0.25
_DEGENERATE_THRESHOLDS: Tuple[float, ...] = (0.20, 0.25, 0.30)

# Sequential = one hue light->dark for a magnitude; diverging = two hues about a neutral zero.
_SEQUENTIAL_CMAP = "Blues"
_DIVERGING_CMAP = "RdBu_r"


@dataclass(frozen=True)
class Matrix:
    """Every stored prediction the two parts read, plus what the datasets say about themselves."""

    conditions: Tuple[str, ...]
    baseline: str
    seeds: Tuple[int, ...]
    facts: Dict[str, DatasetFacts]
    runs: Dict[Tuple[str, str], Dict[int, RunPredictions]]

    def cell(self, train: str, evaluate: str) -> Dict[int, RunPredictions]:
        return self.runs[(train, evaluate)]


# Loading

def _load_configs(configs: Sequence[str | Path]) -> Dict[str, Config]:
    """Config per declared condition, in the given order. The label is authoritative, not the
    filename, exactly as in compare_condition.load_side."""
    loaded: Dict[str, Config] = {}
    for entry in configs:
        cfg = Config.from_yaml(resolve_config_path(str(entry)))
        label = cfg.experiment.condition
        if label in loaded:
            raise ValueError(f"two configs declare condition {label!r}; the matrix is keyed by "
                             f"the declared condition, so it cannot hold both.")
        loaded[label] = cfg
    return loaded


def score_matrix(configs: Sequence[str | Path] = _DEFAULT_CONDITIONS,
                 runs_root: str | Path = _DEFAULT_RUNS_ROOT, download: bool = False,
                 overwrite: bool = False, verbose: bool = True) -> List[Path]:
    """Write whatever cell of the matrix is not already on disk, one test loader per eval split.
    The diagonal is normally already there from src.predict.run_all and is left untouched."""
    loaded = _load_configs(configs)
    written: List[Path] = []
    for evaluate, eval_cfg in loaded.items():
        if verbose:
            print(f"scoring every model on {evaluate}'s test split ...", flush=True)
        written += run_cross_domain(list(loaded.values()), eval_cfg, Path(runs_root),
                                    download=download, overwrite=overwrite)
    return written


def load_matrix(configs: Sequence[str | Path] = _DEFAULT_CONDITIONS,
                baseline: str | Path = _DEFAULT_BASELINE,
                runs_root: str | Path = _DEFAULT_RUNS_ROOT) -> Matrix:
    """Read each condition's dataset facts and every (train, eval) cell's stored predictions."""
    runs_root = Path(runs_root)
    loaded = _load_configs(configs)
    baseline = Config.from_yaml(resolve_config_path(str(baseline))).experiment.condition
    if baseline not in loaded:
        raise ValueError(f"baseline {baseline!r} is not among the conditions {tuple(loaded)}")

    facts: Dict[str, DatasetFacts] = {}
    cells: Dict[str, List] = {}
    for name, cfg in loaded.items():
        facts[name] = dataset_facts(cfg.data.path)
        found = discover_cells(runs_root, name)
        verify_cells(expected_cells(cfg), found)      # every seed the config prescribes, no more
        cells[name] = found

    seed_sets = {name: frozenset(c.cell.seed for c in found) for name, found in cells.items()}
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"the conditions do not share one seed set, so nothing can be paired by "
                         f"seed: { {k: sorted(v) for k, v in seed_sets.items()} }")
    seeds = tuple(sorted(next(iter(seed_sets.values()))))

    runs: Dict[Tuple[str, str], Dict[int, RunPredictions]] = {}
    for train in loaded:
        for evaluate in loaded:
            filename = predictions_name(train, evaluate)
            per_seed: Dict[int, RunPredictions] = {}
            for c in cells[train]:
                npz = c.path / filename
                if not npz.exists():
                    raise FileNotFoundError(
                        f"{npz} missing: the {train} model is not scored on {evaluate}'s test "
                        f"split. Run scripts.compare_all_conditions.score_matrix() first.")
                per_seed[c.cell.seed] = read_predictions(npz, c.cell.seed)
            runs[(train, evaluate)] = per_seed

    return Matrix(conditions=tuple(loaded), baseline=baseline, seeds=seeds,
                  facts=facts, runs=runs)


# Guards

def check_guards(matrix: Matrix, verbose: bool = True) -> List[Tuple[str, bool, str]]:
    """Every way this comparison could measure something other than what it claims: the datasets
    must be different content on the same split, and every scored cell the same test rows."""
    checks: List[Tuple[str, bool, str]] = []

    def add(name: str, ok: object, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    pairs = [(a, b) for i, a in enumerate(matrix.conditions) for b in matrix.conditions[i + 1:]]
    for a, b in pairs:
        left, right = matrix.facts[a], matrix.facts[b]
        # Equal checksums mean a config error pointed two arms at the same file; the near-zero
        # delta that follows is an artefact that reads exactly like a confirmed null.
        add(f"{a} vs {b}: datasets differ", left.checksum != right.checksum,
            f"{left.checksum} vs {right.checksum}")
        # The split is derived from these, so equality is what makes both test sets the same rows.
        for key, x, y in (("subset_seed", left.subset_seed, right.subset_seed),
                          ("split_seed", left.split_seed, right.split_seed),
                          ("split", left.split, right.split),
                          ("frame count", left.n_frames, right.n_frames)):
            add(f"{a} vs {b}: {key} shared", x == y, f"{x} vs {y}")

    everything = [(k, r) for k, per_seed in matrix.runs.items() for r in per_seed.values()]
    (_, reference) = everything[0]
    misaligned = sorted({f"{train}->{evaluate}#{r.seed}" for (train, evaluate), r in everything
                         if not (np.array_equal(r.true, reference.true)
                                 and np.array_equal(r.snr, reference.snr))})
    add("test rows aligned across every cell", not misaligned,
        f"{reference.true.size} frames" + (f"; misaligned: {misaligned}" if misaligned else ""))

    if verbose:
        width = max(len(name) for name, _, _ in checks)
        print("--- guards ---")
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        raise GuardFailure(f"{len(failed)} guard(s) failed: {failed}. The comparison would not "
                           f"measure the cost of the impairments, so no number is reported.")
    return checks


def print_datasets(matrix: Matrix) -> None:
    """What each arm actually read, off the file's own attrs rather than off the config's intent."""
    print("--- what was compared ---")
    for name in matrix.conditions:
        d = matrix.facts[name]
        print(f"  {name}")
        print(f"    dataset   : {d.path.name}")
        print(f"    condition : {d.condition}")
        print(f"    theta     : {d.theta}")
        print(f"    checksum  : {d.checksum}{' (computed here)' if d.checksum_computed else ''}")
    print(f"  seeds: {list(matrix.seeds)}")


# Part 1 -- the paired cost of each condition

def macro_accuracy_by_seed(matrix: Matrix, train: str, evaluate: str | None = None) -> np.ndarray:
    """Macro (balanced) accuracy at SNR >= 0 dB, one entry per seed, in matrix.seeds order."""
    runs = matrix.cell(train, evaluate or train)
    values = []
    for seed in matrix.seeds:
        r = runs[seed]
        high = r.snr >= HIGH_SNR_MIN
        values.append(balanced_accuracy(r.pred[high], r.true[high]))
    return np.array(values)


def paired_deltas(matrix: Matrix) -> Dict[str, np.ndarray]:
    """condition -> its per-seed delta macro accuracy vs the baseline, same seed on both sides.
    The baseline's own entry is exactly zero, which is what makes it the reference row."""
    base = macro_accuracy_by_seed(matrix, matrix.baseline)
    return {c: macro_accuracy_by_seed(matrix, c) - base for c in matrix.conditions}


def psk_recall(matrix: Matrix, condition: str) -> np.ndarray:
    """(PSK class, seed) recall at SNR >= 0 dB for a condition's models on their own test split."""
    columns = [MODULATION_CLASSES.index(name) for name in _PSK_FAMILY]
    runs = matrix.cell(condition, condition)
    out = np.empty((len(columns), len(matrix.seeds)))
    for j, seed in enumerate(matrix.seeds):
        r = runs[seed]
        high = r.snr >= HIGH_SNR_MIN
        out[:, j] = per_class_recall(r.pred[high], r.true[high])[columns]
    return out


def degenerate_counts(matrix: Matrix,
                      thresholds: Sequence[float] = _DEGENERATE_THRESHOLDS
                      ) -> Dict[str, Dict[float, int]]:
    """condition -> threshold -> how many of the 3 x n_seeds PSK cells fall below it."""
    return {c: {t: int((psk_recall(matrix, c) < t).sum()) for t in thresholds}
            for c in matrix.conditions}


def unstable_pairs(counts: Dict[str, Dict[float, int]],
                   conditions: Sequence[str]) -> List[Tuple[str, str]]:
    """Condition pairs whose ORDER by degenerate count changes with the threshold; a tie turning
    into a strict order counts, since that is the row ordering ceasing to be readable."""
    return [(a, b) for i, a in enumerate(conditions) for b in conditions[i + 1:]
            if len({np.sign(counts[a][t] - counts[b][t]) for t in counts[a]}) > 1]


def summary_table(matrix: Matrix, threshold: float = _DEGENERATE_THRESHOLD,
                  counts: Dict[str, Dict[float, int]] | None = None) -> Dict[str, dict]:
    """One row per condition: median, mean and IQR of the paired deltas, and the degenerate count."""
    deltas = paired_deltas(matrix)
    counts = counts or degenerate_counts(matrix, (threshold,))
    rows = {}
    for c in matrix.conditions:
        d = deltas[c]
        q1, q3 = np.percentile(d, [25, 75])
        rows[c] = {"median": float(np.median(d)), "mean": float(d.mean()),
                   "iqr": float(q3 - q1), "degenerate": counts[c][threshold],
                   "deltas": d}
    return rows


def print_summary(matrix: Matrix, threshold: float = _DEGENERATE_THRESHOLD) -> Dict[str, dict]:
    counts = degenerate_counts(matrix, sorted({threshold, *_DEGENERATE_THRESHOLDS}))
    rows = summary_table(matrix, threshold, counts)
    n_seeds = len(matrix.seeds)
    n_cells = len(_PSK_FAMILY) * n_seeds
    header = f"PSK cells < {threshold:.2f}"

    print(f"\n=== Part 1: cost vs {matrix.baseline}, macro accuracy at SNR >= {HIGH_SNR_MIN} dB ===")
    print(f"  n = {n_seeds} seeds, paired (same seed on both sides). Deltas in percentage points.")
    print("  Median and mean are side by side on purpose: they diverge when a run degenerates.")
    print(f"\n  {'condition':<20}{'median':>9}{'mean':>9}{'IQR':>9}{header:>20}")
    for c in matrix.conditions:
        r = rows[c]
        cell = f"{r['degenerate']} / {n_cells}"
        print(f"  {c:<20}{100 * r['median']:>+9.2f}{100 * r['mean']:>+9.2f}{100 * r['iqr']:>9.2f}"
              f"{cell:>20}")

    print(f"\n--- threshold check: degenerate PSK cells, of {n_cells} "
          f"({len(_PSK_FAMILY)} classes x {n_seeds} seeds) ---")
    print(f"  PSK only ({', '.join(_PSK_FAMILY)}): QAM recall is continuously distributed, so no "
          f"threshold\n  separates degeneration from normal spread there.")
    print(f"\n  {'condition':<20}" + "".join(f"{f'< {t:.2f}':>9}" for t in _DEGENERATE_THRESHOLDS))
    for c in matrix.conditions:
        print(f"  {c:<20}" + "".join(f"{counts[c][t]:>9}" for t in _DEGENERATE_THRESHOLDS))

    flipped = unstable_pairs(counts, matrix.conditions)
    print(f"\n  row ordering by count is {'NOT stable' if flipped else 'STABLE'} across the three "
          f"thresholds.")
    for a, b in flipped:
        print(f"    {a} vs {b}: " + ", ".join(
            f"{counts[a][t]}-{counts[b][t]} at < {t:.2f}" for t in _DEGENERATE_THRESHOLDS))
    return rows


# Significance and seed-to-seed noise

def holm(p_values: ArrayLike) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER), in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((p.size - np.arange(p.size)) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(1.0, adjusted)
    return out


def benjamini_hochberg(p_values: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR), in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * p.size / np.arange(1, p.size + 1)
    out = np.empty_like(p)
    out[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return out


def paired_wilcoxon(differences: ArrayLike) -> dict:
    """Exact two-sided Wilcoxon signed-rank on per-seed paired differences, with median and IQR.
    All-zero differences carry no evidence either way, so p is 1 rather than an error."""
    d = np.asarray(differences, dtype=float)
    q1, median, q3 = np.percentile(d, [25, 50, 75])
    p = 1.0 if not np.any(d) else float(wilcoxon(d, method="exact").pvalue)
    return {"median": float(median), "q25": float(q1), "q75": float(q3),
            "iqr": float(q3 - q1), "p": p, "n": d.size}


def significance(matrix: Matrix) -> Dict[str, dict]:
    """Each condition's paired deltas vs the baseline, tested; Holm and BH over that family."""
    deltas = paired_deltas(matrix)
    family = [c for c in matrix.conditions if c != matrix.baseline]
    tests = {c: paired_wilcoxon(deltas[c]) for c in family}
    p = [tests[c]["p"] for c in family]
    for c, h, b in zip(family, holm(p), benjamini_hochberg(p)):
        tests[c].update(p_holm=float(h), p_bh=float(b))
    return tests


def class_recall_by_seed(matrix: Matrix, condition: str) -> np.ndarray:
    """(seed, class) recall at SNR >= 0 dB for a condition's models on their own test split."""
    runs = matrix.cell(condition, condition)
    rows = []
    for seed in matrix.seeds:
        r = runs[seed]
        high = r.snr >= HIGH_SNR_MIN
        rows.append(per_class_recall(r.pred[high], r.true[high]))
    return np.array(rows)


def noise_floor(matrix: Matrix, condition: str) -> float:
    """Median over seed pairs of the mean |recall_i - recall_j| over classes: how far two runs of
    the SAME condition sit apart, the yardstick a class-recall shift must exceed."""
    recall = class_recall_by_seed(matrix, condition)
    pairs = [(i, j) for i in range(len(recall)) for j in range(i + 1, len(recall))]
    return float(np.median([np.nanmean(np.abs(recall[i] - recall[j])) for i, j in pairs]))


def print_significance(matrix: Matrix) -> Dict[str, dict]:
    tests = significance(matrix)
    print(f"\n--- significance vs {matrix.baseline}: exact two-sided Wilcoxon signed-rank, paired "
          f"by seed, n = {len(matrix.seeds)} ---")
    print(f"  family of {len(tests)}; Holm controls FWER, BH controls FDR. With n = "
          f"{len(matrix.seeds)} the smallest attainable raw p is {2 / 2 ** len(matrix.seeds):.4f}.")
    print(f"\n  {'condition':<20}{'median':>9}{'IQR':>9}{'p':>9}{'p_holm':>9}{'p_BH':>9}"
          f"{'floor (pp)':>12}")
    print(f"  {matrix.baseline:<20}{'':>45}{100 * noise_floor(matrix, matrix.baseline):>12.2f}")
    for c, t in tests.items():
        print(f"  {c:<20}{100 * t['median']:>+9.2f}{100 * t['iqr']:>9.2f}{t['p']:>9.4f}"
              f"{t['p_holm']:>9.4f}{t['p_bh']:>9.4f}{100 * noise_floor(matrix, c):>12.2f}")
    print("  floor = median over seed pairs of mean |recall_i - recall_j| over classes, SNR >= "
          f"{HIGH_SNR_MIN} dB.")
    return tests


# Part 2 -- the cross-domain matrix

def cross_domain_matrix(matrix: Matrix) -> np.ndarray:
    """M[i, j] = median over seeds of macro accuracy at SNR >= 0 dB, for the model trained on
    conditions[i] scored on conditions[j]'s test split."""
    return np.array([[float(np.median(macro_accuracy_by_seed(matrix, train, evaluate)))
                      for evaluate in matrix.conditions]
                     for train in matrix.conditions])


def off_diagonal_cost(values: np.ndarray) -> np.ndarray:
    """Each cell minus its ROW's diagonal: what a model loses by leaving the data it was trained on."""
    return values - np.diag(values)[:, None]


def _print_matrix(values: np.ndarray, conditions: Sequence[str], title: str,
                  scale: float = 100.0, sign: str = "") -> None:
    width = max(11, max(len(c) for c in conditions) + 2)
    print(f"\n{title}")
    print(f"  {'train \\ eval':<20}" + "".join(f"{c:>{width}}" for c in conditions))
    for i, train in enumerate(conditions):
        print(f"  {train:<20}"
              + "".join(f"{scale * values[i, j]:>{sign}{width}.2f}" for j in range(len(conditions))))


def print_cross_domain(matrix: Matrix) -> np.ndarray:
    values = cross_domain_matrix(matrix)
    _print_matrix(values, matrix.conditions,
                  f"=== Part 2: macro accuracy at SNR >= {HIGH_SNR_MIN} dB, median over "
                  f"{len(matrix.seeds)} seeds (%) ===")
    _print_matrix(off_diagonal_cost(values), matrix.conditions,
                  "--- cell minus its row's diagonal: the cost of moving a model off its own "
                  "data (pp) ---", sign="+")
    return values


def plot_matrix(values: np.ndarray, conditions: Sequence[str], title: str, cbar_label: str,
                diverging: bool = False):
    """5x5 heatmap, annotated with its own numbers. Sequential for a magnitude, diverging about
    zero for a difference -- the colour job follows the quantity, not the figure."""
    scaled = 100.0 * np.asarray(values, dtype=float)
    if diverging:
        limit = float(np.nanmax(np.abs(scaled))) or 1.0
        cmap, vmin, vmax = _DIVERGING_CMAP, -limit, limit
    else:
        cmap, vmin, vmax = _SEQUENTIAL_CMAP, float(scaled.min()), float(scaled.max())

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    image = ax.imshow(scaled, cmap=cmap, vmin=vmin, vmax=vmax)

    colours = matplotlib.colormaps[cmap]
    span = (vmax - vmin) or 1.0
    # Sub-point differences need the second decimal; one would round most of them to "-0.0".
    spec = ("+" if diverging else "") + (".1f" if np.ptp(scaled) >= 10 else ".2f")
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            r, g, b, _ = colours((scaled[i, j] - vmin) / span)
            ink = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "#22262b"
            ax.text(j, i, f"{scaled[i, j]:{spec}}", ha="center", va="center",
                    fontsize=8.5, color=ink)

    ax.set_xticks(np.arange(len(conditions)), conditions, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(conditions)), conditions, fontsize=8)
    ax.set_xlabel("evaluated on"); ax.set_ylabel("trained on")
    ax.set_title(title, fontsize=10)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.046, label=cbar_label)
    fig.tight_layout()
    return fig


def plot_deltas(rows: Dict[str, dict], baseline: str, snr_min: int = HIGH_SNR_MIN):
    """The paired deltas themselves, one strip per condition, with the median marked. Showing all
    n points is what makes a mean-median gap visible as the one run that dragged it."""
    conditions = [c for c in rows if c != baseline]
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(conditions) + 1.6))
    ax.axvline(0.0, color="#9aa1a9", lw=1)
    for i, c in enumerate(conditions):
        values = 100.0 * rows[c]["deltas"]
        ax.plot(values, np.full(values.size, i), "o", ms=7, color="#3d63dd", alpha=0.45,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(100.0 * rows[c]["median"], i, "|", ms=22, mew=2.5, color="#22262b")
    ax.set_yticks(np.arange(len(conditions)), conditions, fontsize=9)
    ax.set_ylim(len(conditions) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"delta macro accuracy vs {baseline} (pp), one point per seed")
    ax.set_title(f"paired per-seed deltas, SNR >= {snr_min} dB (bar = median)", fontsize=10)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# Report

def compare_all_conditions(conditions: Sequence[str] = _DEFAULT_CONDITIONS,
                           baseline: str = _DEFAULT_BASELINE,
                           runs_root: str | Path = _DEFAULT_RUNS_ROOT,
                           threshold: float = _DEGENERATE_THRESHOLD, plot: bool = True,
                           out_dir: str | Path | None = _DEFAULT_OUT_DIR,
                           verbose: bool = True) -> dict:
    """Both parts end to end; returns a report dict. Guards run first and refuse to continue."""
    matrix = load_matrix(conditions, baseline, runs_root)
    if verbose:
        print_datasets(matrix)
        print()
    guards = check_guards(matrix, verbose=verbose)

    rows = print_summary(matrix, threshold) if verbose else summary_table(matrix, threshold)
    tests = print_significance(matrix) if verbose else significance(matrix)
    values = print_cross_domain(matrix) if verbose else cross_domain_matrix(matrix)

    report = {"matrix": matrix, "guards": guards, "summary": rows, "significance": tests,
              "noise_floor": {c: noise_floor(matrix, c) for c in matrix.conditions},
              "cross_domain": values, "cross_domain_cost": off_diagonal_cost(values),
              "degenerate": degenerate_counts(matrix), "figures": {}}

    if plot:
        report["figures"] = {
            "paired_deltas": plot_deltas(rows, baseline),
            "cross_domain": plot_matrix(
                values, matrix.conditions,
                f"macro accuracy at SNR >= {HIGH_SNR_MIN} dB, median over {len(matrix.seeds)} seeds",
                "macro accuracy (%)"),
            "cross_domain_cost": plot_matrix(
                off_diagonal_cost(values), matrix.conditions,
                "cell minus its row's diagonal: cost of leaving the training domain",
                "delta macro accuracy (pp)", diverging=True),
        }
        if out_dir is not None:
            directory = Path(out_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for name, fig in report["figures"].items():
                fig.savefig(directory / f"all_conditions_{name}.png", dpi=120)
            if verbose:
                print(f"\nsaved {len(report['figures'])} figure(s) to {directory.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="All five conditions: paired cost vs the "
                                                 "baseline, and the cross-domain matrix.")
    parser.add_argument("--conditions", nargs="+", default=list(_DEFAULT_CONDITIONS))
    parser.add_argument("--baseline", default=_DEFAULT_BASELINE)
    parser.add_argument("--runs-root", default=_DEFAULT_RUNS_ROOT)
    parser.add_argument("--threshold", type=float, default=_DEGENERATE_THRESHOLD,
                        help="Recall below which a (PSK class, seed) cell counts as degenerate.")
    parser.add_argument("--score", action="store_true",
                        help="Write the missing off-diagonal predictions first (inference only).")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")  # console is headless
    if args.score:
        score_matrix(args.conditions, args.runs_root)
    compare_all_conditions(conditions=args.conditions, baseline=args.baseline,
                           runs_root=args.runs_root, threshold=args.threshold,
                           plot=not args.no_plot, out_dir=args.out_dir, verbose=True)


if __name__ == "__main__":
    main()
