from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, resolve_config_path
from src.data import KEY_X, MODULATION_CLASSES
from src.metrics import snr_bucket
from src.predict import discover_cells

_DEFAULT_OUT_DIR = "notebooks/outputs"
_DEFAULT_RUNS_ROOT = "runs"
_N_CLASSES = len(MODULATION_CLASSES)
_HIGH_SNR_MIN = 0
_CHECKSUM_BATCH = 4096
_TOP_PAIRS = 8
_CHECKPOINT_BY_CONSTRUCTION = "best"

@dataclass(frozen=True)
class DatasetFacts:
    """What a dataset file says about itself, read from its attrs."""

    path: Path
    condition: str
    theta: str
    checksum: str
    checksum_computed: bool
    subset_seed: object
    split_seed: object
    split: Tuple[float, ...]
    n_frames: int


@dataclass(frozen=True)
class RunPredictions:
    """One cell's stored test-set predictions."""

    seed: int
    pred: np.ndarray
    true: np.ndarray
    snr: np.ndarray
    checkpoint: str
    run_id: str


@dataclass(frozen=True)
class Side:
    """One arm of the comparison: a config, the dataset it names, and its per-seed runs."""

    label: str
    config_path: Path
    dataset: DatasetFacts
    runs: Dict[int, RunPredictions]

    @property
    def seeds(self) -> List[int]:
        return sorted(self.runs)


def content_checksum(path: str | Path) -> Tuple[str, bool]:
    """(checksum, was_computed) for a dataset file. make_subset writes no `content_checksum`,
    so the clean subset's is computed here with the same construction rather than skipped."""
    with h5py.File(path, "r") as f:
        stored = f.attrs.get("content_checksum")
        if stored is not None:
            return str(stored), False
        digest = hashlib.blake2b(digest_size=16)
        x = f[KEY_X]
        for start in range(0, x.shape[0], _CHECKSUM_BATCH):
            digest.update(np.ascontiguousarray(x[start:start + _CHECKSUM_BATCH]).tobytes())
    return f"blake2b16:{digest.hexdigest()}", True


def dataset_facts(path: str | Path) -> DatasetFacts:
    """Read one dataset's self-description, including the checksum identifying its content."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. The guards read the dataset's own attrs, "
                                f"so the comparison cannot run without both files present.")
    checksum, computed = content_checksum(path)
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        declared = attrs.get("n_frames")
        n_frames = int(declared) if declared is not None else int(f[KEY_X].shape[0])
    return DatasetFacts(
        path=path,
        # The clean subset predates the condition machinery and declares neither.
        condition=str(attrs.get("condition", "(none declared -- clean subset)")),
        theta=str(attrs.get("theta", "(none declared -- R_0 only)")),
        checksum=checksum,
        checksum_computed=computed,
        subset_seed=attrs.get("subset_seed"),
        split_seed=attrs.get("split_seed"),
        split=tuple(float(v) for v in np.atleast_1d(attrs.get("split", []))),
        n_frames=n_frames,
    )


def load_side(config: str | Path, runs_root: str | Path = _DEFAULT_RUNS_ROOT) -> Side:
    """Load one side: its config, its dataset's facts, and every seed's stored predictions."""
    config_path = resolve_config_path(str(config))
    cfg = Config.from_yaml(config_path)
    label = cfg.experiment.condition

    try:
        cells = discover_cells(Path(runs_root), label)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Download the checkpoints and write predictions first:\n"
            f"    src.predict.run_all(Config.from_yaml(resolve_config_path({str(config)!r})), "
            f"condition={label!r})"
        ) from None

    runs: Dict[int, RunPredictions] = {}
    for cell in cells:
        # discover_cells globs by directory name; the logged config is what is authoritative.
        if cell.cell.condition != label:
            raise RuntimeError(f"{cell.path}: sits under {label!r} but its meta.json logs "
                               f"condition {cell.cell.condition!r}. Fix before comparing.")
        npz_path = cell.path / "predictions.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} missing -- the checkpoint is downloaded but "
                                    f"never scored. Run src.predict.run_all for {label!r}.")
        with np.load(npz_path) as z:
            runs[cell.cell.seed] = RunPredictions(
                seed=cell.cell.seed,
                pred=z["pred"], true=z["true"], snr=z["snr"],
                checkpoint=str(z["checkpoint"]) if "checkpoint" in z.files
                           else _CHECKPOINT_BY_CONSTRUCTION,
                run_id=str(z["run_id"]),
            )

    return Side(label=label, config_path=config_path,
                dataset=dataset_facts(cfg.data.path), runs=runs)


class GuardFailure(RuntimeError):
    """A guard failed, so the comparison would not measure what it claims to."""


def check_guards(baseline: Side, condition: Side) -> List[Tuple[str, bool, str]]:
    """Every way this comparison could measure something other than what it claims.
    Returns [(name, ok, detail), ...]; the caller reports them all, then refuses to continue."""
    checks: List[Tuple[str, bool, str]] = []

    def add(name: str, ok: object, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    b, c = baseline.dataset, condition.dataset

    # Equal checksums mean a config error pointed both runs at the same file; the near-zero
    # delta that follows is an artefact that reads exactly like a confirmed null.
    add("datasets differ (content_checksum)", b.checksum != c.checksum,
        f"{b.checksum} vs {c.checksum}"
        + ("  [computed, not declared]" if b.checksum_computed or c.checksum_computed else ""))

    # The split is derived from these, so equality is what makes both test sets the same frames.
    for name, left, right in (("subset_seed shared", b.subset_seed, c.subset_seed),
                              ("split_seed shared", b.split_seed, c.split_seed),
                              ("split fractions shared", b.split, c.split),
                              ("frame count shared", b.n_frames, c.n_frames)):
        add(name, left == right, f"{left} vs {right}")

    runs = list(baseline.runs.values()) + list(condition.runs.values())
    if not runs:
        add("predictions present", False, "neither side has a scored run")
        return checks

    reference = runs[0]
    misaligned = sorted({r.seed for r in runs[1:]
                         if not (np.array_equal(r.true, reference.true)
                                 and np.array_equal(r.snr, reference.snr))})
    add("test rows aligned across all runs", not misaligned,
        f"{reference.true.size} frames"
        + (f"; misaligned seeds: {misaligned}" if misaligned else ""))

    ckpt_b = sorted({r.checkpoint for r in baseline.runs.values()})
    ckpt_c = sorted({r.checkpoint for r in condition.runs.values()})
    add("same checkpoint on both sides", ckpt_b == ckpt_c and len(ckpt_b) == 1,
        f"{baseline.label}: {ckpt_b or ['-']}, {condition.label}: {ckpt_c or ['-']}")

    return checks


def pair_seeds(baseline: Side, condition: Side,
               unpaired: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """(paired, baseline-only, condition-only) seeds. With none in common, a difference of means
    is a DIFFERENT estimator -- seed variance stays in -- so it has to be asked for."""
    b, c = set(baseline.runs), set(condition.runs)
    common = sorted(b & c)
    if not common and not unpaired:
        raise ValueError(
            "no seed is present on both sides, so no paired delta exists.\n"
            f"  {baseline.label}: seeds {sorted(b)}\n"
            f"  {condition.label}: seeds {sorted(c)}\n"
            "Train the missing seeds, or pass unpaired=True / --unpaired to compare the two "
            "means instead -- a different estimator, with the seed variance left in."
        )
    return common, sorted(b - c), sorted(c - b)


def per_class_recall(pred: np.ndarray, true: np.ndarray,
                     n_classes: int = _N_CLASSES) -> np.ndarray:
    """Recall per class; NaN for a class with no frames in the selection."""
    recall = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = true == c
        if mask.any():
            recall[c] = float((pred[mask] == c).mean())
    return recall


def balanced_accuracy(pred: np.ndarray, true: np.ndarray, n_classes: int = _N_CLASSES) -> float:
    """Unweighted mean per-class recall. The split is stratified so it tracks raw accuracy, but
    it keeps a single collapsed class visible instead of averaged away by the other 23."""
    return float(np.nanmean(per_class_recall(pred, true, n_classes)))


def confusion(pred: np.ndarray, true: np.ndarray, n_classes: int = _N_CLASSES) -> np.ndarray:
    """Row-normalized confusion, entry (i, j) = P(predicted j | true i); NaN row if absent.
    Normalized before differencing, so the difference is one of probabilities, not of counts."""
    matrix = np.full((n_classes, n_classes), np.nan)
    for c in range(n_classes):
        mask = true == c
        if mask.any():
            matrix[c] = np.bincount(pred[mask].astype(np.int64),
                                    minlength=n_classes)[:n_classes] / mask.sum()
    return matrix


def snr_grid(snr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(grid, per-row bucket) via src.metrics' rule, so this bins SNR exactly as training did."""
    bucket_of = np.array([snr_bucket(float(v)) for v in snr])
    return np.array(sorted(set(bucket_of.tolist()))), bucket_of

def run_metrics(run: RunPredictions, grid: np.ndarray, bucket_of: np.ndarray,
                n_classes: int = _N_CLASSES) -> Dict[str, Any]:
    """Every metric the comparison differences, for one run. Same keys on both sides."""
    high = run.snr >= _HIGH_SNR_MIN
    return {
        "balanced_accuracy": balanced_accuracy(run.pred, run.true, n_classes),
        "balanced_accuracy_high": balanced_accuracy(run.pred[high], run.true[high], n_classes),
        "balanced_accuracy_by_snr": np.array([
            balanced_accuracy(run.pred[bucket_of == b], run.true[bucket_of == b], n_classes)
            for b in grid
        ]),
        "recall_high": per_class_recall(run.pred[high], run.true[high], n_classes),
        "confusion_high": confusion(run.pred[high], run.true[high], n_classes),
    }


def _subtract(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Elementwise left - right over matching metric keys."""
    return {k: left[k] - right[k] for k in left}


def _mean_over_seeds(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of a list of metric dicts, key by key."""
    return {k: np.mean([np.asarray(m[k]) for m in metrics], axis=0) for k in metrics[0]}


def top_confusion_pairs(diff: np.ndarray, n: int = _TOP_PAIRS,
                        class_names: Sequence[str] = MODULATION_CLASSES):
    """Off-diagonal cells that GREW the most: which class pairs started being confused.
    Diagonal excluded (it is the recall delta) and so are cells that did not grow."""
    off = np.asarray(diff, dtype=float).copy()
    np.fill_diagonal(off, np.nan)
    ranked = np.argsort(np.where(np.isnan(off), -np.inf, off), axis=None)[::-1][:n]
    rows, cols = np.unravel_index(ranked, off.shape)
    return [(class_names[int(i)], class_names[int(j)], float(off[i, j]))
            for i, j in zip(rows, cols) if np.isfinite(off[i, j]) and off[i, j] > 0]


def plot_delta_vs_snr(grid, baseline_curve, condition_curve, delta, spread=None,
                      labels=("baseline", "condition")):
    """Balanced accuracy vs SNR for both sides, and their difference, in two panels.
    Expectation: the impairment costs most where the signal is marginal, so delta deepens low."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    axes[0].plot(grid, baseline_curve, lw=1.8, marker="o", ms=5, label=labels[0])
    axes[0].plot(grid, condition_curve, lw=1.8, marker="s", ms=5, label=labels[1])
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("balanced accuracy vs SNR")
    axes[0].legend(fontsize=8)

    delta_pp = 100.0 * np.asarray(delta)
    axes[1].axhline(0.0, color="grey", lw=1)
    axes[1].plot(grid, delta_pp, lw=1.8, marker="o", ms=5, color="crimson")
    if spread is not None:
        band = 100.0 * np.asarray(spread)
        axes[1].fill_between(grid, delta_pp - band, delta_pp + band, color="crimson", alpha=0.15)
    axes[1].set_ylabel("delta balanced accuracy (pp)")
    axes[1].set_title(f"delta ({labels[1]} - {labels[0]})")

    for ax in axes:
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


def plot_delta_recall(recall_delta, class_names=MODULATION_CLASSES, snr_min=_HIGH_SNR_MIN):
    """Per-class recall delta at SNR >= snr_min, worst-hit first. Coloured by SIGN, not class:
    a cost and a gain are opposite states of one quantity, not two categories."""
    recall_delta = np.asarray(recall_delta)
    order = np.argsort(np.where(np.isnan(recall_delta), np.inf, recall_delta))
    values = 100.0 * recall_delta[order]

    fig, ax = plt.subplots(figsize=(7.5, 0.30 * order.size + 1.6))
    ax.barh(np.arange(order.size), values, height=0.72,
            color=["crimson" if v < 0 else "tab:blue" for v in values])
    ax.set_yticks(np.arange(order.size))
    ax.set_yticklabels([class_names[int(i)] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="grey", lw=1)
    ax.set_xlabel("delta recall (percentage points)")
    ax.set_title(f"per-class recall delta, SNR >= {snr_min} dB")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


def plot_confusion_difference(diff, class_names=MODULATION_CLASSES, snr_min=_HIGH_SNR_MIN):
    """(condition - baseline) row-normalized confusion; red = the pair became MORE likely.
    Diagonal blanked: it is the recall delta, and left in it would flatten every other cell."""
    off = np.asarray(diff, dtype=float).copy()
    np.fill_diagonal(off, np.nan)
    limit = float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else 0.0
    limit = limit or 1.0

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("0.9")

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    image = ax.imshow(100.0 * off, cmap=cmap, vmin=-100.0 * limit, vmax=100.0 * limit)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"confusion difference, SNR >= {snr_min} dB (diagonal blanked)")
    fig.colorbar(image, ax=ax, fraction=0.046, label="delta P(pred | true), pp")
    fig.tight_layout()
    return fig


# Report

def _print_header(baseline: Side, condition: Side) -> None:
    """What was actually compared, read off the files rather than off the intent."""
    print(f"=== {condition.label} vs {baseline.label} ===\n")
    print("--- what was compared ---")
    for role, side in (("baseline", baseline), ("condition", condition)):
        d = side.dataset
        print(f"  {role}")
        print(f"    config    : {side.config_path}")
        print(f"    dataset   : {d.path.name}")
        print(f"    condition : {d.condition}")
        print(f"    theta     : {d.theta}")
        print(f"    checksum  : {d.checksum}{' (computed here)' if d.checksum_computed else ''}")
        print(f"    seeds     : {side.seeds}")


def _print_deltas(report: dict) -> None:
    """Headline scalars. With n=1 there is no interval to print, and none is printed."""
    n, delta, spread = report["n"], report["delta"], report["spread"]
    estimator = "paired by seed" if report["paired"] else "UNPAIRED difference of means"

    print(f"\n--- delta balanced accuracy ({estimator}) ---")
    if report["paired"] and n == 1:
        print("  n=1: a single paired run. No interval, no spread, no significance test --")
        print("  this is one observation, not an estimate with an uncertainty.")
    elif report["paired"]:
        print(f"  n={n} paired seeds; sd is over those {n} per-seed deltas.")
    else:
        print(f"  {report['n_baseline']} baseline seed(s) vs {report['n_condition']} condition "
              f"seed(s); the seed variance is left IN this estimator.")

    for key, label in (("balanced_accuracy", "full SNR range"),
                       ("balanced_accuracy_high", f"SNR >= {_HIGH_SNR_MIN} dB")):
        line = f"  {label:<18}{100.0 * float(delta[key]):+8.2f} pp"
        if report["paired"] and n > 1:
            line += f"   (sd {100.0 * float(spread[key]):.2f} pp)"
        print(line)


def _print_snr_profile(report: dict) -> None:
    print("\n--- delta by SNR ---")
    print(f"  {'SNR':>5}{'baseline':>11}{'condition':>11}{'delta (pp)':>13}")
    for snr, base, cond, delta in zip(report["snr_grid"], report["baseline_by_snr"],
                                      report["condition_by_snr"],
                                      report["delta"]["balanced_accuracy_by_snr"]):
        print(f"  {snr:>5}{base:>11.3f}{cond:>11.3f}{100.0 * delta:>13.2f}")


def _print_class_profile(report: dict) -> None:
    recall = report["delta"]["recall_high"]
    order = np.argsort(np.where(np.isnan(recall), np.inf, recall))
    print(f"\n--- per-class recall delta, SNR >= {_HIGH_SNR_MIN} dB (worst first) ---")
    for i in order:
        print(f"  {MODULATION_CLASSES[int(i)]:>10}{100.0 * recall[int(i)]:>10.2f} pp")

    print(f"\n--- class pairs that started being confused (top {_TOP_PAIRS}) ---")
    for true_name, pred_name, value in report["top_confusions"]:
        print(f"  {true_name:>10} -> {pred_name:<10}{100.0 * value:>8.2f} pp")
    if not report["top_confusions"]:
        print("  none: no off-diagonal cell grew")


def compare_condition(baseline_config: str | Path, condition_config: str | Path,
                      runs_root: str | Path = _DEFAULT_RUNS_ROOT, unpaired: bool = False,
                      plot: bool = True, out_dir: str | Path | None = _DEFAULT_OUT_DIR,
                      verbose: bool = True) -> dict:
    """Full condition-vs-baseline comparison; returns a report dict. Guards run first and refuse
    to continue: a comparison that cannot vouch for which datasets it read is worth nothing."""
    baseline = load_side(baseline_config, runs_root)
    condition = load_side(condition_config, runs_root)

    if verbose:
        _print_header(baseline, condition)

    guards = check_guards(baseline, condition)
    if verbose:
        print("\n--- guards ---")
        for name, ok, detail in guards:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<37}{detail}")
    failed = [name for name, ok, _ in guards if not ok]
    if failed:
        raise GuardFailure(f"{len(failed)} guard(s) failed: {failed}. The comparison would not "
                           f"measure the cost of the impairment, so no delta is reported.")

    paired_seeds, only_baseline, only_condition = pair_seeds(baseline, condition, unpaired)
    paired = bool(paired_seeds)
    if verbose:
        print("\n--- seeds ---")
        print(f"  {'paired':<28}{paired_seeds}")
        print(f"  {baseline.label + ' only (excluded)':<28}{only_baseline or '-'}")
        print(f"  {condition.label + ' only (excluded)':<28}{only_condition or '-'}")

    reference = baseline.runs[paired_seeds[0]] if paired else next(iter(baseline.runs.values()))
    grid, bucket_of = snr_grid(reference.snr)

    base_metrics = {s: run_metrics(r, grid, bucket_of) for s, r in baseline.runs.items()}
    cond_metrics = {s: run_metrics(r, grid, bucket_of) for s, r in condition.runs.items()}

    if paired:
        per_seed = [_subtract(cond_metrics[s], base_metrics[s]) for s in paired_seeds]
        delta = _mean_over_seeds(per_seed)
        n = len(paired_seeds)
        spread = ({k: np.std([np.asarray(d[k]) for d in per_seed], axis=0, ddof=1)
                   for k in delta} if n > 1 else None)
        base_side = _mean_over_seeds([base_metrics[s] for s in paired_seeds])
        cond_side = _mean_over_seeds([cond_metrics[s] for s in paired_seeds])
    else:
        base_side = _mean_over_seeds(list(base_metrics.values()))
        cond_side = _mean_over_seeds(list(cond_metrics.values()))
        delta = _subtract(cond_side, base_side)
        per_seed, spread, n = [], None, 0

    report = {
        "baseline": baseline, "condition": condition, "guards": guards,
        "paired": paired, "n": n,
        "n_baseline": len(baseline.runs), "n_condition": len(condition.runs),
        "seeds": {"paired": paired_seeds, "only_baseline": only_baseline,
                  "only_condition": only_condition},
        "snr_grid": grid,
        "baseline_by_snr": base_side["balanced_accuracy_by_snr"],
        "condition_by_snr": cond_side["balanced_accuracy_by_snr"],
        "delta": delta, "spread": spread, "per_seed": per_seed,
        "top_confusions": top_confusion_pairs(delta["confusion_high"]),
        "figures": {},
    }

    if verbose:
        _print_deltas(report)
        _print_snr_profile(report)
        _print_class_profile(report)

    if plot:
        report["figures"] = {
            "delta_vs_snr": plot_delta_vs_snr(
                grid, base_side["balanced_accuracy_by_snr"], cond_side["balanced_accuracy_by_snr"],
                delta["balanced_accuracy_by_snr"],
                spread["balanced_accuracy_by_snr"] if spread else None,
                labels=(baseline.label, condition.label)),
            "delta_recall": plot_delta_recall(delta["recall_high"]),
            "confusion_difference": plot_confusion_difference(delta["confusion_high"]),
        }
        if out_dir is not None:
            directory = Path(out_dir)
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"compare_{condition.label}_vs_{baseline.label}"
            for name, fig in report["figures"].items():
                fig.savefig(directory / f"{stem}_{name}.png", dpi=120)
            if verbose:
                print(f"\nsaved {len(report['figures'])} figure(s) to {directory.resolve()}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired condition-vs-baseline comparison of a trained matrix.")
    parser.add_argument("--baseline", help="Config name/path for the baseline side.")
    parser.add_argument("--condition", help="Config name/path for the condition side.")
    parser.add_argument("--runs-root", default=_DEFAULT_RUNS_ROOT,
                        help="Root holding <condition>/seed<N>/predictions.npz.")
    parser.add_argument("--unpaired", action="store_true",
                        help="With no common seed, compare the means instead -- a DIFFERENT "
                             "estimator, with the seed variance left in.")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--rules", action="store_true",
                        help="Print the registered decision rules and exit.")
    args = parser.parse_args()

    if args.baseline is None or args.condition is None:
        parser.error("--baseline and --condition are required (use --rules to see the rules).")

    matplotlib.use("Agg")  # console is headless
    report = compare_condition(
        baseline_config=args.baseline, condition_config=args.condition,
        runs_root=args.runs_root, unpaired=args.unpaired,
        plot=not args.no_plot, out_dir=args.out_dir, verbose=True,
    )
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
