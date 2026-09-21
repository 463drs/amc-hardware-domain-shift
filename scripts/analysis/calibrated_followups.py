"""calibrated_full follow-ups, from saved predictions only: the pre-registered real-set comparison,
error-structure similarity to the real set, the 7 x 8 cross-domain matrix, the positive control.

  python scripts/analysis/calibrated_followups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compare_all_conditions import (cross_domain_matrix, load_matrix,  # noqa: E402
                                            macro_accuracy_by_seed, paired_wilcoxon)
from scripts.compare_condition import balanced_accuracy, confusion, read_predictions  # noqa: E402
from src.data import MODULATION_CLASSES  # noqa: E402

CONDITIONS = ("baseline_100", "phase_noise_100", "iq_imbalance_100", "quantization_100",
              "all_100", "calibrated_100", "calibrated_full_100")
SEEDS = tuple(range(100, 110))
EVAL_PASS = ROOT / "outputs" / "eval_pass_20260919_161555" / "predictions"
# Scored after the real set was frozen, with theta fit to those captures: diagnostic, not blind.
DIAGNOSTIC = ROOT / "outputs" / "eval_diag_calibrated_full_20260921" / "predictions"
NON_BLIND = {"calibrated_full_100"}
MATCHED = (0, 16)
TOP_BIN = 16
N = len(MODULATION_CLASSES)


# Loading

def real_predictions(name: str = "real") -> tuple[dict, np.ndarray, np.ndarray]:
    """(condition, seed) -> predictions on a real set, plus the shared true and snr vectors."""
    z = np.load(EVAL_PASS / f"{name}.npz")
    true, snr = z["true"].astype(np.int64), z["snr"].astype(np.int64)
    pred = {(str(c), int(s)): z["pred"][i].astype(np.int64)
            for i, (c, s) in enumerate(zip(z["model_condition"], z["model_seed"]))}
    d = np.load(DIAGNOSTIC / f"{name}.npz")
    if not (np.array_equal(d["true"], true) and np.array_equal(d["snr"], snr)):
        raise RuntimeError(f"{name}: diagnostic rows do not align with the evaluation pass")
    for i, s in enumerate(SEEDS):
        pred[("calibrated_full_100", s)] = d["pred"][i].astype(np.int64)
    return pred, true, snr


def own_test(condition: str, seed: int):
    return read_predictions(ROOT / "runs" / condition / f"seed{seed}" / "predictions.npz", seed)


def real_balanced(pred: dict, true, snr, condition: str) -> np.ndarray:
    """24-class balanced accuracy over the matched 0..16 dB range, one entry per seed."""
    k = (snr >= MATCHED[0]) & (snr <= MATCHED[1])
    return np.array([balanced_accuracy(pred[(condition, s)][k], true[k]) for s in SEEDS])


# 1. The pre-registered comparison

def primary(pred, true, snr) -> None:
    print(f"\n## 1. real set, {MATCHED[0]}..{MATCHED[1]} dB, 24 classes, balanced accuracy; exact "
          "two-sided Wilcoxon, paired by seed, single comparison, no correction")
    ref = real_balanced(pred, true, snr, "all_100")
    for c in ("calibrated_full_100", "calibrated_100"):
        acc = real_balanced(pred, true, snr, c)
        t = paired_wilcoxon(acc - ref)
        flag = "  [DIAGNOSTIC, NOT BLIND]" if c in NON_BLIND else ""
        print(f"  {c} - all_100: median {100 * t['median']:+.2f} pp, IQR {100 * t['iqr']:.2f} "
              f"[{100 * t['q25']:+.2f}, {100 * t['q75']:+.2f}], p = {t['p']:.4f}, n = {t['n']}{flag}")
        print(f"    per seed (pp): {np.round(100 * (acc - ref), 2).tolist()}")
        print(f"    absolute: {c} {100 * np.median(acc):.2f}, all_100 {100 * np.median(ref):.2f}")


# 2. Error structure

def mean_confusion(rows) -> np.ndarray:
    """Seed-mean of row-normalised confusions; mean, not median, so every row still sums to 1."""
    return np.nanmean(np.stack([confusion(p, t) for p, t in rows]), axis=0)


def at_bin(p, t, z, b=TOP_BIN):
    k = z == b
    return p[k], t[k]


def similarity(a: np.ndarray, b: np.ndarray) -> dict:
    off = ~np.eye(N, dtype=bool)
    return {"spearman_recall": float(spearmanr(np.diag(a), np.diag(b)).statistic),
            "pearson_full": float(pearsonr(a.ravel(), b.ravel()).statistic),
            "pearson_off": float(pearsonr(a[off], b[off]).statistic)}


def error_structure(pred, true, snr) -> None:
    print(f"\n## 2. error structure at {TOP_BIN:+d} dB (seed-mean row-normalised confusion)")
    real = {c: [at_bin(pred[(c, s)], true, snr) for s in SEEDS]
            for c in ("baseline_100", "calibrated_full_100")}
    synth = {c: [at_bin(r.pred.astype(np.int64), r.true.astype(np.int64), r.snr)
                 for r in (own_test(c, s) for s in SEEDS)]
             for c in ("baseline_100", "calibrated_full_100")}
    target = mean_confusion(real["baseline_100"])
    pairs = {"(a) calibrated_full synth vs (b) baseline real": synth["calibrated_full_100"],
             "reference: baseline synth vs baseline real": synth["baseline_100"]}
    print(f"  {'comparison':<50}{'Spearman recall':>17}{'Pearson off-diag':>18}{'Pearson full':>14}")
    for label, rows in pairs.items():
        s = similarity(mean_confusion(rows), target)
        print(f"  {label:<50}{s['spearman_recall']:>17.3f}{s['pearson_off']:>18.3f}"
              f"{s['pearson_full']:>14.3f}")
        per_seed = [similarity(confusion(*rows[i]), confusion(*real["baseline_100"][i]))
                    for i in range(len(SEEDS))]
        spread = {k: [x[k] for x in per_seed] for k in per_seed[0]}
        print("    same-seed pairs, median [min, max]: " + ", ".join(
            f"{k} {np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]" for k, v in spread.items()))

    print("\n  recall at +16 dB: (a) calibrated_full synth, (b) baseline real, baseline synth")
    a, b = mean_confusion(synth["calibrated_full_100"]), target
    ref = mean_confusion(synth["baseline_100"])
    for i, name in enumerate(MODULATION_CLASSES):
        print(f"    {name:<10}{100 * a[i, i]:>7.1f}{100 * b[i, i]:>7.1f}{100 * ref[i, i]:>7.1f}")

    print("\n  real set: predicted / occurring, median over seeds (> 1 = sink, absorbs other classes;"
          " * marks > 1.5)")
    k_all = (snr >= MATCHED[0]) & (snr <= MATCHED[1])
    k_top = snr == TOP_BIN
    occurs = {k: np.bincount(true[m], minlength=N) for k, m in (("0-16", k_all), ("+16", k_top))}
    cols = [(c, k, m) for c in ("baseline_100", "calibrated_full_100")
            for k, m in (("0-16", k_all), ("+16", k_top))]
    print(f"    {'class':<10}" + "".join(f"{c[:12] + ' ' + k:>18}" for c, k, _ in cols)
          + f"{'recall base +16':>17}")
    for i, name in enumerate(MODULATION_CLASSES):
        ratios = [np.median([np.bincount(pred[(c, s)][m], minlength=N)[i] / occurs[k][i]
                             for s in SEEDS]) for c, k, m in cols]
        print(f"    {name:<10}" + "".join(f"{r:>17.2f}{'*' if r > 1.5 else ' '}" for r in ratios)
              + f"{100 * b[i, i]:>16.1f}%")


# 3. The 7 x 8 matrix

def full_matrix(pred, true, snr) -> None:
    matrix = load_matrix(CONDITIONS, "baseline_100")
    values = 100 * cross_domain_matrix(matrix)
    real = [100 * np.median(real_balanced(pred, true, snr, c)) for c in CONDITIONS]
    print(f"\n## 3. A (7 x 8): balanced accuracy, median over {len(SEEDS)} seeds (%); synthetic "
          f"columns SNR >= 0 dB, real column {MATCHED[0]}..{MATCHED[1]} dB. * = diagnostic, not blind")
    short = [c.replace("_100", "") for c in CONDITIONS]
    print(f"  {'train \\ eval':<22}" + "".join(f"{s[:13]:>14}" for s in short) + f"{'REAL':>10}")
    for i, c in enumerate(CONDITIONS):
        mark = "*" if c in NON_BLIND else " "
        print(f"  {c:<22}" + "".join(f"{v:>14.2f}" for v in values[i]) + f"{real[i]:>9.2f}{mark}")


# 4. Positive control

def positive_control() -> None:
    print("\n## 4. positive control: phase_noise_100_ex (sigma_w = 0.01 rad/sample), seed 100 only")
    matrix = load_matrix(("baseline_100",), "baseline_100")
    base = 100 * macro_accuracy_by_seed(matrix, "baseline_100")
    q1, med, q3 = np.percentile(base, [25, 50, 75])
    cell = ROOT / "runs" / "phase_noise_100_ex" / "seed100"

    def acc(path, seed=100):
        r = read_predictions(path, seed)
        k = r.snr >= 0
        return 100 * balanced_accuracy(r.pred[k], r.true[k])

    on_clean = acc(cell / "predictions__on_baseline_100.npz")
    own = acc(cell / "predictions.npz")
    print(f"  baseline_100 on its own test, SNR >= 0: median {med:.2f}, IQR {q3 - q1:.2f} "
          f"[{q1:.2f}, {q3:.2f}], seed 100 = {base[0]:.2f}")
    for label, v in (("ex model on the baseline test", on_clean), ("ex model on its own test", own)):
        d = v - med
        print(f"  {label:<30} {v:.2f}  -> vs baseline median {d:+.2f} pp, vs seed 100 "
              f"{v - base[0]:+.2f} pp; |diff| {'>' if abs(d) > q3 - q1 else '<='} IQR ({q3 - q1:.2f})")
    cost = np.array([acc(ROOT / "runs" / "baseline_100" / f"seed{s}"
                         / "predictions__on_phase_noise_100_ex.npz", s) for s in SEEDS]) - base
    t = paired_wilcoxon(cost / 100)
    print(f"  baseline models on the ex test (paired by seed): median {100 * t['median']:+.2f} pp, "
          f"IQR {100 * t['iqr']:.2f}, p = {t['p']:.4f}")


def main() -> None:
    pred, true, snr = real_predictions("real")
    primary(pred, true, snr)
    error_structure(pred, true, snr)
    full_matrix(pred, true, snr)
    positive_control()


if __name__ == "__main__":
    main()
