"""Receiver loss: balanced accuracy on a TX-only reference minus balanced accuracy on the real set.

The reference matches the frozen real set bin for bin: every tx/<class>.bin through the real set's
framing without the receiver (tx_only_diagnostic.tx_frames), white noise at each bin's centre
(noise-bw SNR), unit_power, 2000 frames per class per bin like the real set. The generator
mismatch with RadioML is common to both sets, so the difference is what the receiver chain costs.

  python scripts/analysis/receiver_loss.py            # scores once, then reuses the predictions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "measure"))

from build_real_hdf5 import to_layout  # noqa: E402
from tx_only_diagnostic import add_awgn, score, tx_frames  # noqa: E402
from scripts.compare_all_conditions import holm, paired_wilcoxon  # noqa: E402
from scripts.compare_condition import balanced_accuracy  # noqa: E402
from src.config import Config, resolve_config_path  # noqa: E402
from src.data import MODULATION_CLASSES, read_labels_and_snr  # noqa: E402
from src.predict import load_model  # noqa: E402

CONDITIONS = ("baseline_100", "phase_noise_100", "iq_imbalance_100", "quantization_100",
              "all_100", "calibrated_100", "calibrated_full_100")
SEEDS = tuple(range(100, 110))
BASELINE, PRIMARY_REF = "baseline_100", "all_100"
NON_BLIND = {"calibrated_full_100"}   # checkpoints postdate the real set; theta fit to it
REAL_SET = ROOT / "data" / "real_captures_2db_noise-bw.hdf5"
REAL_PASSES = (ROOT / "outputs" / "eval_pass_20260919_161555",
               ROOT / "outputs" / "eval_diag_calibrated_full_20260921")
OUT = ROOT / "outputs" / "tx_reference"
MATCHED = (0, 16)
NOISE_SEED = 20260922


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Real-set predictions, with the checkpoint each came from

def real_predictions() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """(condition, seed) -> predictions on the real set and -> the sha256 of the checkpoint."""
    pred, sha, true, snr = {}, {}, None, None
    for run in REAL_PASSES:
        z = np.load(run / "predictions" / "real.npz")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if true is None:
            true, snr = z["true"].astype(np.int64), z["snr"].astype(np.int64)
        elif not (np.array_equal(z["true"], true) and np.array_equal(z["snr"], snr)):
            raise RuntimeError(f"{run}: real rows do not align with {REAL_PASSES[0]}")
        conds = z["model_condition"] if "model_condition" in z else ["calibrated_full_100"] * len(SEEDS)
        seeds = z["model_seed"] if "model_seed" in z else SEEDS
        for i, (c, s) in enumerate(zip(conds, seeds)):
            pred[(str(c), int(s))] = z["pred"][i].astype(np.int64)
        for ck in manifest["checkpoints"]:
            sha[(ck["condition"], int(ck["seed"]))] = ck["sha256"]
    return pred, sha, true, snr


# The TX-only reference

def build_and_score(real_true: np.ndarray, real_snr: np.ndarray, real_sha: dict) -> Path:
    """Score every model on the reference, one (class, bin) block at a time; frames are not kept."""
    cfg = {c: Config.from_yaml(resolve_config_path(c)) for c in CONDITIONS}
    norm = {cfg[c].data.normalization for c in CONDITIONS}
    if len(norm) != 1:
        raise RuntimeError(f"conditions disagree on normalization: {norm}")
    norm = norm.pop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids, models, checkpoints = [], [], []
    for c in CONDITIONS:
        for s in SEEDS:
            path = ROOT / "runs" / c / f"seed{s}" / "best.pt"
            digest = sha256(path)
            if digest != real_sha[(c, s)]:
                raise RuntimeError(f"{path} is not the checkpoint behind its real-set predictions")
            ids.append((c, s))
            models.append(load_model(path, cfg[c], device))
            checkpoints.append({"condition": c, "seed": s, "path": str(path.relative_to(ROOT)),
                                "sha256": digest})

    k_off = json.loads((ROOT / "tx" / "manifest.json").read_text())["f_off_cycles"]
    bins = np.unique(real_snr)
    preds, trues, snrs = [], [], []
    for k, name in enumerate(MODULATION_CLASSES):
        frames = tx_frames(name, k_off)
        for b in bins:
            n = int(np.sum((real_true == k) & (real_snr == b)))
            if n > len(frames):
                raise RuntimeError(f"{name} {b:+d} dB: real set holds {n}, the TX loop gives {len(frames)}")
            rng = np.random.default_rng([NOISE_SEED, k, int(b) + 100])
            x = to_layout(add_awgn(frames[:n], float(b), rng), norm, np.float32)
            preds.append(score(models, x, norm, device).astype(np.int16))
            trues.append(np.full(n, k, np.int16))
            snrs.append(np.full(n, b, np.int16))
        print(f"\r  scored {k + 1}/{len(MODULATION_CLASSES)} classes x {len(bins)} bins "
              f"x {len(models)} models", end="", flush=True)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "predictions.npz"
    np.savez_compressed(out, pred=np.concatenate(preds, axis=1), true=np.concatenate(trues),
                        snr=np.concatenate(snrs), model_condition=np.array([c for c, _ in ids]),
                        model_seed=np.array([s for _, s in ids]))
    manifest = {"what": "TX-only reference matching the real set bin for bin (no receiver)",
                "real_set": str(REAL_SET.relative_to(ROOT)), "noise": "white complex AWGN, "
                "noise-bw SNR = bin centre (signal power / noise in 128 kHz)",
                "noise_seed": NOISE_SEED, "normalization": norm, "spur_notch": False,
                "tx_manifest_sha256": sha256(ROOT / "tx" / "manifest.json"),
                "tx_files_sha256": {c: sha256(ROOT / "tx" / f"{c}.bin") for c in MODULATION_CLASSES},
                "checkpoints": checkpoints}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


# Metrics

def per_seed(pred: np.ndarray, true: np.ndarray, snr: np.ndarray) -> dict:
    """Balanced accuracy over the matched range and per bin."""
    k = (snr >= MATCHED[0]) & (snr <= MATCHED[1])
    out = {"matched": balanced_accuracy(pred[k], true[k])}
    for b in np.unique(snr):
        out[int(b)] = balanced_accuracy(pred[snr == b], true[snr == b])
    return out


def _mi(v) -> str:
    q1, m, q3 = np.percentile(100 * np.asarray(v), [25, 50, 75])
    return f"{m:+6.2f} [{q1:+6.2f}, {q3:+6.2f}]"


def report(tx_path: Path, real: dict, real_true, real_snr) -> None:
    z = np.load(tx_path)
    tx_true, tx_snr = z["true"].astype(np.int64), z["snr"].astype(np.int64)
    for b in np.unique(real_snr):          # the reference must hold what the real set holds
        for k in range(len(MODULATION_CLASSES)):
            if np.sum((tx_true == k) & (tx_snr == b)) != np.sum((real_true == k) & (real_snr == b)):
                raise RuntimeError(f"class {k} bin {b:+d}: frame counts differ from the real set")
    tx = {(str(c), int(s)): z["pred"][i].astype(np.int64)
          for i, (c, s) in enumerate(zip(z["model_condition"], z["model_seed"]))}

    m_tx = {key: per_seed(p, tx_true, tx_snr) for key, p in tx.items()}
    m_real = {key: per_seed(real[key], real_true, real_snr) for key in tx}
    loss = {c: np.array([m_tx[(c, s)]["matched"] - m_real[(c, s)]["matched"] for s in SEEDS])
            for c in CONDITIONS}
    bins = [int(b) for b in np.unique(real_snr)]

    print(f"\n## receiver_loss = balanced acc (TX-only) - balanced acc (real), "
          f"{MATCHED[0]}..{MATCHED[1]} dB, 24 classes; median [q25, q75] over {len(SEEDS)} seeds, "
          "pp. * = diagnostic, not blind")
    print(f"  {'condition':<22}{'TX-only':>24}{'real':>24}{'receiver_loss':>24}{'IQR':>7}")
    for c in CONDITIONS:
        t = [m_tx[(c, s)]["matched"] for s in SEEDS]
        r = [m_real[(c, s)]["matched"] for s in SEEDS]
        q1, q3 = np.percentile(100 * loss[c], [25, 75])
        mark = "*" if c in NON_BLIND else " "
        print(f"  {c:<21}{mark}{_mi(t):>24}{_mi(r):>24}{_mi(loss[c]):>24}{q3 - q1:>7.2f}")

    print("\n## receiver_loss per bin, median over seeds (pp)")
    print(f"  {'condition':<22}" + "".join(f"{b:>+7d}" for b in bins))
    for c in CONDITIONS:
        print(f"  {c:<22}" + "".join(
            f"{100 * np.median([m_tx[(c, s)][b] - m_real[(c, s)][b] for s in SEEDS]):>+7.2f}"
            for b in bins))
    print(f"  {'TX-only, ' + BASELINE:<22}" + "".join(
        f"{100 * np.median([m_tx[(BASELINE, s)][b] for s in SEEDS]):>7.1f}" for b in bins))
    print(f"  {'real, ' + BASELINE:<22}" + "".join(
        f"{100 * np.median([m_real[(BASELINE, s)][b] for s in SEEDS]):>7.1f}" for b in bins))

    print("\n## paired Wilcoxon on receiver_loss (exact, two-sided, paired by seed)")
    for c, tag in (("calibrated_full_100", "PRIMARY, single pre-registered test, no correction; "
                    "DIAGNOSTIC, NOT BLIND"), ("calibrated_100", "secondary, uncorrected")):
        t = paired_wilcoxon(loss[c] - loss[PRIMARY_REF])
        print(f"  {c} - {PRIMARY_REF}: median {100 * t['median']:+.2f} pp, IQR {100 * t['iqr']:.2f} "
              f"[{100 * t['q25']:+.2f}, {100 * t['q75']:+.2f}], p = {t['p']:.4f}   ({tag})")
    family = [c for c in CONDITIONS if c != BASELINE]
    tests = [paired_wilcoxon(loss[c] - loss[BASELINE]) for c in family]
    adjusted = holm([t["p"] for t in tests])
    print(f"\n  each condition - {BASELINE}, Holm over {len(family)}:")
    for c, t, h in zip(family, tests, adjusted):
        mark = "  *" if c in NON_BLIND else ""
        print(f"    {c:<22} median {100 * t['median']:+6.2f} pp, IQR {100 * t['iqr']:5.2f}, "
              f"p = {t['p']:.4f}, p_holm = {h:.4f}{mark}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rescore", action="store_true", help="rebuild and rescore the reference")
    args = p.parse_args()

    real, sha, true, snr = real_predictions()
    cls, z, *_ = read_labels_and_snr(REAL_SET)
    if not (np.array_equal(cls, true) and np.array_equal(z, snr)):
        raise RuntimeError("stored real-set predictions do not align with the frozen real set")
    path = OUT / "predictions.npz"
    if args.rescore or not path.exists():
        path = build_and_score(true, snr, sha)
    report(path, real, true, snr)


if __name__ == "__main__":
    main()
