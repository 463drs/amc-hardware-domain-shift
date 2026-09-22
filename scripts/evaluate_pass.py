"""The single evaluation pass: 6 conditions x 10 seeds on their own synthetic test split, the clean
baseline test split, the real set and the diagnostic CFO-corrected real set. No training, no tuning.

Predictions are written first and every metric is computed from the saved predictions only. Output
goes to a new outputs/eval_pass_<utc>/; no existing prediction file or run directory is touched.

  python scripts/evaluate_pass.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_condition import balanced_accuracy, confusion, per_class_recall   # noqa: E402
from src.config import Config                                                          # noqa: E402
from src.data import (MODULATION_CLASSES, RadioMLDataset, _build_split_indices,         # noqa: E402
                      read_labels_and_snr)
from src.predict import _VERIFIED_DATA_KEYS, _normalize_data_value, load_model         # noqa: E402

CONDITIONS = ("baseline_100", "phase_noise_100", "iq_imbalance_100", "quantization_100",
              "all_100", "calibrated_100", "calibrated_full_100")
SEEDS = tuple(range(100, 110))
BASELINE = "baseline_100"
REAL = {"real": "data/real_captures_2db_noise-bw.hdf5",
        "real_cfodc": "data/real_captures_2db_noise-bw_cfodc.hdf5"}
MATCHED = (0, 16)          # SNR range both domains cover; every synthetic-vs-real gap uses it
TOP_BIN = 16               # confusion matrices are taken here, the top real bin
CHUNK = 8192               # frames read per HDF5 access; each is scored by every model before the next
BATCH = 1024
N = len(MODULATION_CLASSES)
# By construction these pairs are spectrally identical (make_tx.ANALOG_SRC); merged, 24 -> 22 classes.
PAIRS = {"AM-SSB": ("AM-SSB-WC", "AM-SSB-SC"), "AM-DSB": ("AM-DSB-WC", "AM-DSB-SC")}
MERGE = np.arange(N)
for _wc, _sc in PAIRS.values():
    MERGE[MODULATION_CLASSES.index(_sc)] = MODULATION_CLASSES.index(_wc)
SNR_NOTE = ("SNR bins are aligned by NOMINAL label only. Real SNR is measured (noise-bw, referred to "
            "128 kHz); RadioML SNR is an assigned label under different bandwidth and reference "
            "assumptions -- at the same label the real out-of-band floor sits ~26 dB higher. Equal "
            "bins do not mean equal physical conditions, and every synthetic-minus-real gap includes "
            "the contribution of this mismatch. It cannot be corrected post hoc.")


# Pre-flight (hard failures)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint(cond: str, seed: int) -> Path:
    """runs/<cond>/seed<N>/best.pt (W&B download), else the single local training dir's best.pt."""
    run = ROOT / "runs" / cond / f"seed{seed}" / "best.pt"
    if run.exists():
        return run
    local = [p for p in (ROOT / "outputs").glob(f"{cond}_{seed}*/best.pt")
             if p.parent.name == f"{cond}_{seed}" or p.parent.name.startswith(f"{cond}_{seed}_")]
    if len(local) != 1:
        raise FileNotFoundError(f"{cond} seed {seed}: {len(local)} local checkpoints, none in runs/")
    return local[0]


def preflight(cfgs: dict[str, Config]) -> list[dict]:
    """Resolve 60 checkpoints and refuse any that could have seen an evaluation set."""
    cells, newest = [], 0.0
    for cond in CONDITIONS:
        want = {k: _normalize_data_value(k, getattr(cfgs[cond].data, k)) for k in _VERIFIED_DATA_KEYS}
        for seed in SEEDS:
            path = checkpoint(cond, seed)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            data = (ck.get("config") or {}).get("data") or {}
            diffs = [k for k in _VERIFIED_DATA_KEYS if k in data and _normalize_data_value(k, data[k]) != want[k]]
            if not data or diffs:
                raise RuntimeError(f"{path}: training data config missing or differs in {diffs}")
            metric = ((ck.get("config") or {}).get("train") or {}).get("early_stopping_metric")
            if metric not in ("val_accuracy_snr_geq_0db", "val_accuracy", "val_loss"):
                raise RuntimeError(f"{path}: best.pt selected on {metric!r}, not a validation metric")
            newest = max(newest, path.stat().st_mtime)
            cells.append({"condition": cond, "seed": seed, "path": str(path.relative_to(ROOT)),
                          "sha256": sha256(path), "epoch": ck.get("epoch"),
                          "best_val": ck.get("best_metric"), "selected_on": metric})
    for cond in CONDITIONS:
        sp, *_ = _build_split_indices(cfgs[cond].data)
        if np.intersect1d(sp["test"], np.union1d(sp["train"], sp["val"])).size:
            raise RuntimeError(f"{cond}: test split overlaps train/val")
    trained = {_normalize_data_value("path", cfgs[c].data.path) for c in CONDITIONS}
    for name, p in REAL.items():
        with h5py.File(ROOT / p, "r") as f:
            created = _dt.datetime.fromisoformat(str(f.attrs["created_utc"])).timestamp()
            if name == "real_cfodc" and "diagnostic" not in f.attrs:
                raise RuntimeError(f"{p} is not marked diagnostic")
        if created <= newest or Path(p).name in trained:
            raise RuntimeError(f"{p} predates a checkpoint or is a training file")
    return cells


# Inference

@torch.no_grad()
def score(models: list, path: str, idx: np.ndarray, cls: np.ndarray, snr: np.ndarray,
          normalization: str, device, label: str) -> np.ndarray:
    """(n_models, n_rows) int16 predictions; frames go through RadioMLDataset's own preload path."""
    with h5py.File(ROOT / path, "r") as f:
        frame_len = int(f["X"].shape[1])
    out = np.empty((len(models), len(idx)), dtype=np.int16)
    for a in range(0, len(idx), CHUNK):
        b = min(a + CHUNK, len(idx))
        ds = RadioMLDataset(ROOT / path, idx[a:b], cls[a:b], snr[a:b], normalization, frame_len, preload=True)
        for s in range(0, b - a, BATCH):
            x = ds.x[s:s + BATCH].to(device, non_blocking=True)
            for m, model in enumerate(models):
                out[m, a + s:a + s + len(x)] = model(x).argmax(dim=1).cpu().numpy()
        print(f"\r  {label}: {b}/{len(idx)} frames x {len(models)} models", end="", flush=True)
    print()
    return out


# Metrics (from saved predictions only)

def bal(pred, true, merged: bool = False) -> float:
    if merged:
        pred, true = MERGE[pred], MERGE[true]
    return balanced_accuracy(pred, true)


def seed_metrics(pred: np.ndarray, true: np.ndarray, snr: np.ndarray) -> dict:
    """Every number for one model on one set."""
    hi_all, hi_matched = snr >= 0, (snr >= MATCHED[0]) & (snr <= MATCHED[1])
    m = {"bal24_snr_ge0": bal(pred[hi_all], true[hi_all]),
         "bal22_snr_ge0": bal(pred[hi_all], true[hi_all], True),
         "bal24_0_16": bal(pred[hi_matched], true[hi_matched]),
         "bal22_0_16": bal(pred[hi_matched], true[hi_matched], True)}
    for b in np.unique(snr):
        k = snr == b
        m[f"acc_bin_{int(b):+d}"] = float((pred[k] == true[k]).mean())
    top = snr == TOP_BIN
    m["confusion_top"] = confusion(pred[top], true[top]) if top.any() else None
    rec = per_class_recall(pred[hi_matched], true[hi_matched])
    for name, (wc, sc) in PAIRS.items():
        i, j = MODULATION_CLASSES.index(wc), MODULATION_CLASSES.index(sc)
        k = hi_matched & ((true == i) | (true == j))
        m[f"{name}_pair_recall_0_16"] = float(np.isin(pred[k], (i, j)).mean())
        m[f"{name}_within_pair_swap_0_16"] = float(
            ((true[k] == i) & (pred[k] == j) | (true[k] == j) & (pred[k] == i)).mean())
        m[f"{wc}_recall_0_16"], m[f"{sc}_recall_0_16"] = float(rec[i]), float(rec[j])
    return m


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "n": int(v.size)}


def main() -> int:
    p = argparse.ArgumentParser(description="Run the single evaluation pass.")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test only: every Nth row of each set, so all bins and classes stay present.")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / args.out_root / f"eval_pass_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    cfgs = {c: Config.from_yaml(ROOT / "configs" / f"{c}.yaml") for c in CONDITIONS}
    norms = {cfgs[c].data.normalization for c in CONDITIONS}
    if len(norms) != 1:
        raise RuntimeError(f"conditions disagree on normalization: {norms}")
    norm = norms.pop()
    print("pre-flight")
    cells = preflight(cfgs)
    print(f"  {len(cells)} checkpoints resolved and verified; no evaluation set seen in training or selection")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    manifest = {"created_utc": stamp, "git_head": git, "worktree_dirty": dirty, "device": str(device),
                "torch": torch.__version__, "normalization": norm, "matched_range_db": MATCHED,
                "top_bin_db": TOP_BIN, "snr_note": SNR_NOTE, "checkpoints": cells, "sets": {}}

    models = {c: [load_model(ROOT / x["path"], cfgs[c], device) for x in cells if x["condition"] == c]
              for c in CONDITIONS}
    all_models = [m for c in CONDITIONS for m in models[c]]

    # Each set's (path, rows, labels, snr) and which models score it.
    sets: dict[str, tuple] = {}
    for c in CONDITIONS:
        sp, cls, snr, *_ = _build_split_indices(cfgs[c].data)
        te = sp["test"]
        sets[f"synth_own:{c}"] = (cfgs[c].data.path, te, cls[te], snr[te], [c])
    sp, cls, snr, *_ = _build_split_indices(cfgs[BASELINE].data)
    te = sp["test"]
    sets["synth_clean"] = (cfgs[BASELINE].data.path, te, cls[te], snr[te], list(CONDITIONS))
    for name, path in REAL.items():
        cls, snr, *_ = read_labels_and_snr(ROOT / path)
        sets[name] = (path, np.arange(len(cls)), cls, snr, list(CONDITIONS))

    if args.limit:
        sets = {k: (p_, i[::args.limit], c_[::args.limit], z[::args.limit], cs)
                for k, (p_, i, c_, z, cs) in sets.items()}
        manifest["smoke_test_every_nth_row"] = args.limit
    preds = {}
    for name, (path, idx, cls, snr, conds) in sets.items():
        ms = [m for c in conds for m in models[c]] if conds != list(CONDITIONS) else all_models
        pr = score(ms, str(path), idx, cls, snr, norm, device, name)
        ids = [(c, s) for c in conds for s in SEEDS]
        np.savez_compressed(_mk(out / "predictions") / (name.replace(":", "__") + ".npz"),
                            pred=pr, true=cls.astype(np.int16), snr=snr.astype(np.int16),
                            rows=idx.astype(np.int64), model_condition=np.array([c for c, _ in ids]),
                            model_seed=np.array([s for _, s in ids]), source=str(path))
        with h5py.File(ROOT / path, "r") as f:
            manifest["sets"][name] = {"path": str(path), "rows": int(len(idx)),
                                      "created_utc": str(f.attrs.get("created_utc", "")),
                                      "bins": sorted(int(b) for b in np.unique(snr))}
        preds[name] = (pr, ids)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    report(out, sets, preds)
    return 0


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def report(out: Path, sets: dict, preds: dict) -> None:
    """Per-seed metrics -> per-condition median + IQR, gaps, confusions; all from the predictions."""
    per_seed = {}                               # (set, cond, seed) -> metrics
    for name, (pr, ids) in preds.items():
        _, _, cls, snr, _ = sets[name]
        for row, (c, s) in enumerate(ids):
            per_seed[(name.split(":")[0], c, s)] = seed_metrics(pr[row].astype(np.int64),
                                                               cls.astype(np.int64), snr)
    set_names = ("synth_own", "synth_clean", "real", "real_cfodc")

    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "condition", "seed", "metric", "value"])
        for (st, c, s), m in sorted(per_seed.items()):
            for k, v in m.items():
                if k != "confusion_top":
                    w.writerow([st, c, s, k, f"{v:.6f}"])

    rows = []
    for st in set_names:
        for c in CONDITIONS:
            keys = [k for k in per_seed[(st, c, SEEDS[0])] if k != "confusion_top"]
            for k in keys:
                rows.append({"set": st, "condition": c, "metric": k,
                             **summarize(per_seed[(st, c, s)].get(k) for s in SEEDS)})
    _write(out / "summary.csv", rows)

    gaps = []
    for c in CONDITIONS:
        for classes in ("24", "22"):
            k = f"bal{classes}_0_16"
            syn = np.array([per_seed[("synth_own", c, s)][k] for s in SEEDS])
            real = np.array([per_seed[("real", c, s)][k] for s in SEEDS])
            cfo = np.array([per_seed[("real_cfodc", c, s)][k] for s in SEEDS])
            for gname, v in (("synth_minus_real", syn - real), ("real_minus_cfodc", real - cfo),
                             ("cfodc_minus_real", cfo - real)):
                gaps.append({"condition": c, "classes": classes, "gap": gname, "range_db": "0..16",
                             "paired_over": "seed", **summarize(v)})
    _write(out / "gaps.csv", gaps)

    conf_dir = _mk(out / "confusion_top_bin")
    for st in set_names:
        for c in CONDITIONS:
            mats = [per_seed[(st, c, s)]["confusion_top"] for s in SEEDS]
            if mats[0] is None:
                continue
            med = np.nanmedian(np.stack(mats), axis=0)
            with (conf_dir / f"{st}__{c}__{TOP_BIN:+d}dB.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["true\\pred", *MODULATION_CLASSES])
                for i, name in enumerate(MODULATION_CLASSES):
                    w.writerow([name, *(f"{v:.4f}" for v in med[i])])

    text = render(per_seed, gaps)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten to {out}")


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def _mi(vals) -> str:
    s = summarize(vals)
    return f"{100 * s['median']:5.1f} [{100 * s['q25']:5.1f}, {100 * s['q75']:5.1f}]"


def render(per_seed: dict, gaps: list[dict]) -> str:
    """Markdown report: median [q25, q75] over 10 seeds, in percent."""
    L = ["# Evaluation pass", "", f"> {SNR_NOTE}", "",
         "Median [q25, q75] over 10 seeds, percent. 24-class is primary; 22-class merges the "
         "WC/SC pairs, which are indistinguishable by construction.", ""]
    L += ["## Balanced accuracy", "",
          "| condition | synth own, SNR>=0 (0-30) | synth own, 0-16 | synth clean, 0-16 | real, 0-16 | real CFO-corr, 0-16 |",
          "|---|---|---|---|---|---|"]
    for n in ("24", "22"):
        for c in CONDITIONS:
            v = lambda st, k: _mi(per_seed[(st, c, s)][k] for s in SEEDS)
            L.append(f"| {c} ({n}) | {v('synth_own', f'bal{n}_snr_ge0')} | {v('synth_own', f'bal{n}_0_16')} | "
                     f"{v('synth_clean', f'bal{n}_0_16')} | {v('real', f'bal{n}_0_16')} | "
                     f"{v('real_cfodc', f'bal{n}_0_16')} |")
    L += ["", "## Domain gap (paired by seed, 0-16 dB, percentage points)", "",
          "| condition | classes | synthetic - real | real - real CFO-corrected |", "|---|---|---|---|"]
    for c in CONDITIONS:
        for n in ("24", "22"):
            g = {x["gap"]: x for x in gaps if x["condition"] == c and x["classes"] == n}
            f = lambda x: f"{100 * x['median']:+5.1f} [{100 * x['q25']:+5.1f}, {100 * x['q75']:+5.1f}]"
            L.append(f"| {c} | {n} | {f(g['synth_minus_real'])} | {f(g['real_minus_cfodc'])} |")
    L += ["", "Real minus CFO-corrected is negative when removing the carrier offset helps; its "
          "magnitude is the share of the gap attributable to carrier offset.", ""]
    for st, title in (("synth_own", "own synthetic test"), ("real", "real (primary)"),
                      ("real_cfodc", "real, CFO-corrected (diagnostic)")):
        bins = sorted(int(k[8:]) for k in per_seed[(st, CONDITIONS[0], SEEDS[0])] if k.startswith("acc_bin_"))
        L += [f"## Accuracy per 2 dB bin, {title} (median over seeds)", "",
              "| condition | " + " | ".join(f"{b:+d}" for b in bins) + " |", "|---|" + "---|" * len(bins)]
        for c in CONDITIONS:
            L.append(f"| {c} | " + " | ".join(
                f"{100 * np.median([per_seed[(st, c, s)][f'acc_bin_{b:+d}'] for s in SEEDS]):.1f}"
                for b in bins) + " |")
        L.append("")
    L += ["## AM pairs (0-16 dB)", "",
          "| condition | set | AM-SSB pair recall | SSB within-pair swaps | AM-DSB pair recall | DSB within-pair swaps |",
          "|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        for st in ("synth_own", "real", "real_cfodc"):
            v = lambda k: _mi(per_seed[(st, c, s)][k] for s in SEEDS)
            L.append(f"| {c} | {st} | {v('AM-SSB_pair_recall_0_16')} | {v('AM-SSB_within_pair_swap_0_16')} | "
                     f"{v('AM-DSB_pair_recall_0_16')} | {v('AM-DSB_within_pair_swap_0_16')} |")
    L += ["", f"Confusion matrices at {TOP_BIN:+d} dB (median over seeds, row-normalized): "
          "confusion_top_bin/<set>__<condition>__+16dB.csv", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
