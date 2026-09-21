"""Generator or receiver? Score the transmitted files with no receiver in the path.

Each tx/<class>.bin goes through the real set's own framing minus the RTL-SDR: f_off removed and
decimated to 1.024 MS/s (make_tx.decimate_to_rx), 1024-sample frames, the training normalizer.
No spur notch, since the spurs are receiver artefacts. Scored clean and with AWGN at the real
top bin's SNR (noise-bw definition), next to the real set and baseline's own synthetic test.

  python scripts/analysis/tx_only_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "measure"))

from build_real_hdf5 import to_layout  # noqa: E402
from check_bins import REF_BW_KHZ  # noqa: E402
from make_tx import FRAME_LEN, FS_RX, decimate_to_rx, to_frames  # noqa: E402
from scripts.compare_condition import read_predictions  # noqa: E402
from src.config import Config, resolve_config_path  # noqa: E402
from src.data import BATCHED_NORMALIZERS, MODULATION_CLASSES  # noqa: E402
from src.predict import load_model  # noqa: E402

TX_DIR = ROOT / "tx"
REAL = ROOT / "outputs" / "eval_pass_20260919_161555" / "predictions" / "real.npz"
SEEDS = tuple(range(100, 110))
SNR_DB = 16                # the real set's top bin
N = len(MODULATION_CLASSES)
GAP_PP = 15.0              # a drop this large between two stages is called a failure there
NOISE_SEED = 20260921


def tx_frames(name: str, k_off: int) -> np.ndarray:
    """One TX loop -> complex frames at 1.024 MS/s, f_off removed, no receiver."""
    raw = np.fromfile(TX_DIR / f"{name}.bin", dtype=np.int8).astype(np.float64)
    x = raw[0::2] + 1j * raw[1::2]
    return to_frames(decimate_to_rx(x, k_off), x.size // 2 // FRAME_LEN)


def add_awgn(frames: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White complex noise with signal power / noise-in-128-kHz = snr_db (the noise-bw SNR)."""
    p_sig = np.mean(np.abs(frames) ** 2)
    var = p_sig / 10 ** (snr_db / 10) * (FS_RX / (REF_BW_KHZ * 1e3))
    noise = rng.normal(size=frames.shape) + 1j * rng.normal(size=frames.shape)
    return frames + noise * np.sqrt(var / 2)


@torch.no_grad()
def score(models, x: np.ndarray, normalization: str, device) -> np.ndarray:
    """(n_models, n_frames) predictions for stored-layout (N, T, 2) frames, read as RadioMLDataset does."""
    t = BATCHED_NORMALIZERS[normalization](torch.from_numpy(x).permute(0, 2, 1).contiguous())
    t = t.to(device)
    return np.stack([np.concatenate([m(t[i:i + 1024]).argmax(1).cpu().numpy()
                                     for i in range(0, len(t), 1024)]) for m in models])


def recall_and_top(pred: np.ndarray, true: np.ndarray, c: int) -> tuple[float, str, float]:
    """Recall of class c pooled over seeds, and its most frequent wrong prediction with share."""
    p = pred[:, true == c].ravel()
    wrong = np.bincount(p[p != c], minlength=N)
    top = int(wrong.argmax())
    return float((p == c).mean()), MODULATION_CLASSES[top], float(wrong[top] / p.size)


def verdict(synth: float, tx: float, real: float) -> str:
    gen, rx = synth - tx > GAP_PP, tx - real > GAP_PP
    return {(True, True): "both", (True, False): "GENERATOR",
            (False, True): "RECEIVER/chain", (False, False): "ok"}[(gen, rx)]


def main() -> None:
    cfg = Config.from_yaml(resolve_config_path("baseline_100"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(ROOT / "runs" / "baseline_100" / f"seed{s}" / "best.pt", cfg, device)
              for s in SEEDS]
    k_off = json.loads((TX_DIR / "manifest.json").read_text())["f_off_cycles"]
    rng = np.random.default_rng(NOISE_SEED)

    clean, noisy, labels = [], [], []
    for c, name in enumerate(MODULATION_CLASSES):
        f = tx_frames(name, k_off)
        clean.append(to_layout(f, cfg.data.normalization, np.float32))
        noisy.append(to_layout(add_awgn(f, SNR_DB, rng), cfg.data.normalization, np.float32))
        labels.append(np.full(len(f), c))
    labels = np.concatenate(labels)
    sets = {"tx_clean": score(models, np.concatenate(clean), cfg.data.normalization, device),
            f"tx_{SNR_DB:+d}": score(models, np.concatenate(noisy), cfg.data.normalization, device)}

    z = np.load(REAL)
    rows = [i for i, (c, s) in enumerate(zip(z["model_condition"], z["model_seed"]))
            if c == "baseline_100" and int(s) in SEEDS]
    top = z["snr"] == SNR_DB
    real_pred, real_true = z["pred"][rows][:, top], z["true"][top]

    own = [read_predictions(ROOT / "runs" / "baseline_100" / f"seed{s}" / "predictions.npz", s)
           for s in SEEDS]
    k = own[0].snr == SNR_DB
    synth_pred, synth_true = np.stack([r.pred[k] for r in own]), own[0].true[k]

    print(f"baseline_100 x {len(SEEDS)} seeds, pooled. {len(labels) // N} TX frames per class; "
          f"AWGN at {SNR_DB:+d} dB noise-bw. Recall % / top wrong prediction (share %).")
    print(f"verdict: a drop > {GAP_PP:.0f} pp synthetic -> TX{SNR_DB:+d} is GENERATOR, "
          f"TX{SNR_DB:+d} -> real is RECEIVER/chain.\n")
    hdr = ["synthetic +16", "TX clean", f"TX {SNR_DB:+d} dB", "real +16"]
    print(f"{'class':<10}" + "".join(f"{h:>27}" for h in hdr) + f"{'verdict':>17}")
    table = []
    for c, name in enumerate(MODULATION_CLASSES):
        cells = [recall_and_top(synth_pred, synth_true, c),
                 recall_and_top(sets["tx_clean"], labels, c),
                 recall_and_top(sets[f"tx_{SNR_DB:+d}"], labels, c),
                 recall_and_top(real_pred, real_true, c)]
        v = verdict(100 * cells[0][0], 100 * cells[2][0], 100 * cells[3][0])
        print(f"{name:<10}" + "".join(f"{100 * r:>7.1f} {t:>10} ({100 * s:>4.1f})"
                                      for r, t, s in cells) + f"{v:>17}")
        table.append({"class": name, "verdict": v, **{h: {"recall": r, "top": t, "share": s}
                                                       for h, (r, t, s) in zip(hdr, cells)}})
    out = ROOT / "outputs" / "tx_only_diagnostic.json"
    out.write_text(json.dumps(table, indent=1))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
