"""Generate per-frame predictions from trained checkpoints.

Produces predictions only; metric computation lives elsewhere so that metric
definitions can change without re-running any model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config, resolve_config_path
from src.data import build_dataloaders, MODULATION_CLASSES
from src.models import build_model

_SPLIT_KEYS = ("path", "subset_seed", "split_seed", "frames_per_pair",
               "snr_min", "snr_max", "split")

@dataclass(frozen=True, order=True)
class Cell:
    """One matrix cell: a training condition and a training seed."""
    condition: str
    seed: int


@dataclass(frozen=True)
class CellDir:
    """A downloaded cell on disk."""
    cell: Cell
    path: Path
    meta: dict


def expected_cells(cfg: Config) -> Set[Cell]:
    """Cells the config prescribes."""
    return {
        Cell(condition=cfg.experiment.condition, seed=s)
        for s in cfg.train.seeds
    }


def discover_cells(root: Path, condition: str | None = None) -> List[CellDir]:
    """
    Scan the download root for cells, reading each meta.json.
    If condition is given - search only for the said condition.
    """
    found: List[CellDir] = []
    search_area = "*/*/meta.json" if condition is None else f"{condition}/*/meta.json" 
    for meta_path in sorted(root.glob(search_area)):
        meta = json.loads(meta_path.read_text())
        name_split = meta["run_name"].split("_")

        found.append(
            CellDir(
                cell=Cell(condition=name_split[0], seed=int(name_split[1])),
                path=meta_path.parent,
                meta=meta,
            )
        )
    if not found:
        raise FileNotFoundError(f"no cells under {root} matching {search_area!r}")
    return found
    
def verify_cells(expected: Set[Cell], found: List[CellDir]) -> None:
    """Fail loudly on a missing or extra cell."""
    got = {c.cell for c in found}
    missing, extra = expected - got, got - expected
    if missing or extra:
        raise RuntimeError(f"cell mismatch. missing={sorted(missing)} extra={sorted(extra)}")

def load_model(path: Path, cfg: Config, device: torch.device) -> torch.nn.Module:
    """Load a checkpoint into a freshly built model, in eval mode.

    weights_only=False because our checkpoints carry RNG state objects alongside the
    tensors (see src.checkpointing) -- these are trusted, self-produced files.
    """
    model = build_model(cfg.model, len(MODULATION_CLASSES))
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the test set once. Returns (pred, true, snr), aligned and ordered."""
    pred, true, snr = [], [], []
    for iq, y, z in loader:
        logits = model(iq.to(device, non_blocking=True))
        pred.append(logits.argmax(dim=1).cpu().numpy())
        true.append(y.numpy())
        snr.append(z.numpy())
    return (
        np.concatenate(pred).astype(np.int16),
        np.concatenate(true).astype(np.int16),
        np.concatenate(snr).astype(np.int16),
    )

def verify_split(cfg: Config, found: List[CellDir]) -> None:
    """Evaluating on a split that differs from the training one silently inflates accuracy."""
    current = {k: getattr(cfg.data, k) for k in _SPLIT_KEYS}
    for c in found:
        train_cfg = c.meta["config"]
        for k, v in current.items():
            got = train_cfg.get(k)
            if got is not None and str(got) != str(v):
                raise RuntimeError(
                    f"{c.cell}: split parameter {k!r} differs: trained with {got!r}, "
                    f"evaluating with {v!r}"
                )

def save_predictions(
    dest: Path, pred: np.ndarray, true: np.ndarray, snr: np.ndarray, meta: dict
) -> Path:
    """Write predictions next to the checkpoint, self-describing."""
    assert len(pred) == len(true) == len(snr), "prediction arrays are misaligned"
    out = dest / "predictions.npz"
    np.savez_compressed(
        out,
        pred=pred, true=true, snr=snr,
        run_id=meta["run_id"],
        dataset_hash=meta["config"].get("dataset_hash", ""),
    )
    return out


def run_all(cfg: Config, root: Path = Path("runs")) -> List[Path]:
    """Verify, then produce predictions for every cell."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    found = discover_cells(root)
    verify_cells(expected_cells(cfg), found)
    verify_split(cfg, found)

    _, _, test_loader = build_dataloaders(cfg, seed=0, verbose=False)

    written: List[Path] = []
    for c in found:
        model = load_model(c.path / "best.pt", cfg, device)
        pred, true, snr = predict(model, test_loader, device)
        written.append(save_predictions(c.path, pred, true, snr, c.meta))
    return written