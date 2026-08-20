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

from src.config import Config
from src.load_best_models import download_checkpoints_by_config
from src.data import build_dataloaders, MODULATION_CLASSES
from src.models import build_model

# Every DataConfig field whose change makes an offline number incomparable to the training run --
# either by moving WHICH frames land in the test split, or by changing their VALUES. `preload` is
# deliberately excluded (as in src.fingerprint): it selects HOW frames are read, all-into-RAM vs
# one-at-a-time, never which frames or what they contain.
_VERIFIED_DATA_KEYS = ("path", "subset_seed", "split_seed", "frames_per_pair",
                       "snr_min", "snr_max", "split", "normalization")


def _normalize_data_value(key: str, value: object) -> object:
    """Canonical form of one data-config value, so equal experiments compare equal.

    Each normalization removes a FALSE alarm; none of them can hide a real difference:
      * path      -> basename. config.py anchors it to an absolute repo-root path, so a run
                     trained on Kaggle and evaluated locally differs only in the prefix.
      * sequences -> tuple of floats. `split` is a tuple on the live Config and a JSON list once
                     it has been through W&B; str() of those two never matches.
      * numbers   -> float, so 30 and 30.0 do not read as a config change.
    """
    if key == "path":
        return str(value).replace("\\", "/").rsplit("/", 1)[-1]
    if isinstance(value, (list, tuple)):
        return tuple(float(x) for x in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)

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


def _cell_from_meta(meta: dict, meta_path: Path) -> Cell:
    """Run identity from the logged config, never from splitting run_name on '_'.

    A condition containing an underscore (rtl_sdr_gain0) makes name-splitting yield
    condition "rtl" and seed "sdr"; src.train logs both as top-level config keys instead.
    """
    cfg = meta.get("config")
    if not isinstance(cfg, dict) or "condition" not in cfg or "seed" not in cfg:
        raise RuntimeError(
            f"{meta_path}: logged config has no top-level 'condition'/'seed', so this run's "
            f"identity cannot be read. It predates src.train recording them; re-download the "
            f"metadata (src.load_best_models) or retrain the cell."
        )
    return Cell(condition=str(cfg["condition"]), seed=int(cfg["seed"]))


def discover_cells(root: Path, condition: str | None = None) -> List[CellDir]:
    """
    Scan the download root for cells, reading each meta.json.
    If condition is given - search only for the said condition.
    """
    found: List[CellDir] = []
    search_area = "*/*/meta.json" if condition is None else f"{condition}/*/meta.json"
    for meta_path in sorted(root.glob(search_area)):
        meta = json.loads(meta_path.read_text())
        found.append(
            CellDir(cell=_cell_from_meta(meta, meta_path), path=meta_path.parent, meta=meta)
        )
    if not found:
        raise FileNotFoundError(f"no cells under {root} matching {search_area!r}")
    return found
    
def verify_cells(expected: Set[Cell], found: List[CellDir]) -> None:
    """Fail loudly on a missing or extra cell OF THIS EXPERIMENT.

    Cells of another condition are ignored: runs/ holds every downloaded config side by side,
    and a neighbouring experiment's seed is not a defect in this one.
    """
    conditions = {c.condition for c in expected}
    got = {c.cell for c in found if c.cell.condition in conditions}
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
    """Refuse to evaluate a checkpoint whose training data config differs from this one."""
    current = {k: _normalize_data_value(k, getattr(cfg.data, k)) for k in _VERIFIED_DATA_KEYS}

    for c in found:
        stored = c.meta.get("config")
        trained = stored.get("data") if isinstance(stored, dict) else None
        if not isinstance(trained, dict):
            raise RuntimeError(
                f"{c.cell}: {c.path / 'meta.json'} carries no 'config.data' section, so the "
                f"training split cannot be verified against the evaluation config. Re-download "
                f"the run metadata (src.load_best_models) or remove the cell."
            )

        missing = [k for k in _VERIFIED_DATA_KEYS if k not in trained]
        if missing:
            raise RuntimeError(
                f"{c.cell}: stored data config is missing {missing}, so the training split "
                f"cannot be verified. Re-download the run metadata or remove the cell."
            )

        # Report every difference at once: fixing them one exception at a time is needless work.
        diffs = [
            (k, trained[k], getattr(cfg.data, k))
            for k in _VERIFIED_DATA_KEYS
            if _normalize_data_value(k, trained[k]) != current[k]
        ]
        if diffs:
            detail = "\n".join(f"  {k}: trained with {old!r}, evaluating with {new!r}"
                               for k, old, new in diffs)
            raise RuntimeError(
                f"{c.cell}: data config differs from the training run in {len(diffs)} key(s), so "
                f"the test split is not the one this model was held out from:\n{detail}"
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


def run_all(cfg: Config, root: Path = Path("runs"), condition : str | None = None) -> List[Path]:
    """Verify, then produce predictions for every cell.

    Discovery is scoped to this config's condition by default: the test loader below is built
    from THIS cfg, so scoring a neighbouring condition's checkpoints with it would be wrong.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_checkpoints_by_config(cfg)
    found = discover_cells(root, condition or cfg.experiment.condition)
    verify_cells(expected_cells(cfg), found)
    verify_split(cfg, found)

    _, _, test_loader = build_dataloaders(cfg, seed=0, verbose=False)

    written: List[Path] = []
    for c in found:
        model = load_model(c.path / "best.pt", cfg, device)
        pred, true, snr = predict(model, test_loader, device)
        written.append(save_predictions(c.path, pred, true, snr, c.meta))
    return written