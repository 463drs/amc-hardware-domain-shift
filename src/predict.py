"""Generate per-frame predictions from trained checkpoints.

Produces predictions only; metric computation lives elsewhere so that metric
definitions can change without re-running any model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

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

# Off the cross-domain diagonal, `path` differing IS the cell; everything else must still hold,
# or the two files' test splits are not the same rows and the cell measures a split change too.
_CROSS_DOMAIN_KEYS = tuple(k for k in _VERIFIED_DATA_KEYS if k != "path")

# In-domain predictions keep the historical bare name, so every file already written stays put.
_IN_DOMAIN_PREDICTIONS = "predictions.npz"


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

def verify_split(cfg: Config, found: List[CellDir],
                 keys: Tuple[str, ...] = _VERIFIED_DATA_KEYS) -> None:
    """Refuse a checkpoint whose training data config differs from this one; `keys` narrows it
    for the cross-domain matrix, where `path` is the one key meant to differ."""
    current = {k: _normalize_data_value(k, getattr(cfg.data, k)) for k in keys}

    for c in found:
        stored = c.meta.get("config")
        trained = stored.get("data") if isinstance(stored, dict) else None
        if not isinstance(trained, dict):
            raise RuntimeError(
                f"{c.cell}: {c.path / 'meta.json'} carries no 'config.data' section, so the "
                f"training split cannot be verified against the evaluation config. Re-download "
                f"the run metadata (src.load_best_models) or remove the cell."
            )

        missing = [k for k in keys if k not in trained]
        if missing:
            raise RuntimeError(
                f"{c.cell}: stored data config is missing {missing}, so the training split "
                f"cannot be verified. Re-download the run metadata or remove the cell."
            )

        # Report every difference at once: fixing them one exception at a time is needless work.
        diffs = [
            (k, trained[k], getattr(cfg.data, k))
            for k in keys
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
    dest: Path, pred: np.ndarray, true: np.ndarray, snr: np.ndarray, meta: dict,
    filename: str = _IN_DOMAIN_PREDICTIONS, **extra: Any
) -> Path:
    """Write predictions next to the checkpoint, self-describing. `filename` puts a cross-domain
    scoring BESIDE the in-domain one rather than over it; `extra` records which split it was."""
    assert len(pred) == len(true) == len(snr), "prediction arrays are misaligned"
    out = dest / filename
    np.savez_compressed(
        out,
        pred=pred, true=true, snr=snr,
        run_id=meta["run_id"],
        dataset_hash=meta["config"].get("dataset_hash", ""),
        **extra,
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


def predictions_name(train_condition: str, eval_condition: str) -> str:
    """Filename a cell's predictions live under, keyed by the split they were scored on."""
    return (_IN_DOMAIN_PREDICTIONS if train_condition == eval_condition
            else f"predictions__on_{eval_condition}.npz")


def run_cross_domain(train_cfgs: Config | Sequence[Config], eval_cfg: Config,
                     root: Path = Path("runs"), download: bool = False,
                     overwrite: bool = False) -> List[Path]:
    """Score every checkpoint of each train config on `eval_cfg`'s test split -- inference only.
    Many train configs share one loader, and cells already on disk are reused, not rescored."""
    if isinstance(train_cfgs, Config):
        train_cfgs = [train_cfgs]
    eval_label = eval_cfg.experiment.condition

    todo: List[Tuple[Config, CellDir, Path]] = []
    written: List[Path] = []
    for train_cfg in train_cfgs:
        train_label = train_cfg.experiment.condition
        if download:
            download_checkpoints_by_config(train_cfg)
        found = discover_cells(root, train_label)
        verify_cells(expected_cells(train_cfg), found)
        keys = _VERIFIED_DATA_KEYS if train_label == eval_label else _CROSS_DOMAIN_KEYS
        verify_split(eval_cfg, found, keys=keys)

        name = predictions_name(train_label, eval_label)
        for c in found:
            out = c.path / name
            written.append(out)
            if overwrite or not out.exists():
                todo.append((train_cfg, c, out))

    if not todo:
        return written

    # Built only once something is missing: preload=true pulls the whole test split into RAM.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(eval_cfg, seed=0, verbose=False)
    for train_cfg, c, out in todo:
        model = load_model(c.path / "best.pt", train_cfg, device)
        pred, true, snr = predict(model, test_loader, device)
        save_predictions(c.path, pred, true, snr, c.meta, filename=out.name,
                         trained_on=train_cfg.experiment.condition, evaluated_on=eval_label)
    return written