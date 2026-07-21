"""Checkpoint I/O, RNG state capture, and the resume-target contract.

Split from the training loop because it changes for its own reasons -- what a checkpoint must
contain to make a resumed run indistinguishable from an uninterrupted one, and how a user
expresses the intent to resume. Imports nothing from src.train (and must not, to stay acyclic):
the loop calls in here, never the reverse.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.logging_utils import get_logger

logger = get_logger(__name__)


# RNG state (enough to reproduce a resumed run's stream, not just its weights)

def _capture_rng_states() -> dict:
    """Snapshot every global RNG so a resumed run continues the same stream, not a fresh one."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_states(rng_states: dict) -> None:
    """Inverse of _capture_rng_states.

    .cpu() guards against torch.load(map_location=cuda) having moved the (CPU) generator-state
    tensors onto the GPU, which torch.set_rng_state would then reject.
    """
    torch.set_rng_state(rng_states["torch"].cpu())
    cuda_states = rng_states.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    np.random.set_state(rng_states["numpy"])
    random.setstate(rng_states["python"])


# Save / load

def _save_checkpoint(
    path: Path, *, model, optimizer, scheduler, scaler, epoch, best_metric, epochs_no_improve,
    run_id, config_metadata: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "epochs_no_improve": epochs_no_improve,   # so a resume continues the patience count
        "run_id": run_id,
        "rng_states": _capture_rng_states(),       # so a resume continues the RNG stream
    }
    # Makes the checkpoint self-describing: a resume can verify the config still matches.
    # Omitted entirely when absent, so "no config key" cleanly means "legacy checkpoint".
    if config_metadata is not None:
        payload["config"] = config_metadata
    # GradScaler state (current scale + growth tracker) matters only when AMP is active.
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def _load_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, device, restore_rng=True):
    """Restore model/optimizer/scheduler (+ scaler + RNG when present) from a checkpoint.

    Returns the loop bookkeeping needed to continue: epoch, best_metric, epochs_no_improve, plus
    the stored `config` fingerprint (None on checkpoints written before fingerprinting existed,
    which main() reports as "drift cannot be verified").

    RNG is restored last, after the module states are loaded, so the resumed stream matches an
    uninterrupted run. weights_only=False because the payload carries our own RNG state objects
    (numpy/python tuples), not just tensors -- these are trusted, self-produced checkpoints.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and "rng_states" in ckpt:
        _restore_rng_states(ckpt["rng_states"])
    return {
        "epoch": int(ckpt["epoch"]),
        "best_metric": float(ckpt["best_metric"]),
        "epochs_no_improve": int(ckpt.get("epochs_no_improve", 0)),
        "config": ckpt.get("config"),
    }


# Resume-target selection

def _checkpoint_epoch(path: Path) -> Optional[int]:
    """Epoch stored in a checkpoint, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
    except Exception:  # corrupt/partial file (e.g. killed mid-write) must not block a resume
        logger.warning("could not read epoch from %s; ignoring it when selecting a resume point.", path)
        return None


def select_resume_checkpoint(*paths: Path) -> Optional[Path]:
    """Pick the checkpoint holding the most training progress (highest stored epoch).

    Resuming from best.pt alone would replay every epoch trained since the last improvement --
    exactly the wasted GPU time the periodic last.pt exists to avoid. Returns None when no
    candidate exists, i.e. this is a fresh run.
    """
    chosen, chosen_epoch = None, None
    for path in paths:
        epoch = _checkpoint_epoch(path)
        if epoch is not None and (chosen_epoch is None or epoch > chosen_epoch):
            chosen, chosen_epoch = path, epoch
    return chosen


def _resolve_resume_target(
    resume: "str | bool | None", fresh: bool, run_dir: Path, last_ckpt: Path, best_ckpt: Path
) -> Optional[Path]:
    """Resolve the three-state resume contract into a checkpoint path (None == start fresh).

    Intent is expressed per invocation and the default is to refuse rather than guess: the
    launch command normally lives in a notebook cell, so a persistent flag would silently
    become permanent, and resuming-by-accident produces a trajectory nobody asked for.

      resume=None,  no checkpoints -> fresh (the ordinary case)
      resume=None,  checkpoints    -> raise, naming the run dir, the epoch and both escapes
      resume=True                  -> newer of last.pt/best.pt; raise if there is none
      resume=<path>                -> that exact checkpoint
      fresh=True                   -> discard whatever exists and start at epoch 1
    """
    existing = select_resume_checkpoint(last_ckpt, best_ckpt)

    if fresh:
        if existing is not None:
            logger.warning("--fresh: ignoring existing checkpoint %s; training from epoch 1.",
                           existing)
        return None

    if resume is None:
        if existing is not None:
            raise RuntimeError(
                f"{run_dir} already contains a checkpoint at epoch {_checkpoint_epoch(existing)}; "
                f"pass --resume to continue from it, --fresh to discard and retrain, "
                f"or --run-id NAME to start a separate run."
            )
        return None

    if resume is True:   # bare --resume: auto-select
        if existing is None:
            raise RuntimeError(
                f"--resume was requested but {run_dir} contains no usable checkpoint "
                f"(looked for {last_ckpt.name} and {best_ckpt.name}). An explicit resume that "
                f"finds nothing is a mistake, not a fresh start -- use --fresh to train anew."
            )
        return existing

    path = Path(resume)  # --resume <path>
    if not path.is_file():
        raise FileNotFoundError(f"--resume {str(resume)!r}: no such checkpoint file.")
    return path
