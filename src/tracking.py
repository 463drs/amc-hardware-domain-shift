"""Experiment tracking (Weights & Biases).

Separated from the training loop because tracking changes for entirely different reasons than
the loop does -- and because keeping the wandb import in one lazily-evaluated place is what lets
the test suite and `experiment.mode: disabled` runs work with wandb not installed at all.
"""

from __future__ import annotations

import hashlib
import torch

from src.logging_utils import get_logger

logger = get_logger(__name__)

def _environment_metadata() -> dict:
    """Runtime environment actually used, for the W&B record (not the fingerprint)."""
    meta = {
        "env/torch": torch.__version__,
        "env/cuda": torch.version.cuda,
        "env/gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return meta

def _deterministic_run_id(run_name: str) -> str:
    """Stable W&B run id derived only from the run name (condition + seed).

    Deterministic on purpose: a restarted Kaggle session recomputes the SAME id, so
    wandb.init(id=..., resume="allow") continues the existing run/curve instead of spawning a
    fragmented duplicate. A timestamp here would defeat exactly that, so there is none.
    """
    return hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:16]


def _init_wandb(experiment, run_name: str, wandb_id: str, config_dict: dict,
                allow_val_change: bool = False):
    """Initialize W&B tracking, or return None when disabled.

    wandb is imported lazily so the rest of the package (and the test suite) never require the
    package; with experiment.mode == "disabled" it is skipped entirely -- no import, no
    account, no network. The returned run (or None) is what main() logs metrics through.

    `allow_val_change` is set only when resuming with a deliberately-changed config: with
    resume="allow", wandb otherwise keeps the ORIGINAL config values and silently ignores the
    new ones, leaving the record showing a config that is not what is being trained.
    """
    if experiment.mode == "disabled":
        logger.info("W&B disabled (experiment.mode=disabled); not tracking this run.")
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            f"experiment.mode={experiment.mode!r} needs the 'wandb' package; "
            f"install it (pip install wandb) or set experiment.mode: disabled."
        ) from exc
    config_dict = {**config_dict, **_environment_metadata()}
    run = wandb.init(
        project=experiment.project,
        group=experiment.condition,
        name=run_name,
        id=wandb_id,
        resume="allow",
        mode=experiment.mode,
        config=config_dict,
        allow_val_change=allow_val_change,
    )
    logger.info("W&B init: project=%s group=%s name=%s id=%s mode=%s",
                experiment.project, experiment.condition, run_name, wandb_id, experiment.mode)
    return run
