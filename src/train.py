"""Core training module for the AMC domain-shift experiments: run one training job.

Structure, from lowest to highest altitude:
  * build_optimizer(model, cfg)      -- name+kwargs from TrainConfig -> torch optimizer
  * build_scheduler(optimizer, cfg)  -- name+kwargs from TrainConfig -> LR scheduler
  * train_one_epoch(...)             -- one pass over the train loader (optional AMP)
  * validate(...)                    -- one no-grad pass, returns loss/accuracy + per-SNR bins
  * fit(...)                         -- the training LOOP: epochs, scheduler stepping,
                                        best-metric tracking, early stopping, best-checkpoint.
                                        Filesystem-agnostic (checkpoint path is an argument),
                                        so it is unit-testable without config/logging/IO setup.
  * main(config_path, ...)           -- thin entry point: resolve config, configure logging,
                                        build everything, handle resume, call fit.

Concerns that change for their own reasons live in sibling modules:
  src.checkpointing  checkpoint I/O, RNG state, the resume-target contract
  src.fingerprint    config fingerprinting + drift diffing
  src.metrics        SNR bucketing (torch-free, shared with the future eval script)
  src.tracking       Weights & Biases

Everything that affects results comes from the config (optimizer, LR, weight decay, LR
schedule, epochs, early stopping, AMP, seeds); the compute device is auto-detected and never
configurable, so a config reproduces the same experiment on any machine.

Run:  python scripts/train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import dataclasses
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.checkpointing import _load_checkpoint, _resolve_resume_target, _save_checkpoint
from src.config import (
    Config,
    TrainConfig,
    get_device,
    resolve_config_path,
    set_cudnn_determinism,
    set_seed,
)
from src.data import MODULATION_CLASSES, build_dataloaders
from src.fingerprint import _config_fingerprint, _fingerprint_diff, _format_fingerprint_diff
from src.logging_utils import configure_logging, get_logger
from src.metrics import snr_bucket
from src.models import build_model
from src.tracking import _deterministic_run_id, _init_wandb

logger = get_logger(__name__)

# Runtime artifacts (logs, checkpoints) land here, under a per-run subdirectory. Ignored by
# git (.gitignore: outputs/, *.pt).
_REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = _REPO_ROOT / "outputs"

# Named registries: the config selects by name, the concrete class is resolved here (kept out
# of config.py so config loading never imports torch.optim). New choices are one line each.
OPTIMIZERS: Dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

SCHEDULERS: Dict[str, Callable[..., object]] = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "step": torch.optim.lr_scheduler.StepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}

# early_stopping_metric -> (key in the validate() dict, optimization direction).
_VAL_METRIC_KEY = {"val_loss": "loss", "val_accuracy": "accuracy"}
_METRIC_MODE = {"val_loss": "min", "val_accuracy": "max"}


def _progress_disabled(progress: Optional[bool]) -> bool:
    """Resolve the `progress` argument into tqdm's `disable` flag.

    None auto-detects: bars are shown only when stderr is a TTY, so redirected output (nohup on
    a rented box, Kaggle cell logs) does not get thousands of carriage-return lines interleaved
    with the logging output. True/False override explicitly. Deliberately NOT a config field --
    it changes console output only, never results.
    """
    if progress is not None:
        return not progress
    stream = sys.stderr
    return not (hasattr(stream, "isatty") and stream.isatty())


# Builders

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Construct the optimizer named by the config.

    learning_rate and weight_decay are first-class TrainConfig fields (the two most-swept
    knobs) and are passed explicitly; cfg.optimizer.kwargs carries any extras (betas,
    momentum, ...). An unknown name fails loudly here, the same way data._get_normalizer
    rejects an unknown normalization.
    """
    name = cfg.optimizer.name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {cfg.optimizer.name!r}. Available: {sorted(OPTIMIZERS)}.")
    return OPTIMIZERS[name](
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        **cfg.optimizer.kwargs,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    """Construct the LR scheduler named by the config, forwarding its kwargs.

    For ReduceLROnPlateau the `mode` (min/max) is DERIVED from early_stopping_metric rather
    than trusted from kwargs: an omitted or stale `mode` would otherwise silently let the
    scheduler optimize in the wrong direction (PyTorch defaults it to "min"), exactly the kind
    of silent misconfiguration this config system fails loudly on elsewhere. If kwargs carries
    a conflicting `mode` we override it and log a warning so the discrepancy leaves a trace.
    """
    name = cfg.lr_scheduler.name.lower()
    if name not in SCHEDULERS:
        raise ValueError(
            f"Unknown lr_scheduler {cfg.lr_scheduler.name!r}. Available: {sorted(SCHEDULERS)}."
        )
    scheduler_cls = SCHEDULERS[name]
    kwargs = dict(cfg.lr_scheduler.kwargs)  # copy: never mutate the validated config

    if scheduler_cls is torch.optim.lr_scheduler.ReduceLROnPlateau:
        required_mode = _METRIC_MODE[cfg.early_stopping_metric]
        given_mode = kwargs.get("mode")
        if given_mode is not None and given_mode != required_mode:
            logger.warning(
                "lr_scheduler.kwargs['mode']=%r disagrees with early_stopping_metric=%r "
                "(which implies mode=%r); overriding to %r.",
                given_mode, cfg.early_stopping_metric, required_mode, required_mode,
            )
        kwargs["mode"] = required_mode

    return scheduler_cls(optimizer, **kwargs)


# Train / validate one epoch

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    progress: Optional[bool] = None,
) -> Dict[str, float]:
    """Run one training pass; return mean loss and accuracy over the epoch.

    Mixed precision is used only when `scaler` is enabled (see main); on CPU or with AMP
    disabled the autocast context is a no-op, so this one code path covers both.

    `progress` controls the tqdm bar (None = auto per _progress_disabled). The bar is an
    in-the-moment display only; the per-epoch summary is logged by fit(), not here.
    """
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    running_loss, correct, total = 0.0, 0, 0

    for iq, labels, _snr in tqdm(loader, desc="train", leave=False,
                                 disable=_progress_disabled(progress)):
        iq = iq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(iq)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch = labels.size(0)
        running_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch

    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    progress: Optional[bool] = None,
) -> Dict[str, object]:
    """Run one no-grad pass; return aggregate loss/accuracy plus a per-SNR-bin breakdown.

    RadioMLDataset already yields SNR per item, so bucketing accuracy by 2 dB bins here is free
    and avoids a separate post-hoc pass. Returns:
      loss, accuracy         -- aggregate over the whole val set
      snr_accuracy           -- {snr_bin_dB: accuracy}, 2 dB bins (the accuracy-vs-SNR curve)
      accuracy_snr_geq_0db   -- aggregate accuracy over snr >= 0 dB (the headline metric)

    `progress` controls the tqdm bar (None = auto per _progress_disabled).
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    per_bin_correct: Dict[int, int] = defaultdict(int)
    per_bin_total: Dict[int, int] = defaultdict(int)
    correct_geq0, total_geq0 = 0, 0

    for iq, labels, snr in tqdm(loader, desc="val", leave=False,
                                disable=_progress_disabled(progress)):
        iq = iq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(iq)
        loss = criterion(logits, labels)

        batch = labels.size(0)
        is_correct = logits.argmax(dim=1) == labels
        running_loss += loss.item() * batch
        correct += int(is_correct.sum().item())
        total += batch

        # Bucket this batch's correctness by SNR. Loop over the (<=26) distinct SNR values, not
        # per item; snr/correctness are small, so move them to CPU once.
        snr_cpu = snr.to("cpu").view(-1)
        correct_cpu = is_correct.to("cpu").view(-1)
        for s in torch.unique(snr_cpu).tolist():
            mask = snr_cpu == s
            b = snr_bucket(s)
            per_bin_total[b] += int(mask.sum().item())
            per_bin_correct[b] += int(correct_cpu[mask].sum().item())
        geq0 = snr_cpu >= 0
        total_geq0 += int(geq0.sum().item())
        correct_geq0 += int(correct_cpu[geq0].sum().item())

    snr_accuracy = {b: per_bin_correct[b] / per_bin_total[b] for b in sorted(per_bin_total)}
    acc_geq0 = correct_geq0 / total_geq0 if total_geq0 > 0 else float("nan")
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "snr_accuracy": snr_accuracy,
        "accuracy_snr_geq_0db": acc_geq0,
    }


def _is_improvement(current: float, best: float, mode: str) -> bool:
    return current < best if mode == "min" else current > best


# Training loop

def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_epochs: int,
    early_stopping_patience: int,
    early_stopping_metric: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    start_epoch: int = 1,
    best_metric: Optional[float] = None,
    epochs_no_improve: int = 0,
    checkpoint_path: Optional[Path] = None,
    last_checkpoint_path: Optional[Path] = None,
    checkpoint_metadata: Optional[dict] = None,
    run_id: str = "",
    on_epoch_end: Optional[Callable[[int, Dict[str, object]], None]] = None,
    progress: Optional[bool] = None,
) -> dict:
    """Run the multi-epoch training loop and return a summary.

    Each epoch: train_one_epoch -> validate -> step the scheduler -> track the monitored
    metric -> checkpoint on improvement -> early-stop after `early_stopping_patience` epochs
    without improvement (`max_epochs` is the hard ceiling regardless).

    Kept free of config/logging/tracking/resume concerns so it can be unit-tested directly:
    pass a tiny model and fake loaders, `checkpoint_path=None` to skip disk writes, and
    `on_epoch_end=None` to skip metric reporting. `on_epoch_end(epoch, metrics)` is where
    main() hooks W&B logging. `start_epoch`/`best_metric`/`epochs_no_improve` carry state from
    a resume checkpoint -- `epochs_no_improve` defaults to 0 so a fresh run is unaffected, and a
    resumed run continues its patience count instead of getting a fresh budget.

    Two checkpoints: `checkpoint_path` (best.pt) is written only on improvement, while
    `last_checkpoint_path` (last.pt) is written at the end of EVERY epoch so a session death
    mid-patience-window resumes from the latest epoch instead of replaying trained epochs.

    `checkpoint_metadata` is opaque here: fit() forwards it verbatim into every checkpoint and
    never inspects it. main() passes the config fingerprint, which keeps Config out of this
    function and preserves its unit-testability (tiny model + fake loaders, no config, no IO).
    """
    metric_key = _VAL_METRIC_KEY[early_stopping_metric]
    mode = _METRIC_MODE[early_stopping_metric]
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    if best_metric is None:
        best_metric = math.inf if mode == "min" else -math.inf
    best_epoch = start_epoch - 1
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, max_epochs + 1):
        last_epoch = epoch
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler,
                             progress=progress)
        va = validate(model, val_loader, criterion, device, progress=progress)
        monitored = va[metric_key]

        # ReduceLROnPlateau needs the monitored metric; step-based schedulers do not.
        if plateau:
            scheduler.step(monitored)
        else:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "epoch %3d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f (>=0dB %.4f) | lr %.2e",
            epoch, max_epochs, tr["loss"], tr["accuracy"],
            va["loss"], va["accuracy"], va["accuracy_snr_geq_0db"], lr,
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, {
                "train/loss": tr["loss"], "train/accuracy": tr["accuracy"],
                "val/loss": va["loss"], "val/accuracy": va["accuracy"],
                "val/accuracy_snr_geq_0db": va["accuracy_snr_geq_0db"],
                "val/snr_accuracy": va["snr_accuracy"],   # {bin: acc}; main formats for W&B
                "lr": lr,
            })

        improved = _is_improvement(monitored, best_metric, mode)
        if improved:
            best_metric = monitored
            best_epoch = epoch
            epochs_no_improve = 0
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
                                 scheduler=scheduler, scaler=scaler, epoch=epoch,
                                 best_metric=best_metric, epochs_no_improve=epochs_no_improve,
                                 run_id=run_id, config_metadata=checkpoint_metadata)
                logger.info("  new best %s=%.4f -> saved %s", early_stopping_metric, best_metric, checkpoint_path)
            else:
                logger.info("  new best %s=%.4f", early_stopping_metric, best_metric)
        else:
            epochs_no_improve += 1

        # Periodic checkpoint: written EVERY epoch regardless of improvement, after the counters
        # above are updated, so a session death mid-patience-window resumes from here rather
        # than replaying every epoch since the last improvement. Saved before the early-stop
        # break so the final epoch is captured too.
        if last_checkpoint_path is not None:
            _save_checkpoint(last_checkpoint_path, model=model, optimizer=optimizer,
                             scheduler=scheduler, scaler=scaler, epoch=epoch,
                             best_metric=best_metric, epochs_no_improve=epochs_no_improve,
                             run_id=run_id, config_metadata=checkpoint_metadata)

        # Only a non-improving epoch can early-stop; an improvement just reset the counter.
        if not improved and epochs_no_improve >= early_stopping_patience:
            logger.info("early stopping at epoch %d (no %s improvement for %d epochs)",
                        epoch, early_stopping_metric, early_stopping_patience)
            break

    return {
        "best_metric_name": early_stopping_metric,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_run": last_epoch,
    }


# main() helpers

def _load_resume_state(
    resume_path: Optional[Path], *, model, optimizer, scheduler, scaler, device,
    current_fingerprint: dict, allow_config_change: bool, early_stopping_metric: str,
):
    """Load a resume checkpoint and verify its config against the current one.

    Returns (start_epoch, best_metric, epochs_no_improve, config_drift_allowed). With
    resume_path None this is the fresh-run identity: (1, None, 0, False).

    NOTE the split of responsibilities with _resolve_resume_target: resolution must happen in
    main() *before* set_seed, because whether to seed at all depends on it and seeding has to
    precede build_model (model init consumes the RNG). Loading can only happen *after* the
    modules exist, so the two necessarily sit on opposite sides of construction.
    """
    if resume_path is None:
        return 1, None, 0, False

    # Restores model/optimizer/scheduler/scaler AND the RNG state (last), so training
    # continues the same stream. set_seed was deliberately skipped by the caller.
    state = _load_checkpoint(
        resume_path, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, device=device,
    )
    start_epoch = state["epoch"] + 1
    best_metric = state["best_metric"]
    resumed_no_improve = state["epochs_no_improve"]
    logger.info("resumed from %s at epoch %d (best %s=%.4f, epochs_no_improve=%d)",
                resume_path, state["epoch"], early_stopping_metric,
                best_metric, resumed_no_improve)

    # Config drift: resuming under a changed config yields a trajectory matching no YAML on
    # disk, and (with wandb resume="allow") a W&B record showing the OLD config.
    config_drift_allowed = False
    stored = state["config"]
    if stored is None:
        logger.warning(
            "checkpoint %s predates config fingerprinting; cannot verify config drift.",
            resume_path,
        )
    else:
        diff = _fingerprint_diff(stored, current_fingerprint)
        if diff:
            detail = (f"config changed since {resume_path} was written "
                      f"({len(diff)} key(s) differ):\n{_format_fingerprint_diff(diff)}")
            if allow_config_change:
                config_drift_allowed = True
                logger.warning("%s\ncontinuing anyway (allow_config_change); W&B config "
                               "will be updated for: %s",
                               detail, ", ".join(key for key, _, _ in diff))
            else:
                raise RuntimeError(
                    f"{detail}\nResuming would train under a config matching no run on "
                    f"record. Pass --allow-config-change to override, --fresh to retrain "
                    f"from scratch, or --run-id NAME to start a separate run."
                )

    return start_epoch, best_metric, resumed_no_improve, config_drift_allowed


def _make_wandb_epoch_callback(run):
    """Build fit()'s on_epoch_end callback, logging each epoch's metrics to a W&B run."""

    def _on_epoch_end(epoch: int, metrics: Dict[str, object]) -> None:
        if run is None:
            return
        # Split the per-SNR dict out of the flat scalars; log each SNR bin as its own scalar
        # series (val_snr/<bin>dB) so W&B renders per-SNR accuracy curves over epochs, next to
        # the aggregate train/val scalars.
        snr_accuracy = metrics.get("val/snr_accuracy") or {}
        payload = {k: v for k, v in metrics.items() if k != "val/snr_accuracy"}
        for bin_db, acc in snr_accuracy.items():
            payload[f"val_snr/{bin_db}dB"] = acc
        run.log(payload, step=epoch)

    return _on_epoch_end


# Entry point

def main(
    config_path: str,
    resume: "str | bool | None" = None,
    fresh: bool = False,
    run_id: Optional[str] = None,
    allow_config_change: bool = False,
) -> dict:
    """Train a model end-to-end from a config; return the fit() summary plus run metadata.

    Parameters
    ----------
    config_path : a config name ("baseline"), filename, or path (see resolve_config_path).
    resume      : None = start fresh, but REFUSE if the run dir already holds a checkpoint;
                  True = continue from the newer of last.pt/best.pt; a path = that checkpoint.
    fresh       : discard any existing checkpoints and train from epoch 1.
    run_id      : optional override of the run NAME; defaults to "<condition>_<seed>".
    allow_config_change : downgrade the config-drift check from an error to a warning.
    """
    config = Config.from_yaml(resolve_config_path(config_path))
    tcfg = config.train
    exp = config.experiment

    # Run identity is a deterministic function of condition + training seed (NOT the config
    # filename, NOT a timestamp), so a restarted session reuses the same W&B run and the same
    # local checkpoint directory. `run_id` (CLI) overrides only the human-readable name.
    run_name = run_id or f"{exp.condition}_{tcfg.seed}"
    wandb_id = _deterministic_run_id(run_name)
    run_dir = OUTPUTS_DIR / run_name
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    configure_logging(log_file=run_dir / "train.log")
    logger.info("run=%s  wandb_id=%s  config=%s", run_name, wandb_id, config.source_path)

    resume_path = _resolve_resume_target(resume, fresh, run_dir, last_ckpt, best_ckpt)

    # cuDNN determinism applies to BOTH paths. A resumed run skips set_seed (it restores the
    # checkpointed RNG stream instead), so these flags must be pinned separately or the resumed
    # process silently falls back to PyTorch's non-deterministic defaults.
    set_cudnn_determinism()
    # Fresh run: seed all RNGs from the config. Resumed run: do NOT reseed -- the checkpoint's
    # RNG state is restored below (after the modules load) so the stream continues from where
    # the interrupted run left off, instead of snapping back to the epoch-1 initial state.
    if resume_path is None:
        set_seed(tcfg.seed)
    device = get_device()
    logger.info("device=%s  amp_enabled=%s", device, tcfg.amp_enabled)

    train_loader, val_loader, _test_loader = build_dataloaders(config)

    model = build_model(config.model, n_classes=len(MODULATION_CLASSES)).to(device)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg)
    criterion = nn.CrossEntropyLoss()

    amp_active = tcfg.amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active)

    fingerprint = _config_fingerprint(config)
    start_epoch, best_metric, resumed_no_improve, config_drift_allowed = _load_resume_state(
        resume_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        device=device, current_fingerprint=fingerprint,
        allow_config_change=allow_config_change,
        early_stopping_metric=tcfg.early_stopping_metric,
    )

    run = _init_wandb(exp, run_name, wandb_id, dataclasses.asdict(config),
                      allow_val_change=config_drift_allowed)

    try:
        summary = fit(
            model, train_loader, val_loader, optimizer, scheduler, criterion, device,
            max_epochs=tcfg.max_epochs,
            early_stopping_patience=tcfg.early_stopping_patience,
            early_stopping_metric=tcfg.early_stopping_metric,
            scaler=scaler,
            start_epoch=start_epoch,
            best_metric=best_metric,
            epochs_no_improve=resumed_no_improve,
            checkpoint_path=best_ckpt,
            last_checkpoint_path=last_ckpt,
            checkpoint_metadata=fingerprint,
            run_id=run_name,
            on_epoch_end=_make_wandb_epoch_callback(run),
        )
    finally:
        # Always close the W&B run, even if training raises, so the run isn't left "running".
        if run is not None:
            run.finish()

    summary["run_name"] = run_name
    summary["wandb_id"] = wandb_id
    summary["best_checkpoint"] = str(best_ckpt)
    summary["last_checkpoint"] = str(last_ckpt)
    logger.info("done. best %s=%.4f at epoch %d (ran %d epoch(s))",
                summary["best_metric_name"], summary["best_metric"],
                summary["best_epoch"], summary["epochs_run"])
    return summary
