"""Tests for the checkpoint/resume reproducibility fixes and per-SNR validation.

These exercise fit()/validate()/build_scheduler()/_save_checkpoint()/_load_checkpoint()
directly with tiny models and fake loaders -- no config file, no W&B, no real data -- which is
exactly the property fit() is designed to preserve.

Covers src.train plus its extracted siblings (src.checkpointing, src.fingerprint, src.metrics)
and the scripts/train.py CLI wrapper.
"""

import io
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import scripts.train as train_cli
import src.train as train
from src.checkpointing import _load_checkpoint, _save_checkpoint, select_resume_checkpoint
from src.config import (
    Config,
    TrainConfig,
    resolve_config_path,
    set_cudnn_determinism,
    set_seed,
)
from src.fingerprint import _config_fingerprint
from src.metrics import snr_bucket

REPO_ROOT = Path(__file__).resolve().parents[1]

CPU = torch.device("cpu")


def _tiny_loader(n=8, n_classes=3, snr_value=0):
    xb = torch.randn(n, 4)
    yb = torch.randint(0, n_classes, (n,))
    snr = torch.full((n,), snr_value, dtype=torch.long)
    return [(xb, yb, snr)]


def _tiny_setup(lr=0.0):
    """A frozen-by-default linear model plus optimizer/scheduler, for loop-level tests."""
    model = nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    return model, opt, sched


# Issue 1: RNG state is checkpointed and restored on a save/load round-trip.

def test_checkpoint_roundtrip_restores_rng(tmp_path):
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    model = nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    # Advance every RNG so we are not comparing the trivial just-seeded state.
    _ = torch.randn(7); _ = np.random.rand(7); _ = random.random()

    ckpt = tmp_path / "ckpt.pt"
    _save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sched, scaler=None,
                           epoch=4, best_metric=0.25, epochs_no_improve=2, run_id="r")

    # _save_checkpoint does not advance any RNG, so the persisted state == the state right now.
    exp_torch = torch.get_rng_state().clone()
    exp_np = np.random.get_state()
    exp_py = random.getstate()
    ref_next = torch.randn(3)  # what an uninterrupted run would draw next

    # Perturb every RNG, then confirm a resume restores them exactly.
    torch.manual_seed(999); np.random.seed(999); random.seed(999)
    assert not torch.equal(torch.get_rng_state(), exp_torch)

    state = _load_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sched,
                                   scaler=None, device=CPU)

    assert torch.equal(torch.get_rng_state(), exp_torch)
    np_now = np.random.get_state()
    assert np_now[0] == exp_np[0]
    assert np.array_equal(np_now[1], exp_np[1])
    assert np_now[2] == exp_np[2]
    assert random.getstate() == exp_py
    # Behavioural check: the next draw matches the uninterrupted continuation.
    assert torch.equal(torch.randn(3), ref_next)
    # Bookkeeping is round-tripped too ("config" is None: no metadata was passed in).
    assert state == {"epoch": 4, "best_metric": 0.25, "epochs_no_improve": 2, "config": None}


# Issue 2: a run resumed mid-patience-window continues the count instead of resetting.

def _run_until_early_stop(epochs_no_improve_start, patience=3, max_epochs=10):
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    loader = _tiny_loader()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # frozen weights: val never changes
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    summary = train.fit(
        model, loader, loader, opt, sched, nn.CrossEntropyLoss(), CPU,
        max_epochs=max_epochs, early_stopping_patience=patience,
        early_stopping_metric="val_loss",
        best_metric=-1.0,   # a positive val loss can never beat this, so it never "improves"
        epochs_no_improve=epochs_no_improve_start,
    )
    return summary["epochs_run"]


def test_resume_continues_patience_count():
    # Fresh run (counter 0): needs `patience` non-improving epochs -> stops at epoch 3.
    assert _run_until_early_stop(0, patience=3) == 3
    # Resumed mid-window (counter 2): only 1 more non-improving epoch -> stops at epoch 1.
    assert _run_until_early_stop(2, patience=3) == 1


def test_fit_default_patience_is_zero():
    # Not passing epochs_no_improve preserves the pre-fix behaviour (fresh budget).
    assert _run_until_early_stop(0, patience=2) == 2


# Periodic last.pt: written every epoch, unlike best.pt which only lands on improvement.

def test_last_checkpoint_written_every_epoch_without_improvement(tmp_path):
    torch.manual_seed(0)
    model, opt, sched = _tiny_setup()
    best_ckpt, last_ckpt = tmp_path / "best.pt", tmp_path / "last.pt"

    summary = train.fit(
        model, _tiny_loader(), _tiny_loader(), opt, sched, nn.CrossEntropyLoss(), CPU,
        max_epochs=3, early_stopping_patience=99, early_stopping_metric="val_loss",
        best_metric=-1.0,   # never improves, so best.pt is never written
        checkpoint_path=best_ckpt, last_checkpoint_path=last_ckpt, progress=False,
    )

    assert summary["epochs_run"] == 3
    assert not best_ckpt.exists(), "best.pt must only be written on improvement"
    assert last_ckpt.exists(), "last.pt must be written even with no improvement"
    # It holds the LAST epoch, and the patience count accumulated across all 3 epochs.
    state = torch.load(last_ckpt, weights_only=False)
    assert state["epoch"] == 3
    assert state["epochs_no_improve"] == 3
    assert "rng_states" in state


def test_last_checkpoint_captures_final_early_stopped_epoch(tmp_path):
    torch.manual_seed(0)
    model, opt, sched = _tiny_setup()
    last_ckpt = tmp_path / "last.pt"
    train.fit(
        model, _tiny_loader(), _tiny_loader(), opt, sched, nn.CrossEntropyLoss(), CPU,
        max_epochs=10, early_stopping_patience=2, early_stopping_metric="val_loss",
        best_metric=-1.0, last_checkpoint_path=last_ckpt, progress=False,
    )
    # Early stop fires at epoch 2; last.pt is saved before the break, so it records epoch 2.
    assert torch.load(last_ckpt, weights_only=False)["epoch"] == 2


def test_select_resume_checkpoint_prefers_higher_epoch(tmp_path):
    model, opt, sched = _tiny_setup()
    best_ckpt, last_ckpt = tmp_path / "best.pt", tmp_path / "last.pt"

    def save(path, epoch):
        _save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=None,
                               epoch=epoch, best_metric=0.5, epochs_no_improve=0, run_id="r")

    # best.pt from an older improvement, last.pt from a later (non-improving) epoch.
    save(best_ckpt, 4)
    save(last_ckpt, 9)
    assert select_resume_checkpoint(last_ckpt, best_ckpt) == last_ckpt
    assert select_resume_checkpoint(best_ckpt, last_ckpt) == last_ckpt

    # If best.pt is the newer one (improvement on the final epoch), it wins instead.
    save(best_ckpt, 12)
    assert select_resume_checkpoint(last_ckpt, best_ckpt) == best_ckpt


def test_select_resume_checkpoint_handles_missing_and_corrupt(tmp_path):
    missing, corrupt = tmp_path / "nope.pt", tmp_path / "corrupt.pt"
    assert select_resume_checkpoint(missing) is None   # fresh run

    corrupt.write_bytes(b"not a checkpoint")
    assert select_resume_checkpoint(corrupt) is None   # unreadable is ignored, not fatal

    model, opt, sched = _tiny_setup()
    good = tmp_path / "good.pt"
    _save_checkpoint(good, model=model, optimizer=opt, scheduler=sched, scaler=None,
                           epoch=3, best_metric=0.5, epochs_no_improve=1, run_id="r")
    assert select_resume_checkpoint(corrupt, missing, good) == good


# cuDNN determinism must hold on the resume path, where set_seed() is deliberately skipped.

def test_set_cudnn_determinism_standalone():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    set_cudnn_determinism()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_still_pins_cudnn():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    set_seed(0)   # fresh-run path keeps doing both jobs
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


RUN_NAME = "debug_0"   # configs/debug.yaml: experiment.condition=debug, train.seed=0


def _debug_config():
    return Config.from_yaml(resolve_config_path("debug"))


def _stub_main(tmp_path, monkeypatch):
    """Patch the heavy parts of main(); return (seed_calls, fit_kwargs, loaded_paths).

    set_cudnn_determinism, _resolve_resume_target, _load_checkpoint and the fingerprint logic
    are left REAL -- they are what these tests exercise.
    """
    seed_calls, fit_kwargs, loaded = [], {}, []
    monkeypatch.setattr(train, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(train, "configure_logging", lambda **kw: None)
    monkeypatch.setattr(train, "build_dataloaders",
                        lambda cfg: (_tiny_loader(), _tiny_loader(), _tiny_loader()))
    monkeypatch.setattr(train, "build_model", lambda mcfg, n_classes: nn.Linear(4, n_classes))
    monkeypatch.setattr(train, "set_seed", lambda s: seed_calls.append(s))

    real_load = train._load_checkpoint

    def _tracking_load(path, **kwargs):
        loaded.append(path)
        return real_load(path, **kwargs)

    monkeypatch.setattr(train, "_load_checkpoint", _tracking_load)

    def _fake_fit(*a, **k):
        fit_kwargs.update(k)
        start = k.get("start_epoch", 1)
        return {"best_metric_name": "val_loss", "best_metric": 0.2,
                "best_epoch": start, "epochs_run": start}

    monkeypatch.setattr(train, "fit", _fake_fit)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return seed_calls, fit_kwargs, loaded


def _write_checkpoint(path, *, epoch, config_fp, n_classes=24):
    """Write a checkpoint compatible with the model/optimizer/scheduler main() builds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = nn.Linear(4, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    _save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=None,
                           epoch=epoch, best_metric=0.5, epochs_no_improve=1, run_id=RUN_NAME,
                           config_metadata=config_fp)


# Three-state resume contract.

def test_main_no_flag_empty_dir_starts_fresh(tmp_path, monkeypatch):
    seed_calls, fit_kwargs, _ = _stub_main(tmp_path, monkeypatch)
    train.main("debug")
    assert seed_calls == [0], "a fresh run seeds from config train.seed"
    assert fit_kwargs["start_epoch"] == 1
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_main_no_flag_with_existing_checkpoint_refuses(tmp_path, monkeypatch):
    _stub_main(tmp_path, monkeypatch)
    _write_checkpoint(tmp_path / RUN_NAME / "last.pt", epoch=30,
                      config_fp=_config_fingerprint(_debug_config()))
    with pytest.raises(RuntimeError) as err:
        train.main("debug")
    message = str(err.value)
    assert RUN_NAME in message and "30" in message           # names the run dir and the epoch
    assert "--resume" in message and "--fresh" in message and "--run-id" in message


def test_main_resume_continues_from_higher_epoch(tmp_path, monkeypatch):
    seed_calls, fit_kwargs, loaded = _stub_main(tmp_path, monkeypatch)
    fingerprint = _config_fingerprint(_debug_config())
    _write_checkpoint(tmp_path / RUN_NAME / "best.pt", epoch=4, config_fp=fingerprint)
    _write_checkpoint(tmp_path / RUN_NAME / "last.pt", epoch=9, config_fp=fingerprint)

    train.main("debug", resume=True)

    assert loaded == [tmp_path / RUN_NAME / "last.pt"], "must resume the higher-epoch file"
    assert fit_kwargs["start_epoch"] == 10
    assert seed_calls == [], "a resumed run must not reseed; RNG comes from the checkpoint"
    assert torch.backends.cudnn.deterministic is True


def test_main_resume_without_checkpoint_raises(tmp_path, monkeypatch):
    _stub_main(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="no usable checkpoint"):
        train.main("debug", resume=True)


def test_main_resume_explicit_path(tmp_path, monkeypatch):
    _, fit_kwargs, loaded = _stub_main(tmp_path, monkeypatch)
    explicit = tmp_path / "elsewhere" / "snapshot.pt"
    _write_checkpoint(explicit, epoch=12, config_fp=_config_fingerprint(_debug_config()))
    train.main("debug", resume=str(explicit))
    assert loaded == [explicit]
    assert fit_kwargs["start_epoch"] == 13


def test_main_fresh_discards_existing_checkpoint(tmp_path, monkeypatch):
    seed_calls, fit_kwargs, loaded = _stub_main(tmp_path, monkeypatch)
    _write_checkpoint(tmp_path / RUN_NAME / "last.pt", epoch=7,
                      config_fp=_config_fingerprint(_debug_config()))

    train.main("debug", fresh=True)

    assert fit_kwargs["start_epoch"] == 1
    assert seed_calls == [0]
    assert loaded == [], "--fresh must not load the existing checkpoint"
    assert torch.backends.cudnn.deterministic is True


def test_cli_resume_and_fresh_are_mutually_exclusive():
    parser = train_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "debug", "--resume", "--fresh"])
    # Each alone parses into the three-state contract.
    assert parser.parse_args(["--config", "debug"]).resume is None
    assert parser.parse_args(["--config", "debug", "--resume"]).resume is True
    assert parser.parse_args(["--config", "debug", "--resume", "x.pt"]).resume == "x.pt"
    assert parser.parse_args(["--config", "debug", "--fresh"]).fresh is True


def test_cli_help_runs_from_a_clean_interpreter():
    """scripts/train.py must bootstrap sys.path on its own, with no PYTHONPATH help."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout and "--fresh" in result.stdout


# Config fingerprint + drift detection.

def test_saved_checkpoint_stores_config_fingerprint(tmp_path):
    fingerprint = _config_fingerprint(_debug_config())
    model, opt, sched = _tiny_setup()
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=None,
                           epoch=1, best_metric=0.1, epochs_no_improve=0, run_id="r",
                           config_metadata=fingerprint)
    assert torch.load(path, weights_only=False)["config"] == fingerprint


def test_fingerprint_is_machine_independent():
    local, remote = _debug_config(), _debug_config()
    local.source_path = "/home/me/repo/configs/debug.yaml"
    local.data.path = "/home/me/repo/data/GOLD_XYZ_OSC.0001_1024.hdf5"
    remote.source_path = "/kaggle/working/configs/debug.yaml"
    remote.data.path = "/kaggle/input/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5"
    # Same dataset basename + same values => same fingerprint despite different absolute paths.
    assert _config_fingerprint(local) == _config_fingerprint(remote)

    # A genuinely different dataset file must still register as drift.
    remote.data.path = "/kaggle/input/radioml2018/SOMETHING_ELSE.hdf5"
    assert _config_fingerprint(local) != _config_fingerprint(remote)


def test_resume_with_changed_config_raises(tmp_path, monkeypatch):
    _stub_main(tmp_path, monkeypatch)
    stale = _config_fingerprint(_debug_config())
    stale["train"]["learning_rate"] = 0.0003        # checkpoint predates an lr change
    _write_checkpoint(tmp_path / RUN_NAME / "last.pt", epoch=5, config_fp=stale)

    with pytest.raises(RuntimeError) as err:
        train.main("debug", resume=True)
    message = str(err.value)
    assert "train.learning_rate" in message         # names the specific key...
    assert "0.0003" in message and "0.001" in message   # ...and both values
    assert "--allow-config-change" in message


def test_resume_with_changed_config_allowed_warns(tmp_path, monkeypatch, caplog):
    _, fit_kwargs, _ = _stub_main(tmp_path, monkeypatch)
    stale = _config_fingerprint(_debug_config())
    stale["train"]["learning_rate"] = 0.0003
    _write_checkpoint(tmp_path / RUN_NAME / "last.pt", epoch=5, config_fp=stale)

    with caplog.at_level(logging.WARNING):
        train.main("debug", resume=True, allow_config_change=True)

    assert fit_kwargs["start_epoch"] == 6, "must proceed rather than raise"
    assert any("train.learning_rate" in r.getMessage() for r in caplog.records)


def test_resume_from_legacy_checkpoint_warns(tmp_path, monkeypatch, caplog):
    _, fit_kwargs, _ = _stub_main(tmp_path, monkeypatch)
    _write_checkpoint(tmp_path / RUN_NAME / "last.pt", epoch=3, config_fp=None)  # no config key

    with caplog.at_level(logging.WARNING):
        train.main("debug", resume=True)

    assert fit_kwargs["start_epoch"] == 4
    assert any("cannot verify config drift" in r.getMessage() for r in caplog.records)


def test_fit_forwards_checkpoint_metadata(tmp_path):
    """fit() carries the metadata through opaquely, without knowing what it is."""
    torch.manual_seed(0)
    model, opt, sched = _tiny_setup()
    last_ckpt = tmp_path / "last.pt"
    marker = {"anything": "opaque"}
    train.fit(
        model, _tiny_loader(), _tiny_loader(), opt, sched, nn.CrossEntropyLoss(), CPU,
        max_epochs=1, early_stopping_patience=5, early_stopping_metric="val_loss",
        last_checkpoint_path=last_ckpt, checkpoint_metadata=marker, progress=False,
    )
    assert torch.load(last_ckpt, weights_only=False)["config"] == marker


# Issue 3: GradScaler state is checkpointed only when AMP is active.

def test_checkpoint_omits_scaler_when_disabled(tmp_path):
    model = nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # inert on CPU / no-AMP
    ckpt = tmp_path / "ckpt.pt"
    _save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                           epoch=1, best_metric=0.1, epochs_no_improve=0, run_id="r")
    assert "scaler" not in torch.load(ckpt, weights_only=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler state needs CUDA")
def test_checkpoint_roundtrip_restores_scaler(tmp_path):
    model = nn.Linear(4, 3).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    saved_scale = scaler.get_scale()
    ckpt = tmp_path / "ckpt.pt"
    _save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                           epoch=1, best_metric=0.1, epochs_no_improve=0, run_id="r")
    fresh = torch.amp.GradScaler("cuda", enabled=True)
    _load_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sched,
                           scaler=fresh, device=torch.device("cuda"))
    assert fresh.get_scale() == saved_scale


# Issue 4: build_scheduler derives ReduceLROnPlateau mode from early_stopping_metric.

def _tcfg(sched_kwargs, metric="val_loss"):
    return TrainConfig(
        seed=0, batch_size=4, num_workers=0,
        optimizer={"name": "adam", "kwargs": {}},
        learning_rate=1e-3, weight_decay=0.0,
        lr_scheduler={"name": "reduce_on_plateau", "kwargs": sched_kwargs},
        max_epochs=5, early_stopping_patience=3,
        early_stopping_metric=metric, amp_enabled=False,
    )


def _optimizer():
    return torch.optim.Adam(nn.Linear(4, 3).parameters(), lr=1e-3)


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_build_scheduler_overrides_conflicting_mode(caplog):
    cfg = _tcfg({"mode": "max", "factor": 0.5}, metric="val_loss")  # val_loss implies min
    with caplog.at_level(logging.WARNING):
        sched = train.build_scheduler(_optimizer(), cfg)
    assert sched.mode == "min"
    warnings = _warnings(caplog)
    assert warnings and "disagrees with early_stopping_metric" in warnings[0].getMessage()


def test_build_scheduler_fills_missing_mode_without_warning(caplog):
    cfg = _tcfg({"factor": 0.5}, metric="val_accuracy")  # val_accuracy implies max
    with caplog.at_level(logging.WARNING):
        sched = train.build_scheduler(_optimizer(), cfg)
    assert sched.mode == "max"
    assert not _warnings(caplog)  # merely omitting mode is not a conflict


def test_build_scheduler_agreeing_mode_no_warning(caplog):
    cfg = _tcfg({"mode": "min"}, metric="val_loss")
    with caplog.at_level(logging.WARNING):
        sched = train.build_scheduler(_optimizer(), cfg)
    assert sched.mode == "min"
    assert not _warnings(caplog)


# Issue 5: per-SNR validation metrics.

def test_snr_bucket_edges():
    assert snr_bucket(-4) == -4
    assert snr_bucket(-3) == -4   # floors toward -inf
    assert snr_bucket(0) == 0
    assert snr_bucket(11) == 10
    assert snr_bucket(30) == 30


# Progress bars: display-only, auto-off when stderr is not a TTY.

def test_progress_disabled_resolution(monkeypatch):
    assert train._progress_disabled(False) is True    # explicit off
    assert train._progress_disabled(True) is False    # explicit on
    # None auto-detects: a redirected (non-TTY) stderr must suppress the bar so nohup/Kaggle
    # logs don't fill with carriage-return spam.
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert train._progress_disabled(None) is True


def test_validate_returns_per_snr_breakdown():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    xb = torch.randn(6, 4)
    yb = torch.randint(0, 3, (6,))
    snr = torch.tensor([-4, -4, 0, 0, 10, 11], dtype=torch.long)
    out = train.validate(model, [(xb, yb, snr)], nn.CrossEntropyLoss(), CPU)

    assert set(out) == {"loss", "accuracy", "snr_accuracy", "accuracy_snr_geq_0db"}
    # snr=11 buckets to 10, so the bins are exactly {-4, 0, 10}.
    assert set(out["snr_accuracy"]) == {-4, 0, 10}
    for acc in out["snr_accuracy"].values():
        assert 0.0 <= acc <= 1.0
    assert 0.0 <= out["accuracy_snr_geq_0db"] <= 1.0
