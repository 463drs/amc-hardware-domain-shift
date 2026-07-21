"""Config-layer tests: the shipped YAMLs load, and every fail-fast guard fires.

The design contract (see src/config.py) is that all result-affecting fields are required
with NO defaults, so a missing or nonsensical value fails at load time rather than silently
mid-experiment. These tests pin that contract for the training/model additions.
"""

import copy

import pytest
import yaml

from src.config import (
    Config,
    ExperimentConfig,
    ModelConfig,
    NamedComponentConfig,
    TrainConfig,
)

# A minimal, fully-valid config used as the base for mutation tests. Kept independent of the
# repo's shipped YAMLs so a change to those defaults can't quietly weaken these tests.
_VALID = {
    "data": {
        "path": "data/GOLD_XYZ_OSC.0001_1024.hdf5",
        "frames_per_pair": 20,
        "subset_seed": 1234,
        "snr_min": 20,
        "snr_max": 30,
        "split": [0.7, 0.15, 0.15],
        "split_seed": 5678,
        "normalization": "unit_power",
    },
    "model": {"dropout_p": 0.5, "init_scheme": "kaiming_linear"},
    "train": {
        "seed": 0,
        "batch_size": 16,
        "num_workers": 0,
        "optimizer": {"name": "adam", "kwargs": {}},
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "lr_scheduler": {"name": "reduce_on_plateau", "kwargs": {"mode": "min"}},
        "max_epochs": 3,
        "early_stopping_patience": 2,
        "early_stopping_metric": "val_loss",
        "amp_enabled": False,
    },
    "experiment": {
        "project": "amc-hardware-domain-shift",
        "condition": "unit-test",
        "mode": "disabled",
    },
}


def _write(tmp_path, cfg: dict):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_shipped_configs_load():
    """The configs committed to the repo must parse and validate."""
    for name in ("baseline", "debug"):
        cfg = Config.from_yaml(f"configs/{name}.yaml")
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.train, TrainConfig)
        # optimizer/lr_scheduler are wrapped into NamedComponentConfig, not left as dicts.
        assert isinstance(cfg.train.optimizer, NamedComponentConfig)
        assert isinstance(cfg.train.lr_scheduler, NamedComponentConfig)


def test_valid_base_loads(tmp_path):
    cfg = Config.from_yaml(_write(tmp_path, _VALID))
    assert cfg.train.learning_rate == pytest.approx(0.001)
    assert cfg.train.amp_enabled is False
    assert cfg.train.optimizer.name == "adam"
    assert cfg.train.lr_scheduler.kwargs == {"mode": "min"}
    assert cfg.model.dropout_p == pytest.approx(0.5)


# Each case mutates the valid base into an invalid one; loading must raise.
_BAD_CASES = {
    "missing_model_section": lambda c: c.pop("model"),
    "missing_amp_enabled": lambda c: c["train"].pop("amp_enabled"),
    "missing_learning_rate": lambda c: c["train"].pop("learning_rate"),
    "amp_enabled_not_bool": lambda c: c["train"].__setitem__("amp_enabled", 1),
    "learning_rate_zero": lambda c: c["train"].__setitem__("learning_rate", 0),
    "weight_decay_negative": lambda c: c["train"].__setitem__("weight_decay", -1),
    "max_epochs_zero": lambda c: c["train"].__setitem__("max_epochs", 0),
    "patience_zero": lambda c: c["train"].__setitem__("early_stopping_patience", 0),
    "unknown_metric": lambda c: c["train"].__setitem__("early_stopping_metric", "val_f1"),
    "optimizer_typo_key": lambda c: c["train"]["optimizer"].__setitem__("nmae", "x"),
    "optimizer_bare_string": lambda c: c["train"].__setitem__("optimizer", "adam"),
    "unknown_train_key": lambda c: c["train"].__setitem__("lr", 0.1),
    "dropout_out_of_range": lambda c: c["model"].__setitem__("dropout_p", 1.0),
    "empty_init_scheme": lambda c: c["model"].__setitem__("init_scheme", ""),
    "missing_experiment_section": lambda c: c.pop("experiment"),
    "unknown_wandb_mode": lambda c: c["experiment"].__setitem__("mode", "bogus"),
    "empty_condition": lambda c: c["experiment"].__setitem__("condition", ""),
}


@pytest.mark.parametrize("name", list(_BAD_CASES))
def test_invalid_config_fails_fast(tmp_path, name):
    cfg = copy.deepcopy(_VALID)
    _BAD_CASES[name](cfg)
    with pytest.raises((ValueError, TypeError)):
        Config.from_yaml(_write(tmp_path, cfg))
