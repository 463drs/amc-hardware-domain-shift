"""Configuration system for the AMC domain-shift experiments.

Design rules (thesis reproducibility):
  * Everything that changes results comes ONLY from the YAML config: the data subset and
    split (sizes, seeds, SNR range, normalization), the model hyperparameters (dropout,
    init scheme), and the training recipe (optimizer, learning rate, weight decay, LR
    schedule, epochs/early-stopping, batch size, mixed precision).
  * Device selection (CUDA vs CPU) is auto-detected at runtime and is deliberately
    NOT a config field, so the same config produces the same experiment on any
    machine. Hardware must never silently change the experiment.

The config is loaded into typed dataclasses so that typos and bad values fail loudly
at load time rather than silently mid-experiment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

# Ratios must sum to 1 within this tolerance (guards against typos like [0.7, 0.15, 0.1]).
_SPLIT_SUM_TOL = 1e-6

# Early stopping watches exactly one validation metric (see TrainConfig.early_stopping_metric).
_EARLY_STOPPING_METRICS: Tuple[str, ...] = ("val_loss", "val_accuracy")

# W&B run modes (see ExperimentConfig.mode). "disabled" makes tracking a no-op so the debug
# config and the test suite run with no W&B account, network, or even the wandb package.
_WANDB_MODES: Tuple[str, ...] = ("online", "offline", "disabled")


@dataclass
class DataConfig:
    """Everything that defines WHICH frames are used and how they are normalized."""

    path: str
    frames_per_pair: int          # N frames drawn from each (class, SNR) cell; 4096 == full cell
    subset_seed: int              # fixed seed for stratified subsetting; independent of training seed
    snr_min: int                  # inclusive lower SNR bound (dB)
    snr_max: int                  # inclusive upper SNR bound (dB)
    split: Tuple[float, float, float]  # (train, val, test) fractions, within each (class, SNR) cell
    split_seed: int               # fixed seed for the stratified split; independent of training seed
    normalization: str            # named per-frame normalization method (see data.NORMALIZERS)

    def __post_init__(self) -> None:
        # split may arrive from YAML as a list; normalize to a tuple and validate.
        self.split = tuple(float(x) for x in self.split)  # type: ignore[assignment]
        if len(self.split) != 3:
            raise ValueError(f"data.split must have 3 values (train, val, test), got {self.split!r}")
        if any(x < 0.0 for x in self.split):
            raise ValueError(f"data.split fractions must be non-negative, got {self.split!r}")
        if abs(sum(self.split) - 1.0) > _SPLIT_SUM_TOL:
            raise ValueError(f"data.split must sum to 1.0, got {self.split!r} (sum={sum(self.split)})")
        if self.frames_per_pair < 1:
            raise ValueError(f"data.frames_per_pair must be >= 1, got {self.frames_per_pair}")
        if self.snr_min > self.snr_max:
            raise ValueError(f"data.snr_min ({self.snr_min}) must be <= data.snr_max ({self.snr_max})")
        if not isinstance(self.normalization, str) or not self.normalization:
            raise ValueError(f"data.normalization must be a non-empty string, got {self.normalization!r}")


@dataclass
class NamedComponentConfig:
    """A component chosen by name plus the kwargs its constructor receives.

    Used for both `train.optimizer` and `train.lr_scheduler`. The name is resolved to a
    concrete class by a registry in the training code (kept out of this module so config
    loading never needs to import torch.optim); `kwargs` are forwarded verbatim to it.
    Structure is validated at load time so a typo fails here, not several epochs in.
    """

    name: str
    kwargs: Dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"component name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.kwargs, dict):
            raise ValueError(f"component kwargs must be a mapping, got {self.kwargs!r}")


def _coerce_named_component(value: object, label: str) -> "NamedComponentConfig":
    """Turn a raw YAML mapping ({name, kwargs}) into a validated NamedComponentConfig.

    Accepts an already-built NamedComponentConfig (e.g. constructed directly in tests)
    unchanged. Rejects unknown keys so `nmae:`-style typos fail loudly, matching how the
    top-level config sections are validated.
    """
    if isinstance(value, NamedComponentConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must be a mapping with keys 'name' and 'kwargs', got {value!r}"
        )
    _reject_unknown_keys(value, NamedComponentConfig, label)
    return NamedComponentConfig(**value)


@dataclass
class TrainConfig:
    """Training-side knobs that still affect results and therefore live in the config.

    NOTE: `seed` here is the TRAINING seed (model init, shuffling, dropout). It is
    intentionally separate from data.subset_seed and data.split_seed so that the exact
    same data can be reused across many training seeds.
    """

    seed: int
    batch_size: int
    num_workers: int
    optimizer: NamedComponentConfig       # {name, kwargs}; learning_rate/weight_decay passed separately
    learning_rate: float
    weight_decay: float
    lr_scheduler: NamedComponentConfig    # {name, kwargs}, e.g. reduce_on_plateau
    max_epochs: int                       # hard ceiling on training length, regardless of early stopping
    early_stopping_patience: int          # epochs without improvement tolerated before stopping
    early_stopping_metric: str            # which validation metric early stopping watches
    amp_enabled: bool                     # mixed precision; config-controlled so it can be disabled per run

    def __post_init__(self) -> None:
        # optimizer/lr_scheduler arrive from YAML as plain mappings; validate and wrap them.
        self.optimizer = _coerce_named_component(self.optimizer, "train.optimizer")
        self.lr_scheduler = _coerce_named_component(self.lr_scheduler, "train.lr_scheduler")
        # learning_rate/weight_decay may arrive as ints from YAML (e.g. `0`); store as floats.
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)

        if self.batch_size < 1:
            raise ValueError(f"train.batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"train.num_workers must be >= 0, got {self.num_workers}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"train.learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"train.weight_decay must be >= 0, got {self.weight_decay}")
        if self.max_epochs < 1:
            raise ValueError(f"train.max_epochs must be >= 1, got {self.max_epochs}")
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"train.early_stopping_patience must be >= 1, got {self.early_stopping_patience}"
            )
        if self.early_stopping_metric not in _EARLY_STOPPING_METRICS:
            raise ValueError(
                f"train.early_stopping_metric must be one of {list(_EARLY_STOPPING_METRICS)}, "
                f"got {self.early_stopping_metric!r}"
            )
        # A truthiness check would accept 1/0; require an actual bool so YAML `1` fails loudly.
        if not isinstance(self.amp_enabled, bool):
            raise ValueError(f"train.amp_enabled must be a boolean, got {self.amp_enabled!r}")


@dataclass
class ModelConfig:
    """Model hyperparameters that affect results.

    These were previously hardcoded in models.py; hoisting them here makes the model fully
    specified by the YAML. `init_scheme` is a named weight-init method resolved by a registry
    in models.py (kept there, like data.NORMALIZERS, so this module never needs to know the
    concrete init functions).
    """

    dropout_p: float
    init_scheme: str

    def __post_init__(self) -> None:
        self.dropout_p = float(self.dropout_p)
        if not (0.0 <= self.dropout_p < 1.0):
            raise ValueError(f"model.dropout_p must be in [0.0, 1.0), got {self.dropout_p}")
        if not isinstance(self.init_scheme, str) or not self.init_scheme:
            raise ValueError(
                f"model.init_scheme must be a non-empty string, got {self.init_scheme!r}"
            )


@dataclass
class ExperimentConfig:
    """Experiment-tracking metadata (Weights & Biases).

    Unlike the other sections this does NOT change results -- it is provenance/organization.
    It still lives in the config so a run is fully described by its YAML, and, crucially, so
    the run identity is a stable function of `condition` (+ train seed) rather than of the
    config filename: a deterministic W&B run id is what lets a restarted Kaggle session
    continue the SAME run/curve instead of spawning a fragmented duplicate.

    `condition` is the experimental condition (README: baseline, rtl_sdr_*, calibrated, ...),
    used as the W&B group. `project` is the same across all conditions. `mode` is one of
    online/offline/disabled; "disabled" turns tracking into a no-op (used by debug + tests).
    """

    project: str
    condition: str
    mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.project, str) or not self.project:
            raise ValueError(f"experiment.project must be a non-empty string, got {self.project!r}")
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError(f"experiment.condition must be a non-empty string, got {self.condition!r}")
        if self.mode not in _WANDB_MODES:
            raise ValueError(
                f"experiment.mode must be one of {list(_WANDB_MODES)}, got {self.mode!r}"
            )


@dataclass
class Config:
    """Top-level experiment configuration."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    experiment: ExperimentConfig
    # Absolute path of the YAML this config was loaded from (provenance; not an experiment value).
    source_path: str = field(default="", compare=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate a config from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ValueError(f"Config file {path} did not parse to a mapping.")
        for section in ("data", "model", "train", "experiment"):
            if section not in raw:
                raise ValueError(f"Config file {path} is missing required section '{section}'.")

        # Reject unknown keys so a typo like `normalisation:` fails loudly instead of
        # silently falling back to a default.
        _reject_unknown_keys(raw["data"], DataConfig, "data")
        _reject_unknown_keys(raw["model"], ModelConfig, "model")
        _reject_unknown_keys(raw["train"], TrainConfig, "train")
        _reject_unknown_keys(raw["experiment"], ExperimentConfig, "experiment")

        config = cls(
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            experiment=ExperimentConfig(**raw["experiment"]),
            source_path=str(path.resolve()),
        )
        # Anchor a repo-relative data.path so the dataset opens regardless of CWD.
        config.data.path = _anchor_path(config.data.path)
        return config


def _reject_unknown_keys(section: dict, dataclass_type, name: str) -> None:
    """Raise if the YAML section contains keys not present on the dataclass."""
    allowed = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section '{name}'. "
            f"Allowed keys: {sorted(allowed)}."
        )


def set_cudnn_determinism() -> None:
    """Pin cuDNN to deterministic kernels. benchmark=False disables the autotuner, which can
    otherwise pick different (nondeterministic) kernels between runs.

    Split out of set_seed so a RESUMED run can call it directly: such a run deliberately skips
    reseeding (it restores the checkpointed RNG stream instead), and without this it would
    silently fall back to PyTorch's non-deterministic cuDNN defaults.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    """Seed all global RNGs (random, numpy, torch, CUDA) for reproducible training.

    This governs training-time randomness (model init, dropout, DataLoader shuffling).
    The stratified subsetting and splitting use their OWN local generators seeded from
    the config (subset_seed / split_seed), so they stay identical regardless of this
    training seed.

    Also pins cuDNN determinism, so the fresh-run path needs only this one call.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_cudnn_determinism()

# Repo root and the directory holding the YAML configs.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs"


def _anchor_path(path: str | Path) -> str:
    """Resolve a relative path against the repo root; leave absolute paths unchanged.

    Config data.path is written repo-relative (e.g. 'data/...'), so anchoring lets it open
    from ANY working directory -- a notebook launched in notebooks/, a script run elsewhere,
    or Kaggle -- instead of depending on the process CWD.
    """
    p = Path(path)
    return str(p if p.is_absolute() else _REPO_ROOT / p)

def resolve_config_path(name: str) -> Path:
    """Accept a bare config name, a name with extension, or a full path.

    Examples that all resolve to configs/baseline.yaml:
        baseline
        baseline.yaml
        configs/baseline.yaml
    """
    p = Path(name)
    # If it's an existing path as given (full or relative), use it directly.
    if p.exists():
        return p
    # Otherwise treat it as a name inside the configs directory.
    stem = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    candidate = _CONFIG_DIR / stem
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Config {name!r} not found. Looked for {p} and {candidate}. "
        f"Available: {sorted(f.name for f in _CONFIG_DIR.glob('*.yaml'))}"
    )

def resolve_data_path(
    config: str | Path | None = None,
    path: str | Path | None = None,
    default: str | Path | None = None,
) -> Tuple[str, Path | None]:
    """Resolve the HDF5 data path from a config (name or path) or an explicit path.

    This is the single entry point used by the CLI scripts and by notebooks so that both
    accept the same inputs. Precedence: `config` > `path` > `default`.

    Parameters
    ----------
    config : a bare config name ("baseline"), a filename ("baseline.yaml"), or a path --
             anything resolve_config_path accepts. When given, the HDF5 path is read from
             the config's data.path.
    path   : an explicit HDF5 path, used only when `config` is None.
    default: fallback HDF5 path used when both `config` and `path` are None.

    Returns
    -------
    (data_path, config_path) where config_path is the resolved YAML path when a config was
    used, else None (useful for provenance and for naming output files after the config).
    """
    if config is not None:
        config_path = resolve_config_path(str(config))
        return Config.from_yaml(config_path).data.path, config_path  # already anchored
    if path is not None:
        return _anchor_path(path), None
    if default is not None:
        return _anchor_path(default), None
    raise ValueError("Provide one of: config, path, or default.")


def get_device() -> torch.device:
    """Auto-detect the compute device. NOT configurable, by design."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
