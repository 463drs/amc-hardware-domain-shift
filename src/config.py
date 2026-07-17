"""Configuration system for the AMC domain-shift experiments.

Design rules (thesis reproducibility):
  * Everything that changes results comes ONLY from the YAML config: subset size,
    all seeds, the split, the normalization method, and the batch size.
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
from typing import Tuple

import numpy as np
import torch
import yaml

# Ratios must sum to 1 within this tolerance (guards against typos like [0.7, 0.15, 0.1]).
_SPLIT_SUM_TOL = 1e-6


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
class TrainConfig:
    """Training-side knobs that still affect results and therefore live in the config.

    NOTE: `seed` here is the TRAINING seed (model init, shuffling, dropout). It is
    intentionally separate from data.subset_seed and data.split_seed so that the exact
    same data can be reused across many training seeds.
    """

    seed: int
    batch_size: int
    num_workers: int

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"train.batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"train.num_workers must be >= 0, got {self.num_workers}")


@dataclass
class Config:
    """Top-level experiment configuration."""

    data: DataConfig
    train: TrainConfig
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
        for section in ("data", "train"):
            if section not in raw:
                raise ValueError(f"Config file {path} is missing required section '{section}'.")

        # Reject unknown keys so a typo like `normalisation:` fails loudly instead of
        # silently falling back to a default.
        _reject_unknown_keys(raw["data"], DataConfig, "data")
        _reject_unknown_keys(raw["train"], TrainConfig, "train")

        config = cls(
            data=DataConfig(**raw["data"]),
            train=TrainConfig(**raw["train"]),
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


def set_seed(seed: int) -> None:
    """Seed all global RNGs (random, numpy, torch, CUDA) for reproducible training.

    This governs training-time randomness (model init, dropout, DataLoader shuffling).
    The stratified subsetting and splitting use their OWN local generators seeded from
    the config (subset_seed / split_seed), so they stay identical regardless of this
    training seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic. benchmark=False disables autotuning that can pick
    # different (nondeterministic) kernels between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
