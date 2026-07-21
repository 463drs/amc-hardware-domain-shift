"""Data loading for RadioML 2018.01A.

Responsibilities:
  * Select a STRATIFIED subset that preserves the (class, SNR) grid, reproducibly.
  * Split each (class, SNR) cell into train/val/test so every split covers the full grid.
  * Expose a lazy, fork-safe PyTorch Dataset that yields (iq, class_index, snr).

Reproducibility model:
  * Subset selection is seeded by data.subset_seed, per (class, SNR) cell.
  * The split is seeded by data.split_seed, per (class, SNR) cell.
  * Both use LOCAL numpy generators, so they are identical across runs and independent
    of the training seed and of the machine.

Dataset facts (verified from the file, not assumed):
  * X: (N, 1024, 2) float32 -- axis layout is (time, [I, Q]).
  * Y: (N, 24)      int64   -- one-hot class labels; NO class names are stored in the file.
  * Z: (N, 1)       int64   -- SNR in dB.
The column->name mapping therefore comes from the published DeepSig ordering below and is
validated by the constellation sanity check (scripts/sanity_check.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config, DataConfig
from .logging_utils import get_logger

logger = get_logger(__name__)

# Constants

MODULATION_CLASSES: Tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM", "128QAM",
    "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
)

# HDF5 dataset keys (verified present in the file).
KEY_X, KEY_Y, KEY_Z = "X", "Y", "Z"

# Numerical guard for per-frame normalization (prevents divide-by-zero on a silent frame).
# This is a numerical epsilon, not an experiment parameter.
_NORM_EPS: float = 1e-12

  # >= abs(min possible SNR); keeps seed components non-negative
_SNR_SEED_OFFSET = 32 
# Per-frame normalization (named + configurable)

def _normalize_none(iq: torch.Tensor) -> torch.Tensor:
    """No normalization; return the frame as-is."""
    return iq


def _normalize_unit_power(iq: torch.Tensor) -> torch.Tensor:
    """Scale so the mean per-sample power equals 1.

    iq has shape (2, T) with channel 0 = I, channel 1 = Q.
    Mean power P = mean_t( I[t]^2 + Q[t]^2 ). After scaling by 1/sqrt(P), mean power == 1.
    """
    # sum over the 2 channels -> instantaneous power per time step (T,), then mean over time.
    power = iq.pow(2).sum(dim=0).mean()
    return iq / torch.sqrt(power + _NORM_EPS)


# Registry so the config can select a method by name and new methods are trivial to add.
NORMALIZERS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none": _normalize_none,
    "unit_power": _normalize_unit_power,
}


def _get_normalizer(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in NORMALIZERS:
        raise ValueError(
            f"Unknown normalization {name!r}. Available: {sorted(NORMALIZERS)}."
        )
    return NORMALIZERS[name]

# Low-level readers

def read_labels_and_snr(path: str | Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Read per-frame class index and SNR into RAM (small), leaving X on disk.

    Returns
    -------
    class_idx : (N,) int16   -- argmax of the one-hot Y.
    snr       : (N,) int16   -- SNR in dB.
    frame_len : int          -- number of complex samples per frame (X.shape[1]).
    n_classes : int          -- number of classes (Y.shape[1]).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        y = f[KEY_Y][:]                       # (N, C) one-hot; ~245 MB for the full file
        z = f[KEY_Z][:].reshape(-1)           # (N,)
        frame_len = int(f[KEY_X].shape[1])
        n_classes = int(f[KEY_Y].shape[1])
    class_idx = y.argmax(axis=1).astype(np.int16)
    snr = z.astype(np.int16)
    return class_idx, snr, frame_len, n_classes

# Stratified subsetting 

def _cell_groups(class_idx: np.ndarray, snr: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
    """Group row positions by (class, SNR) cell.

    Returns a list of (class_value, snr_value, positions) where `positions` are indices
    into the given arrays, sorted ascending. Does not assume any file ordering.
    """
    # Encode (class, snr) into a single sortable key. SNR is offset to stay non-negative.
    snr_offset = int(snr.min()) if snr.size else 0
    key = class_idx.astype(np.int64) * 1_000_000 + (snr.astype(np.int64) - snr_offset)
    order = np.argsort(key, kind="stable")   # stable keeps ascending positions within a cell
    key_sorted = key[order]
    boundaries = np.nonzero(np.diff(key_sorted))[0] + 1
    groups: List[Tuple[int, int, np.ndarray]] = []
    for grp in np.split(order, boundaries):
        c = int(class_idx[grp[0]])
        s = int(snr[grp[0]])
        groups.append((c, s, grp))
    return groups

def select_subset_indices(
    class_idx: np.ndarray,
    snr: np.ndarray,
    frames_per_pair: int,
    subset_seed: int,
    snr_min: int,
    snr_max: int,
) -> np.ndarray:
    """Select a stratified subset preserving the (class, SNR) grid.

    From each (class, SNR) cell within [snr_min, snr_max], take `frames_per_pair` frames
    (or all frames in the cell if it holds fewer). Selection per cell is seeded by
    (subset_seed, class, snr), so it is:
      * identical across runs and experimental conditions, and
      * stable when the SNR range changes (each surviving cell keeps its exact frames).

    Returns SORTED global indices -- HDF5 reads sorted indices far faster than random ones.
    """
    if frames_per_pair < 1:
        raise ValueError(f"frames_per_pair must be >= 1, got {frames_per_pair}")

    in_range = (snr >= snr_min) & (snr <= snr_max)
    rows = np.nonzero(in_range)[0]
    if rows.size == 0:
        raise ValueError(f"No frames in SNR range [{snr_min}, {snr_max}].")

    selected: List[np.ndarray] = []
    for c, s, local_positions in _cell_groups(class_idx[rows], snr[rows]):
        cell_rows = rows[local_positions]           # ascending global indices for this cell
        n = cell_rows.size
        if frames_per_pair >= n:
            pick = cell_rows                        # take the whole cell
        else:
            snr_seed = int(s) + _SNR_SEED_OFFSET
            rng = np.random.default_rng([subset_seed, int(c), snr_seed])
            chosen = rng.choice(n, size=frames_per_pair, replace=False)
            pick = cell_rows[chosen]
        selected.append(pick)

    return np.sort(np.concatenate(selected))

# Stratified train/val/test split

def stratified_split(
    class_idx: np.ndarray,
    snr: np.ndarray,
    split: Tuple[float, float, float],
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split WITHIN each (class, SNR) cell so every split covers the full grid.

    Operates on positions 0..len(class_idx)-1 (typically positions into an already-selected
    subset). Each cell is shuffled with a generator seeded by (split_seed, class, snr) and
    partitioned by the given fractions. Returns (train_pos, val_pos, test_pos), each sorted.
    """
    train: List[np.ndarray] = []
    val: List[np.ndarray] = []
    test: List[np.ndarray] = []

    for c, s, positions in _cell_groups(class_idx, snr):
        snr_seed = int(s) + _SNR_SEED_OFFSET
        rng = np.random.default_rng([split_seed, int(c), snr_seed])
        shuffled = positions[rng.permutation(positions.size)]
        m = positions.size
        n_train = int(round(split[0] * m))
        n_val = int(round(split[1] * m))
        # Clamp so counts never exceed the cell (rounding can overshoot on tiny cells).
        n_train = min(n_train, m)
        n_val = min(n_val, m - n_train)
        train.append(shuffled[:n_train])
        val.append(shuffled[n_train:n_train + n_val])
        test.append(shuffled[n_train + n_val:])

    def _finish(parts: List[np.ndarray]) -> np.ndarray:
        return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)

    return _finish(train), _finish(val), _finish(test)


# PyTorch Dataset

class RadioMLDataset(Dataset):
    """Lazy, fork-safe Dataset over a fixed set of global HDF5 row indices.

    Yields (iq, class_index, snr):
      * iq          : float32 tensor of shape (2, T), channel 0 = I, channel 1 = Q.
                      The raw file stores each frame as (T, 2) = (time, [I, Q]); we
                      transpose to channels-first for nn.Conv1d. This transpose is the
                      single most likely silent bug, so it is explicit and asserted below.
      * class_index : int, column index into MODULATION_CLASSES.
      * snr         : int, SNR in dB (needed at eval time to build accuracy-vs-SNR curves).

    The HDF5 handle is opened lazily on first access INSIDE each worker process, never in
    __init__. This avoids sharing/pickling an open h5py handle across the DataLoader fork
    (or spawn on Windows), which otherwise corrupts reads.
    """

    def __init__(
        self,
        path: str | Path,
        indices: np.ndarray,
        class_idx: np.ndarray,
        snr: np.ndarray,
        normalization: str,
        frame_length: int,
    ) -> None:
        assert len(indices) == len(class_idx) == len(snr), "indices/labels/snr length mismatch"
        self.path = str(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_idx = np.asarray(class_idx, dtype=np.int64)
        self.snr = np.asarray(snr, dtype=np.int64)
        self.normalization = normalization
        self.frame_length = int(frame_length)
        self._normalize = _get_normalizer(normalization)  # validates the name early
        self._file: Optional[h5py.File] = None             # opened lazily, per worker

    def __len__(self) -> int:
        return len(self.indices)

    def _handle(self) -> h5py.File:
        """Return this worker's own read-only HDF5 handle, opening it on first use."""
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, int]:
        global_row = int(self.indices[i])
        raw = self._handle()[KEY_X][global_row]          # (T, 2) float32, layout (time, [I, Q])
        # Transpose (T, 2) -> (2, T): channels-first for Conv1d. ascontiguousarray is required
        # because torch.from_numpy cannot take the negative strides a bare .T produces.
        iq_np = np.ascontiguousarray(raw.T, dtype=np.float32)
        assert iq_np.shape == (2, self.frame_length), (
            f"expected (2, {self.frame_length}) after transpose, got {iq_np.shape}"
        )
        iq = self._normalize(torch.from_numpy(iq_np))
        return iq, int(self.class_idx[i]), int(self.snr[i])

    def __getstate__(self) -> dict:
        # Never pickle an open HDF5 handle to a worker; each worker reopens its own.
        state = self.__dict__.copy()
        state["_file"] = None
        return state

# Builders (tie subset + split + Dataset together from a Config)

def _build_split_indices(
    cfg: DataConfig,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, int, int]:
    """Compute global row indices for each split from a DataConfig.

    Returns (split_global_indices, class_idx_all, snr_all, frame_len, n_classes).
    """
    class_idx, snr, frame_len, n_classes = read_labels_and_snr(cfg.path)

    subset = select_subset_indices(
        class_idx=class_idx,
        snr=snr,
        frames_per_pair=cfg.frames_per_pair,
        subset_seed=cfg.subset_seed,
        snr_min=cfg.snr_min,
        snr_max=cfg.snr_max,
    )
    subset_classes = class_idx[subset]
    subset_snr = snr[subset]

    train_pos, val_pos, test_pos = stratified_split(
        class_idx=subset_classes,
        snr=subset_snr,
        split=cfg.split,
        split_seed=cfg.split_seed,
    )

    # Map positions-within-subset back to global row indices (kept sorted for fast reads).
    splits = {
        "train": np.sort(subset[train_pos]),
        "val": np.sort(subset[val_pos]),
        "test": np.sort(subset[test_pos]),
    }
    return splits, class_idx, snr, frame_len, n_classes


def build_datasets(
    config: Config, verbose: bool = True
) -> Tuple[RadioMLDataset, RadioMLDataset, RadioMLDataset]:
    """Build (train, val, test) datasets from a validated Config."""
    cfg = config.data
    splits, class_idx, snr, frame_len, _ = _build_split_indices(cfg)

    def _make(name: str) -> RadioMLDataset:
        idx = splits[name]
        return RadioMLDataset(
            path=cfg.path,
            indices=idx,
            class_idx=class_idx[idx],
            snr=snr[idx],
            normalization=cfg.normalization,
            frame_length=frame_len,
        )

    train_ds, val_ds, test_ds = _make("train"), _make("val"), _make("test")

    if verbose:
        total = len(train_ds) + len(val_ds) + len(test_ds)
        logger.info(
            "frames_per_pair=%d snr=[%d,%d] normalization=%s",
            cfg.frames_per_pair, cfg.snr_min, cfg.snr_max, cfg.normalization,
        )
        logger.info(
            "train=%d  val=%d  test=%d  total=%d",
            len(train_ds), len(val_ds), len(test_ds), total,
        )

    return train_ds, val_ds, test_ds


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible RNG seed.

    The Dataset itself is deterministic, but seeding workers keeps any future per-worker
    randomness reproducible and free of cross-worker correlation.
    """
    import random as _random

    base = torch.initial_seed() % (2 ** 32)
    seed = (base + worker_id) % (2 ** 32)
    np.random.seed(seed)
    _random.seed(seed)


def build_dataloaders(
    config: Config, verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders with reproducible shuffling and fork-safe workers.

    Only the train loader shuffles; its shuffle order is seeded by train.seed so a fixed
    training seed reproduces the batch order exactly. Device is auto-detected elsewhere and
    never affects which data is loaded.
    """
    train_ds, val_ds, test_ds = build_datasets(config, verbose=verbose)

    generator = torch.Generator()
    generator.manual_seed(config.train.seed)

    common = dict(
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        worker_init_fn=_worker_init_fn if config.train.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader
