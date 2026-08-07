"""Tests for src.data.

The property under test is the one that is easiest to break silently: `preload=True` reads
the whole subset in blocks and normalizes it BATCHED (BATCHED_NORMALIZERS), while
`preload=False` reads one frame at a time and normalizes it PER-FRAME (NORMALIZERS). Those
are two independent implementations of the same math, so a run's data must not depend on
which one the config picked.

Uses a tiny synthetic HDF5 with the real layout (X: (N, T, 2), Y: one-hot, Z: SNR), so the
tests need neither the 24 GB RadioML file nor the generated subset.
"""

import h5py
import numpy as np
import pytest
import torch

from src.data import (
    BATCHED_NORMALIZERS,
    KEY_X,
    KEY_Y,
    KEY_Z,
    NORMALIZERS,
    RadioMLDataset,
    read_labels_and_snr,
)

N_FRAMES = 40
FRAME_LEN = 16
N_CLASSES = 4
SNR_VALUES = (-10, 0, 10, 20)


@pytest.fixture(scope="module")
def synthetic_h5(tmp_path_factory):
    """A small HDF5 with the same keys/shapes/dtypes as RadioML 2018.01A."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_FRAMES, FRAME_LEN, 2)).astype(np.float32)
    # A frame of pure zeros exercises the divide-by-zero guard in both normalizer paths.
    x[0] = 0.0

    class_idx = np.arange(N_FRAMES) % N_CLASSES
    y = np.zeros((N_FRAMES, N_CLASSES), dtype=np.int64)
    y[np.arange(N_FRAMES), class_idx] = 1
    z = np.asarray([SNR_VALUES[i % len(SNR_VALUES)] for i in range(N_FRAMES)], dtype=np.int64)

    path = tmp_path_factory.mktemp("data") / "synthetic.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset(KEY_X, data=x)
        f.create_dataset(KEY_Y, data=y)
        f.create_dataset(KEY_Z, data=z.reshape(-1, 1))
    return path


def _make(path, normalization, preload, indices):
    class_idx, snr, frame_len, _ = read_labels_and_snr(path)
    return RadioMLDataset(
        path=path,
        indices=indices,
        class_idx=class_idx[indices],
        snr=snr[indices],
        normalization=normalization,
        frame_length=frame_len,
        preload=preload,
    )


@pytest.mark.parametrize("normalization", sorted(NORMALIZERS))
def test_preload_matches_lazy(synthetic_h5, normalization):
    """Identical frames, labels and SNRs regardless of preload -- the check that matters."""
    # Non-contiguous, sorted indices: what the stratified subset actually produces.
    indices = np.arange(0, N_FRAMES, 3)
    preloaded = _make(synthetic_h5, normalization, True, indices)
    lazy = _make(synthetic_h5, normalization, False, indices)

    assert len(preloaded) == len(lazy) == len(indices)
    for i in range(len(preloaded)):
        iq_p, cls_p, snr_p = preloaded[i]
        iq_l, cls_l, snr_l = lazy[i]
        assert iq_p.shape == (2, FRAME_LEN)
        assert torch.allclose(iq_p, iq_l, atol=1e-6), f"frame {i} differs ({normalization})"
        assert (cls_p, snr_p) == (cls_l, snr_l)

def test_unit_power_actually_yields_unit_power(synthetic_h5):
    ds = _make(synthetic_h5, "unit_power", True, np.arange(1, N_FRAMES))  # без нульового кадру
    p = ds.x.pow(2).sum(dim=1).mean(dim=1)
    assert torch.allclose(p, torch.ones_like(p), atol=1e-5)
    
def test_preload_reads_the_requested_rows(synthetic_h5):
    """Guards the block-read indexing in _load_all against an off-by-one / block-boundary bug."""
    indices = np.asarray([0, 1, 7, 19, N_FRAMES - 1])
    ds = _make(synthetic_h5, "none", True, indices)
    ds.x = ds._load_all(block=2)

    with h5py.File(synthetic_h5, "r") as f:
        raw = f[KEY_X][indices]                      # (n, T, 2), as stored
    expected = torch.from_numpy(raw).permute(0, 2, 1)  # channels-first: (n, 2, T)

    assert torch.equal(ds.x, expected.contiguous())


def test_every_normalizer_has_a_batched_twin():
    """preload=True indexes BATCHED_NORMALIZERS directly, so a missing twin is a KeyError."""
    assert set(BATCHED_NORMALIZERS) == set(NORMALIZERS)
