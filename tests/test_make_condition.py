"""Tests for the condition-generation driver, on a tiny synthetic HDF5 with the real layout.
The identity round-trip is primary: it covers order, dtype, normalization and the helpers."""

import h5py
import numpy as np
import pytest
import yaml

from scripts.make_condition import (
    _BATCH,
    load_conditions,
    load_sample_rate_hz,
    make_condition,
    verify_output,
)
from src.data import KEY_X, KEY_Y, KEY_Z
from src.distortions import apply_to_frame, build_compose, frame_rng, sigma_w_from_phase_noise

N_FRAMES = 40
FRAME_LEN = 64
N_CLASSES = 4
SNR_VALUES = (-10, 0, 10, 20)

# Every operator active, so the driver is exercised on a non-trivial chain. The reference is a
# chain-level key: one full-scale level per frame, shared by dc_offset and quantize alike.
_ALL_OPERATORS = [
    {"name": "phase_noise", "kwargs": {"sigma_w": 0.01}},
    {"name": "iq_imbalance", "kwargs": {"gain_db": 0.4, "phase_deg": 3.0}},
    {"name": "dc_offset", "kwargs": {"offset_i": 0.01, "offset_q": -0.005}},
    {"name": "quantize", "kwargs": {"n_bits": 8}},
]
_ALL_SPECS = {"reference": {"name": "peak"}, "operators": _ALL_OPERATORS}

# f_s is a file-level key, not an operator kwarg: it converts the datasheet-spelled condition
# below, and is recorded in every output so the assumption travels with the data.
_SAMPLE_RATE_HZ = 1.024e6
_DATASHEET_OPERATOR = {"name": "phase_noise",
                       "kwargs": {"phase_noise_dbc_hz": -98.0, "offset_hz": 1.0e4}}


@pytest.fixture(scope="module")
def source_h5(tmp_path_factory):
    """A small HDF5 with RadioML's keys, shapes and dtypes, plus the subset attributes."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_FRAMES, FRAME_LEN, 2)).astype(np.float32)
    x[0] = 0.0  # silent frame: exercises the reference-level divide-by-zero guard

    class_idx = np.arange(N_FRAMES) % N_CLASSES
    y = np.zeros((N_FRAMES, N_CLASSES), dtype=np.int64)
    y[np.arange(N_FRAMES), class_idx] = 1
    z = np.asarray([SNR_VALUES[i % len(SNR_VALUES)] for i in range(N_FRAMES)], dtype=np.int64)

    path = tmp_path_factory.mktemp("data") / "source.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset(KEY_X, data=x)
        f.create_dataset(KEY_Y, data=y)
        f.create_dataset(KEY_Z, data=z.reshape(-1, 1))
        f.attrs["frames_per_pair"] = 10
        f.attrs["subset_seed"] = 1234
        f.attrs["split_seed"] = 5678
        f.attrs["snr_min"] = -10
        f.attrs["snr_max"] = 20
    return path


@pytest.fixture(scope="module")
def conditions_file(tmp_path_factory):
    """A conditions YAML covering the identity case and every operator."""
    path = tmp_path_factory.mktemp("configs") / "conditions.yaml"
    table = {"sample_rate_hz": _SAMPLE_RATE_HZ, "conditions": {
        "baseline": [],
        "all": _ALL_SPECS,
        "phase_noise": [_ALL_OPERATORS[0]],
        "phase_noise_datasheet": [_DATASHEET_OPERATOR],
        "quantization": {"reference": {"name": "peak"}, "operators": [_ALL_OPERATORS[3]]},
    }}
    path.write_text(yaml.safe_dump(table), encoding="utf-8")
    return path


def _generate(source, conditions_file, condition, tmp_path, name=None, **kwargs):
    out = tmp_path / (name or f"{condition}.hdf5")
    return make_condition(condition, path=source, conditions=conditions_file,
                          out=out, verbose=False, **kwargs)


def _read_x(path):
    with h5py.File(path, "r") as f:
        return f[KEY_X][:]


# Identity round-trip -- the primary correctness test

def test_baseline_output_is_bit_identical_to_the_source(source_h5, conditions_file, tmp_path):
    out = _generate(source_h5, conditions_file, "baseline", tmp_path)
    with h5py.File(source_h5, "r") as fin, h5py.File(out, "r") as fout:
        assert fout[KEY_X].dtype == fin[KEY_X].dtype
        assert fout[KEY_X].shape == fin[KEY_X].shape
        # Bit-for-bit, not approximately: no stray normalization, no lossy round-trip.
        assert np.array_equal(fout[KEY_X][:], fin[KEY_X][:])
        assert np.array_equal(fout[KEY_Y][:], fin[KEY_Y][:])
        assert np.array_equal(fout[KEY_Z][:], fin[KEY_Z][:])
        assert fout.attrs["theta_is_identity"]


def test_frame_order_is_preserved_one_to_one(source_h5, conditions_file, tmp_path):
    """Labels stay glued to their frames, so the fixed split indices remain valid."""
    out = _generate(source_h5, conditions_file, "all", tmp_path)
    src, dst = _read_x(source_h5), _read_x(out)
    with h5py.File(source_h5, "r") as fin, h5py.File(out, "r") as fout:
        assert np.array_equal(fout[KEY_Y][:], fin[KEY_Y][:])
        assert np.array_equal(fout[KEY_Z][:], fin[KEY_Z][:])
    # Each output frame must be nearest to its OWN source frame, not to some other row.
    for i in (1, 7, N_FRAMES - 1):
        distances = np.linalg.norm(src.reshape(N_FRAMES, -1) - dst[i].ravel(), axis=1)
        assert int(distances.argmin()) == i


def test_output_preserves_shape_dtype_and_finiteness(source_h5, conditions_file, tmp_path):
    out = _generate(source_h5, conditions_file, "all", tmp_path)
    x = _read_x(out)
    assert x.shape == (N_FRAMES, FRAME_LEN, 2)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


# Reproducibility

def test_separate_runs_produce_identical_bytes(source_h5, conditions_file, tmp_path):
    a = _generate(source_h5, conditions_file, "all", tmp_path, name="a.hdf5")
    b = _generate(source_h5, conditions_file, "all", tmp_path, name="b.hdf5")
    assert np.array_equal(_read_x(a), _read_x(b))
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert fa.attrs["content_checksum"] == fb.attrs["content_checksum"]


def test_batch_size_does_not_change_the_output(source_h5, conditions_file, tmp_path, monkeypatch):
    """Iteration order is not part of the key, so re-batching must change nothing."""
    reference = _read_x(_generate(source_h5, conditions_file, "all", tmp_path, name="ref.hdf5"))
    for batch in (1, 3, N_FRAMES * 2):
        monkeypatch.setattr("scripts.make_condition._BATCH", batch)
        out = _generate(source_h5, conditions_file, "all", tmp_path, name=f"b{batch}.hdf5")
        assert np.array_equal(_read_x(out), reference)


def test_partial_rerun_resuming_mid_file_reproduces_the_rows(source_h5, conditions_file, tmp_path):
    """A run cut short and resumed must land byte-identical on the rows it redoes.
    The driver writes whole files, so this locks the keying invariant a resume would rely on."""
    full = _read_x(_generate(source_h5, conditions_file, "all", tmp_path, name="full.hdf5"))
    compose = build_compose(_ALL_SPECS)
    src = _read_x(source_h5)

    # Resume from an arbitrary cut, walking backwards -- an order no full run ever uses.
    cut = 13
    for i in reversed(range(cut, N_FRAMES)):
        assert np.array_equal(apply_to_frame(compose, src[i], "all", i), full[i])
    # And the rows before the cut are untouched by having been generated in a separate pass.
    for i in range(cut):
        assert np.array_equal(apply_to_frame(compose, src[i], "all", i), full[i])


def test_interleaved_generation_order_reproduces_the_file(source_h5, conditions_file, tmp_path):
    """Even a shuffled visit order reproduces every row: the key is the index, not the position."""
    full = _read_x(_generate(source_h5, conditions_file, "all", tmp_path, name="shuf.hdf5"))
    compose = build_compose(_ALL_SPECS)
    src = _read_x(source_h5)
    order = np.random.default_rng(7).permutation(N_FRAMES)
    for i in order:
        assert np.array_equal(apply_to_frame(compose, src[i], "all", int(i)), full[i])


def test_different_conditions_key_different_noise(source_h5, conditions_file, tmp_path):
    a = _read_x(_generate(source_h5, conditions_file, "phase_noise", tmp_path))
    b = _read_x(_generate(source_h5, conditions_file, "all", tmp_path, name="all2.hdf5"))
    assert not np.allclose(a, b)


def test_frame_rng_is_keyed_not_sequential():
    assert np.array_equal(frame_rng("c", 5).normal(size=8), frame_rng("c", 5).normal(size=8))
    assert not np.array_equal(frame_rng("c", 5).normal(size=8), frame_rng("c", 6).normal(size=8))
    assert not np.array_equal(frame_rng("a", 5).normal(size=8), frame_rng("b", 5).normal(size=8))


# Per-frame independence

def test_frame_generated_alone_matches_the_batched_result(source_h5, conditions_file, tmp_path):
    """Phase-noise state must not carry across frames -- RadioML frames are not contiguous."""
    out = _read_x(_generate(source_h5, conditions_file, "all", tmp_path))
    compose = build_compose(_ALL_SPECS)
    src = _read_x(source_h5)
    for i in (0, 1, 17, N_FRAMES - 1):
        alone = apply_to_frame(compose, src[i], "all", i)
        assert np.array_equal(alone, out[i])


def test_phase_ramp_restarts_every_frame(source_h5, conditions_file, tmp_path):
    """A Wiener process running over the concatenated dataset would be physically meaningless."""
    out = _read_x(_generate(source_h5, conditions_file, "phase_noise", tmp_path))
    src = _read_x(source_h5)
    # Frame 0 is silent, so start at 1; phi[0] == 0 means sample 0 is untouched.
    for i in range(1, 6):
        assert np.allclose(out[i][0], src[i][0], atol=1e-6)


# Metadata

def test_metadata_records_theta_condition_scheme_and_checksum(source_h5, conditions_file, tmp_path):
    import json

    out = _generate(source_h5, conditions_file, "all", tmp_path)
    with h5py.File(out, "r") as f:
        assert f.attrs["condition"] == "all"
        theta = json.loads(f.attrs["theta"])
        assert theta["PhaseNoise"]["sigma_w"] == 0.01
        assert theta["IQImbalance"] == {"gain_db": 0.4, "phase_deg": 3.0}
        assert theta["Quantize"]["n_bits"] == 8
        # One chain-level reference, with the point it was measured at -- not one per operator.
        assert theta["_reference"]["kind"] == "peak"
        assert theta["_reference"]["measured_after"] == "IQImbalance"
        assert "reference" not in theta["Quantize"]
        assert "frame_index" in f.attrs["rng_scheme"]
        # The rate theta's per-Hz figures were converted against, alongside theta itself.
        assert f.attrs["sample_rate_hz"] == _SAMPLE_RATE_HZ
        assert f.attrs["content_checksum"].startswith("blake2b16:")
        assert f.attrs["normalization_applied"] == "none"
        # Subset provenance is inherited, so the output is a drop-in for data.path.
        assert f.attrs["subset_seed"] == 1234
        assert f.attrs["split_seed"] == 5678


def test_checksum_matches_the_written_content(source_h5, conditions_file, tmp_path):
    import hashlib

    out = _generate(source_h5, conditions_file, "all", tmp_path)
    with h5py.File(out, "r") as f:
        stored = f.attrs["content_checksum"]
        x = f[KEY_X][:]
    digest = hashlib.blake2b(digest_size=16)
    for start in range(0, len(x), _BATCH):
        digest.update(np.ascontiguousarray(x[start:start + _BATCH]).tobytes())
    assert stored == f"blake2b16:{digest.hexdigest()}"


# Crash safety -- a partial write must never be mistaken for a finished dataset

def _crash_at(monkeypatch, frame_index):
    """Make generation die partway, the way a Ctrl-C or an OOM kill would."""
    import scripts.make_condition as module

    real = module.apply_to_frame

    def exploding(compose, frame, condition, index):
        if index == frame_index:
            raise KeyboardInterrupt("simulated interruption")
        return real(compose, frame, condition, index)

    monkeypatch.setattr(module, "apply_to_frame", exploding)


def test_crash_midway_leaves_no_output_file(source_h5, conditions_file, tmp_path, monkeypatch):
    """The output is published by rename, so an interrupted run produces no target at all."""
    _crash_at(monkeypatch, N_FRAMES // 2)
    out = tmp_path / "crash.hdf5"
    with pytest.raises(KeyboardInterrupt):
        make_condition("all", path=source_h5, conditions=conditions_file, out=out, verbose=False)
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_rerun_after_a_crash_regenerates_completely(source_h5, conditions_file, tmp_path, monkeypatch):
    """The exact scenario: crash, restart, and the frozen dataset must still be correct."""
    reference = _read_x(_generate(source_h5, conditions_file, "all", tmp_path, name="good.hdf5"))

    out = tmp_path / "resumed.hdf5"
    _crash_at(monkeypatch, N_FRAMES // 2)
    with pytest.raises(KeyboardInterrupt):
        make_condition("all", path=source_h5, conditions=conditions_file, out=out, verbose=False)

    monkeypatch.undo()
    make_condition("all", path=source_h5, conditions=conditions_file, out=out, verbose=False)
    assert np.array_equal(_read_x(out), reference)


def test_successful_run_leaves_no_partial_file(source_h5, conditions_file, tmp_path):
    out = _generate(source_h5, conditions_file, "all", tmp_path, name="clean.hdf5")
    assert not out.with_name(out.name + ".partial").exists()


def test_truncated_legacy_file_is_refused_not_silently_reused(source_h5, conditions_file, tmp_path):
    """A file left by a pre-atomic run has the right shape but no attrs; it must not be trusted."""
    out = tmp_path / "legacy.hdf5"
    with h5py.File(source_h5, "r") as fin, h5py.File(out, "w") as fout:
        fout.create_dataset(KEY_X, shape=fin[KEY_X].shape, dtype=fin[KEY_X].dtype)  # all zeros
        fout.create_dataset(KEY_Y, data=fin[KEY_Y][:])
        fout.create_dataset(KEY_Z, data=fin[KEY_Z][:])

    ok, reason = verify_output(out)
    assert not ok and "did not complete" in reason
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        make_condition("all", path=source_h5, conditions=conditions_file, out=out, verbose=False)


def test_verify_output_detects_corrupted_content(source_h5, conditions_file, tmp_path):
    """Structure alone is not enough: flipped bytes inside X must fail the checksum."""
    out = _generate(source_h5, conditions_file, "all", tmp_path, name="corrupt.hdf5")
    assert verify_output(out)[0]

    with h5py.File(out, "r+") as f:
        f[KEY_X][N_FRAMES // 3, 0, 0] += np.float32(1.0)
    ok, reason = verify_output(out)
    assert not ok and "checksum mismatch" in reason
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        make_condition("all", path=source_h5, conditions=conditions_file, out=out, verbose=False)


def test_verification_can_be_disabled(source_h5, conditions_file, tmp_path):
    """Escape hatch for very large files, where re-reading to verify is not free."""
    out = _generate(source_h5, conditions_file, "all", tmp_path, name="noverify.hdf5")
    with h5py.File(out, "r+") as f:
        f[KEY_X][0, 0, 0] += np.float32(1.0)
    make_condition("all", path=source_h5, conditions=conditions_file, out=out,
                   verify=False, verbose=False)


# Driver guards

def test_existing_output_is_left_alone_without_overwrite(source_h5, conditions_file, tmp_path):
    out = _generate(source_h5, conditions_file, "baseline", tmp_path, name="idem.hdf5")
    before = out.stat().st_mtime_ns
    _generate(source_h5, conditions_file, "all", tmp_path, name="idem.hdf5")
    assert out.stat().st_mtime_ns == before  # not rewritten with the other condition
    _generate(source_h5, conditions_file, "all", tmp_path, name="idem.hdf5", overwrite=True)
    with h5py.File(out, "r") as f:
        assert f.attrs["condition"] == "all"


def test_unknown_condition_fails_loudly(source_h5, conditions_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown condition"):
        _generate(source_h5, conditions_file, "does_not_exist", tmp_path)


def test_refuses_to_overwrite_the_source(source_h5, conditions_file):
    with pytest.raises(ValueError, match="must differ"):
        make_condition("baseline", path=source_h5, conditions=conditions_file,
                       out=source_h5, verbose=False)


def test_metadata_records_the_datasheet_figures_theta_was_derived_from(
        source_h5, conditions_file, tmp_path):
    """A file generated from a per-Hz figure must carry the figure, not only its consequence."""
    import json

    out = _generate(source_h5, conditions_file, "phase_noise_datasheet", tmp_path)
    with h5py.File(out, "r") as f:
        theta = json.loads(f.attrs["theta"])["PhaseNoise"]
        assert f.attrs["sample_rate_hz"] == _SAMPLE_RATE_HZ
        assert theta["phase_noise_dbc_hz"] == -98.0
        assert theta["offset_hz"] == 1.0e4
        assert theta["sigma_w"] == pytest.approx(
            sigma_w_from_phase_noise(-98.0, 1.0e4, _SAMPLE_RATE_HZ))


def test_repo_conditions_file_builds():
    """The checked-in configs/conditions.yaml must define the study's conditions and build."""
    table = load_conditions()
    sample_rate_hz = load_sample_rate_hz()
    assert {"baseline", "phase_noise", "iq_imbalance", "quantization", "all"} <= set(table)
    # A number, not the string YAML 1.1 makes of an unsigned exponent.
    assert isinstance(sample_rate_hz, float) and sample_rate_hz > 0
    assert build_compose(table["baseline"], sample_rate_hz=sample_rate_hz).is_identity
    for name, specs in table.items():
        build_compose(specs, sample_rate_hz=sample_rate_hz).params()

    # The study's phase noise comes from the datasheet pair, and every chain that carries the
    # oscillator agrees on it -- a second spelling of the same part could silently drift.
    derived = build_compose(table["phase_noise"], sample_rate_hz=sample_rate_hz).params()
    assert derived["PhaseNoise"]["sigma_w"] == pytest.approx(7.8e-4, rel=0.01)
    assert build_compose(table["all"], sample_rate_hz=sample_rate_hz).params()["PhaseNoise"] == (
        derived["PhaseNoise"])
