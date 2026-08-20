"""Tests for the paired condition-vs-baseline comparison, on a synthetic runs/ tree.
The guards are primary: they are what stops the delta measuring something it does not claim."""

import copy
import json

import h5py
import numpy as np
import pytest
import yaml

from scripts.compare_condition import (
    GuardFailure,
    apply_rule,
    balanced_accuracy,
    compare_condition,
    confusion,
    content_checksum,
    per_class_recall,
    top_confusion_pairs,
)
from src.data import KEY_X, MODULATION_CLASSES

N_CLASSES = len(MODULATION_CLASSES)
SNRS = (-10, -4, 0, 10)
PER_CELL = 5
N_FRAMES = 40

_CONFIG_TEMPLATE = {
    "data": {"path": "", "frames_per_pair": 8, "subset_seed": 1234, "snr_min": -10,
             "snr_max": 10, "split": [0.7, 0.15, 0.15], "split_seed": 5678,
             "normalization": "unit_power", "preload": False},
    "model": {"dropout_p": 0.4, "init_scheme": "kaiming_linear"},
    "train": {"seeds": [100], "batch_size": 8, "num_workers": 0,
              "optimizer": {"name": "adam", "kwargs": {}}, "learning_rate": 0.001,
              "weight_decay": 0.0, "lr_scheduler": {"name": "none", "kwargs": {}},
              "max_epochs": 1, "early_stopping_enabled": False, "early_stopping_patience": 1,
              "early_stopping_metric": "val_accuracy_snr_geq_0db", "amp_enabled": False},
    "experiment": {"project": "test", "condition": "", "mode": "disabled"},
}


# Building a synthetic pair of sides

def _write_dataset(path, fill=0.0, condition=None, theta="{}", checksum=None,
                   subset_seed=1234, split_seed=5678, n_frames=N_FRAMES):
    """A dataset carrying only what the guards read. `fill` is what makes the content differ."""
    with h5py.File(path, "w") as f:
        f.create_dataset(KEY_X, data=np.full((n_frames, 4, 2), fill, dtype=np.float32))
        f.attrs["subset_seed"] = subset_seed
        f.attrs["split_seed"] = split_seed
        f.attrs["split"] = [0.7, 0.15, 0.15]
        f.attrs["n_frames"] = n_frames
        if condition is not None:
            f.attrs["condition"] = condition
            f.attrs["theta"] = theta
        if checksum is not None:
            f.attrs["content_checksum"] = checksum
    return path


def _write_config(path, label, data_path):
    raw = copy.deepcopy(_CONFIG_TEMPLATE)
    raw["data"]["path"] = str(data_path)
    raw["experiment"]["condition"] = label
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _labels():
    """(true, snr) for a stratified test split: every class at every SNR."""
    true = np.repeat(np.arange(N_CLASSES), len(SNRS) * PER_CELL).astype(np.int16)
    snr = np.tile(np.repeat(np.array(SNRS), PER_CELL), N_CLASSES).astype(np.int16)
    return true, snr


def _collapse(true, n_collapsed):
    """Predictions that are perfect except for the first `n_collapsed` classes, which are
    predicted as the next class entirely -- an exactly known recall loss of 1.0 each."""
    pred = true.copy()
    for c in range(n_collapsed):
        pred[true == c] = (c + 1) % N_CLASSES
    return pred


def _write_runs(root, label, per_seed, checkpoint=None):
    """per_seed maps seed -> (pred, true, snr); mirrors what src.predict writes."""
    for seed, (pred, true, snr) in per_seed.items():
        cell = root / label / f"seed{seed}"
        cell.mkdir(parents=True)
        (cell / "meta.json").write_text(json.dumps(
            {"run_id": f"{label}-{seed}", "config": {"condition": label, "seed": seed}}))
        arrays = {"pred": pred, "true": true, "snr": snr, "run_id": f"{label}-{seed}"}
        if checkpoint is not None:
            arrays["checkpoint"] = checkpoint
        np.savez_compressed(cell / "predictions.npz", **arrays)


def _build(tmp_path, baseline_seeds=(100, 101), condition_collapse=None,
           baseline_dataset=None, condition_dataset=None, condition_rows=None,
           baseline_checkpoint=None, condition_checkpoint=None):
    """A full two-sided workspace; returns (baseline_config, condition_config, runs_root)."""
    true, snr = _labels()
    condition_collapse = condition_collapse or {seed: 1 for seed in baseline_seeds}

    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    base_path = baseline_dataset or _write_dataset(data / "clean.hdf5", fill=0.0)
    cond_path = condition_dataset or _write_dataset(
        data / "impaired.hdf5", fill=1.0, condition="phase_noise_exaggerated",
        theta='{"PhaseNoise": {"sigma_w": 0.01}}')

    runs = tmp_path / "runs"
    _write_runs(runs, "base", {s: (true.copy(), true, snr) for s in baseline_seeds},
                baseline_checkpoint)
    _write_runs(runs, "cond",
                {s: (_collapse(true, n), *(condition_rows or (true, snr)))
                 for s, n in condition_collapse.items()},
                condition_checkpoint)

    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    return (_write_config(configs / "base.yaml", "base", base_path),
            _write_config(configs / "cond.yaml", "cond", cond_path),
            runs)


def _compare(tmp_path, **kwargs):
    unpaired = kwargs.pop("unpaired", False)
    base_cfg, cond_cfg, runs = _build(tmp_path, **kwargs)
    return compare_condition(base_cfg, cond_cfg, runs_root=runs, unpaired=unpaired,
                             plot=False, out_dir=None, verbose=False)


# Pairing -- the estimator itself

def test_paired_delta_is_the_mean_of_the_per_seed_differences(tmp_path):
    """One collapsed class costs exactly 1/24 of balanced accuracy, so the mean is exact."""
    report = _compare(tmp_path, condition_collapse={100: 1, 101: 3})
    assert report["paired"] and report["n"] == 2
    assert report["delta"]["balanced_accuracy"] == pytest.approx(-2.0 / N_CLASSES)
    assert report["delta"]["balanced_accuracy_high"] == pytest.approx(-2.0 / N_CLASSES)
    assert report["spread"]["balanced_accuracy"] == pytest.approx(np.std([1, 3], ddof=1) / N_CLASSES)


def test_a_single_common_seed_reports_no_spread(tmp_path):
    """n=1 is one observation; formatting it like an estimate would be the error."""
    report = _compare(tmp_path, baseline_seeds=(100,))
    assert report["paired"] and report["n"] == 1
    assert report["spread"] is None
    assert report["delta"]["balanced_accuracy_high"] == pytest.approx(-1.0 / N_CLASSES)


def test_no_common_seed_fails_and_lists_both_sides(tmp_path):
    with pytest.raises(ValueError, match="no seed is present on both sides") as excinfo:
        _compare(tmp_path, baseline_seeds=(100, 101), condition_collapse={200: 1})
    message = str(excinfo.value)
    assert "[100, 101]" in message and "[200]" in message


def test_the_unpaired_estimator_is_opt_in(tmp_path):
    report = _compare(tmp_path, baseline_seeds=(100, 101), condition_collapse={200: 1},
                      unpaired=True)
    assert not report["paired"] and report["n"] == 0
    assert report["delta"]["balanced_accuracy_high"] == pytest.approx(-1.0 / N_CLASSES)


def test_a_seed_on_one_side_only_is_excluded_and_reported(tmp_path):
    """Seed 101 exists on the baseline alone: it must leave the estimate AND be named."""
    report = _compare(tmp_path, baseline_seeds=(100, 101), condition_collapse={100: 1, 102: 5})
    assert report["seeds"] == {"paired": [100], "only_baseline": [101], "only_condition": [102]}
    assert report["n"] == 1
    # The seed-102 collapse of 5 classes must not have leaked into the mean.
    assert report["delta"]["balanced_accuracy_high"] == pytest.approx(-1.0 / N_CLASSES)


# Guards

def test_identical_datasets_fail_the_checksum_guard(tmp_path):
    """The failure this exists for: one config error points both runs at the same file."""
    data = tmp_path / "data"
    data.mkdir()
    same = _write_dataset(data / "same.hdf5", fill=0.0, condition="phase_noise_exaggerated")
    with pytest.raises(GuardFailure, match="datasets differ"):
        _compare(tmp_path, baseline_dataset=same, condition_dataset=same)


def test_a_different_split_seed_fails(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    other = _write_dataset(data / "impaired.hdf5", fill=1.0, split_seed=999,
                           condition="phase_noise_exaggerated")
    with pytest.raises(GuardFailure, match="split_seed shared"):
        _compare(tmp_path, condition_dataset=other)


def test_a_different_frame_count_fails(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    other = _write_dataset(data / "impaired.hdf5", fill=1.0, n_frames=N_FRAMES // 2,
                           condition="phase_noise_exaggerated")
    with pytest.raises(GuardFailure, match="frame count shared"):
        _compare(tmp_path, condition_dataset=other)


def test_misaligned_test_rows_fail(tmp_path):
    """Attrs can agree while the scored frames do not; the stored vectors settle it."""
    true, snr = _labels()
    order = np.random.default_rng(0).permutation(true.size)
    with pytest.raises(GuardFailure, match="test rows aligned"):
        _compare(tmp_path, condition_rows=(true[order], snr[order]))


def test_different_checkpoints_on_the_two_sides_fail(tmp_path):
    """best-val and final-epoch are different quantities; their difference is not the cost."""
    with pytest.raises(GuardFailure, match="same checkpoint"):
        _compare(tmp_path, baseline_checkpoint="best", condition_checkpoint="last")


def test_the_report_states_what_it_actually_compared(tmp_path):
    report = _compare(tmp_path)
    assert report["condition"].dataset.condition == "phase_noise_exaggerated"
    assert "sigma_w" in report["condition"].dataset.theta
    # The clean subset declares neither, and the report must say so rather than invent one.
    assert "none declared" in report["baseline"].dataset.condition
    assert all(ok for _, ok, _ in report["guards"])


def test_the_clean_subsets_checksum_is_computed_not_skipped(tmp_path):
    """make_subset writes no checksum, so the guard would otherwise be blind on that side."""
    path = _write_dataset(tmp_path / "clean.hdf5", fill=0.5)
    checksum, computed = content_checksum(path)
    assert computed and checksum.startswith("blake2b16:")
    assert content_checksum(_write_dataset(tmp_path / "other.hdf5", fill=0.75))[0] != checksum


def test_the_computed_checksum_matches_the_one_make_condition_writes(tmp_path):
    """The digest is spelled out twice, so pin the two spellings to the same answer."""
    from scripts.make_condition import make_condition
    from src.data import KEY_Y, KEY_Z

    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as f:
        f.create_dataset(KEY_X, data=np.random.default_rng(0).standard_normal(
            (N_FRAMES, 16, 2)).astype(np.float32))
        f.create_dataset(KEY_Y, data=np.eye(4, dtype=np.int64)[np.arange(N_FRAMES) % 4])
        f.create_dataset(KEY_Z, data=np.zeros((N_FRAMES, 1), dtype=np.int64))

    conditions = tmp_path / "conditions.yaml"
    conditions.write_text(yaml.safe_dump(
        {"conditions": {"phase_noise": [{"name": "phase_noise", "kwargs": {"sigma_w": 0.01}}]}}),
        encoding="utf-8")

    out = make_condition("phase_noise", path=source, conditions=conditions,
                         out=tmp_path / "out.hdf5", verbose=False)
    with h5py.File(out, "r") as f:
        stored = str(f.attrs["content_checksum"])
    assert content_checksum(out) == (stored, False)

    with h5py.File(out, "r+") as f:
        del f.attrs["content_checksum"]
    assert content_checksum(out) == (stored, True)


# Metrics

def test_balanced_accuracy_does_not_let_the_majority_hide_a_collapsed_class():
    true = np.array([0] * 90 + [1] * 10)
    pred = np.array([0] * 90 + [0] * 10)   # class 1 gone entirely
    assert (pred == true).mean() == pytest.approx(0.9)
    assert balanced_accuracy(pred, true, n_classes=2) == pytest.approx(0.5)


def test_per_class_recall_is_nan_for_a_class_with_no_frames():
    recall = per_class_recall(np.array([0, 0]), np.array([0, 0]), n_classes=3)
    assert recall[0] == pytest.approx(1.0)
    assert np.isnan(recall[1:]).all()


def test_confusion_rows_are_conditional_probabilities():
    true = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 1])
    matrix = confusion(pred, true, n_classes=3)
    assert matrix[0].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert matrix[1].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert np.isnan(matrix[2]).all()


def test_top_confusion_pairs_rank_growth_and_skip_the_diagonal():
    diff = np.zeros((3, 3))
    diff[0, 0] = 0.9      # a recall change, reported elsewhere -- must not appear here
    diff[0, 1] = 0.4
    diff[2, 1] = 0.2
    names = ("A", "B", "C")
    assert top_confusion_pairs(diff, n=2, class_names=names) == [
        ("A", "B", pytest.approx(0.4)), ("C", "B", pytest.approx(0.2))]


# Decision rules -- fixed before the numbers

def test_the_control_rule_demands_a_cost_of_the_right_sign():
    assert apply_rule("phase_noise_exaggerated", -2.5)[0] == "PASS"
    assert apply_rule("phase_noise_exaggerated", -1.0)[0] == "FAIL"
    verdict, explanation = apply_rule("phase_noise_exaggerated", +3.0)
    assert verdict == "FAIL" and "WRONG SIGN" in explanation


def test_the_datasheet_phase_noise_rule_is_a_declared_null():
    assert apply_rule("phase_noise", -0.4)[0] == "PASS"
    assert apply_rule("phase_noise", +0.9)[0] == "PASS"
    assert apply_rule("phase_noise", -1.6)[0] == "FAIL"


def test_an_unregistered_condition_gets_no_verdict():
    verdict, explanation = apply_rule("iq_imbalance", -12.0)
    assert verdict == "NO RULE" and "_DECISION_RULES" in explanation


def test_the_control_verdict_travels_with_the_report(tmp_path):
    """One collapsed class is -4.17 pp, comfortably past the 2.0 pp bar."""
    report = _compare(tmp_path, baseline_seeds=(100,))
    assert report["verdict"] == "PASS" and report["passed"]
