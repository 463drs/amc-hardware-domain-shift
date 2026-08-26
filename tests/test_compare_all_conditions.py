"""Tests for the five-condition comparison, on a synthetic runs/ tree.

Every prediction here is perfect except for a chosen set of collapsed classes, so each balanced
accuracy is exactly (24 - collapsed) / 24 and the medians, means and IQRs below are arithmetic.
"""

from __future__ import annotations

import copy
import json

import h5py
import numpy as np
import pytest
import yaml

from scripts.compare_all_conditions import (
    GuardFailure,
    check_guards,
    cross_domain_matrix,
    degenerate_counts,
    load_matrix,
    off_diagonal_cost,
    paired_deltas,
    psk_recall,
    summary_table,
)
from src.data import KEY_X, MODULATION_CLASSES
from src.predict import predictions_name

N_CLASSES = len(MODULATION_CLASSES)
SNRS = (-10, -4, 0, 10)
PER_CELL = 10
N_FRAMES = 40
LABELS = ("base", "cond_a", "cond_b")
SEEDS = (100, 101, 102, 103)

PSK = tuple(MODULATION_CLASSES.index(name) for name in ("8PSK", "16PSK", "32PSK"))
QAM = tuple(MODULATION_CLASSES.index(name) for name in ("16QAM", "32QAM", "64QAM"))

_CONFIG_TEMPLATE = {
    "data": {"path": "", "frames_per_pair": 8, "subset_seed": 1234, "snr_min": -10,
             "snr_max": 10, "split": [0.7, 0.15, 0.15], "split_seed": 5678,
             "normalization": "unit_power", "preload": False},
    "model": {"dropout_p": 0.4, "init_scheme": "kaiming_linear"},
    "train": {"seeds": list(SEEDS), "batch_size": 8, "num_workers": 0,
              "optimizer": {"name": "adam", "kwargs": {}}, "learning_rate": 0.001,
              "weight_decay": 0.0, "lr_scheduler": {"name": "none", "kwargs": {}},
              "max_epochs": 1, "early_stopping_enabled": False, "early_stopping_patience": 1,
              "early_stopping_metric": "val_accuracy_snr_geq_0db", "amp_enabled": False},
    "experiment": {"project": "test", "condition": "", "mode": "disabled"},
}


# Building a synthetic workspace

def _labels():
    """(true, snr) for a stratified test split: every class at every SNR."""
    true = np.repeat(np.arange(N_CLASSES), len(SNRS) * PER_CELL).astype(np.int16)
    snr = np.tile(np.repeat(np.array(SNRS), PER_CELL), N_CLASSES).astype(np.int16)
    return true, snr


def _collapse(true, classes):
    """Perfect except `classes`, each predicted as the next class entirely -- recall exactly 0."""
    pred = true.copy()
    for c in classes:
        pred[true == c] = (c + 1) % N_CLASSES
    return pred


def _write_dataset(path, fill, **attrs):
    """A dataset carrying only what the guards read. `fill` is what makes the content differ."""
    n_frames = attrs.get("n_frames", N_FRAMES)
    with h5py.File(path, "w") as f:
        f.create_dataset(KEY_X, data=np.full((n_frames, 4, 2), fill, dtype=np.float32))
        f.attrs["subset_seed"] = attrs.get("subset_seed", 1234)
        f.attrs["split_seed"] = attrs.get("split_seed", 5678)
        f.attrs["split"] = [0.7, 0.15, 0.15]
        f.attrs["n_frames"] = n_frames
        f.attrs["condition"] = attrs.get("condition", "synthetic")
        f.attrs["theta"] = attrs.get("theta", "{}")
    return path


def _write_config(path, label, data_path):
    raw = copy.deepcopy(_CONFIG_TEMPLATE)
    raw["data"]["path"] = str(data_path)
    raw["experiment"]["condition"] = label
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _build(tmp_path, collapsed=None, dataset_attrs=None, shared_dataset=False,
           rows_for=None, skip=()):
    """A full three-condition workspace. `collapsed(train, evaluate, seed) -> class indices`
    decides what each cell gets wrong; returns (config paths, runs root)."""
    collapsed = collapsed or (lambda train, evaluate, seed: ())
    dataset_attrs = dataset_attrs or {}
    true, snr = _labels()

    data, configs, runs = tmp_path / "data", tmp_path / "configs", tmp_path / "runs"
    data.mkdir(exist_ok=True)
    configs.mkdir(exist_ok=True)

    paths = []
    for i, label in enumerate(LABELS):
        name = "shared.hdf5" if shared_dataset else f"{label}.hdf5"
        dataset = _write_dataset(data / name, 0.0 if shared_dataset else float(i),
                                 condition=label, **dataset_attrs.get(label, {}))
        paths.append(_write_config(configs / f"{label}.yaml", label, dataset))

    for train in LABELS:
        for seed in SEEDS:
            cell = runs / train / f"seed{seed}"
            cell.mkdir(parents=True, exist_ok=True)
            (cell / "meta.json").write_text(json.dumps(
                {"run_id": f"{train}-{seed}",
                 "config": {"condition": train, "seed": seed}}), encoding="utf-8")
            for evaluate in LABELS:
                if (train, evaluate) in skip:
                    continue
                rows = (rows_for or {}).get((train, evaluate), (true, snr))
                np.savez_compressed(
                    cell / predictions_name(train, evaluate),
                    pred=_collapse(rows[0], collapsed(train, evaluate, seed)),
                    true=rows[0], snr=rows[1], run_id=f"{train}-{seed}")

    return paths, runs


def _matrix(tmp_path, **kwargs):
    """The loaded matrix, guarded -- nothing downstream is allowed to see an unguarded one."""
    paths, runs = _build(tmp_path, **kwargs)
    matrix = load_matrix(paths, baseline=paths[0], runs_root=runs)
    check_guards(matrix, verbose=False)
    return matrix


# Part 1 -- the paired deltas

def test_the_baseline_row_is_exactly_zero(tmp_path):
    """It is the reference: pairing it against itself must leave nothing behind."""
    matrix = _matrix(tmp_path, collapsed=lambda t, e, s: (0,) if t == "cond_a" else ())
    deltas = paired_deltas(matrix)
    assert deltas["base"].tolist() == [0.0] * len(SEEDS)
    assert deltas["cond_a"] == pytest.approx([-1 / N_CLASSES] * len(SEEDS))


def test_the_delta_is_paired_by_seed_not_a_difference_of_means(tmp_path):
    """The baseline moves per seed too; only same-seed subtraction removes it."""
    def collapsed(train, evaluate, seed):
        early = seed < 102
        if train == "base":
            return (0,) if early else ()             # baseline: 23/24 then 24/24
        return (0, 1) if early else (2,)             # cond_a:   22/24 then 23/24
    deltas = paired_deltas(_matrix(tmp_path, collapsed=collapsed))
    assert deltas["cond_a"] == pytest.approx([-1 / N_CLASSES] * len(SEEDS))


def test_median_and_mean_diverge_when_one_seed_degenerates(tmp_path):
    """The point of showing both: one collapsed run drags the mean and leaves the median."""
    def collapsed(train, evaluate, seed):
        if train != "cond_a":
            return ()
        return tuple(range(12)) if seed == 103 else (0,)   # three healthy seeds, one degenerate
    row = summary_table(_matrix(tmp_path, collapsed=collapsed))["cond_a"]
    assert row["median"] == pytest.approx(-1 / N_CLASSES)
    assert row["mean"] == pytest.approx(-3.75 / N_CLASSES)
    assert row["iqr"] == pytest.approx(2.75 / N_CLASSES)


def test_iqr_is_zero_when_every_seed_agrees(tmp_path):
    matrix = _matrix(tmp_path, collapsed=lambda t, e, s: (0,) if t == "cond_b" else ())
    assert summary_table(matrix)["cond_b"]["iqr"] == pytest.approx(0.0)


# Degenerate cells -- PSK family only

def test_degenerate_cells_count_psk_only(tmp_path):
    """QAM recall is continuous, so a collapsed QAM class must not enter the count."""
    matrix = _matrix(tmp_path, collapsed=lambda t, e, s: QAM if t == "cond_a" else ())
    assert degenerate_counts(matrix)["cond_a"][0.25] == 0
    assert psk_recall(matrix, "cond_a").min() == pytest.approx(1.0)


def test_a_collapsed_psk_class_is_one_degenerate_cell_per_seed(tmp_path):
    matrix = _matrix(tmp_path, collapsed=lambda t, e, s: PSK[:2] if t == "cond_b" else ())
    counts = degenerate_counts(matrix)["cond_b"]
    assert counts == {t: 2 * len(SEEDS) for t in (0.20, 0.25, 0.30)}
    assert psk_recall(matrix, "cond_b").shape == (len(PSK), len(SEEDS))


def test_the_three_thresholds_bracket_a_partly_degraded_class(tmp_path):
    """A recall of exactly 0.25 sits between the thresholds, which is what makes printing all
    three worth doing: it is counted at 0.30 and at neither of the other two."""
    true, snr = _labels()
    pred = true.copy()
    high = np.nonzero((snr >= 0) & (true == PSK[0]))[0]     # 20 frames; 15 wrong -> recall 0.25
    pred[high[:15]] = (PSK[0] + 1) % N_CLASSES

    paths, runs = _build(tmp_path)
    for seed in SEEDS:
        np.savez_compressed(runs / "cond_a" / f"seed{seed}" / "predictions.npz",
                            pred=pred, true=true, snr=snr, run_id=f"cond_a-{seed}")
    matrix = load_matrix(paths, baseline=paths[0], runs_root=runs)
    assert psk_recall(matrix, "cond_a")[0].tolist() == pytest.approx([0.25] * len(SEEDS))
    assert degenerate_counts(matrix)["cond_a"] == {0.20: 0, 0.25: 0, 0.30: len(SEEDS)}


# Part 2 -- the cross-domain matrix

def test_the_diagonal_is_the_in_domain_number(tmp_path):
    """The diagonal reads the cached predictions.npz; the off-diagonal reads its own file."""
    matrix = _matrix(tmp_path, collapsed=lambda t, e, s: (0,) if t == e else (0, 1))
    values = cross_domain_matrix(matrix)
    assert np.diag(values) == pytest.approx([(N_CLASSES - 1) / N_CLASSES] * len(LABELS))
    assert values[~np.eye(len(LABELS), dtype=bool)] == pytest.approx((N_CLASSES - 2) / N_CLASSES)


def test_off_diagonal_cost_is_relative_to_the_rows_own_diagonal(tmp_path):
    """Row-wise, not column-wise: it is what THIS model loses by leaving THIS training domain."""
    def collapsed(train, evaluate, seed):
        if train == "cond_a":
            return () if evaluate == "cond_a" else (0, 1, 2)   # sharp at home, poor away
        return (0,)                                            # flat everywhere
    matrix = _matrix(tmp_path, collapsed=collapsed)
    cost = off_diagonal_cost(cross_domain_matrix(matrix))
    assert np.diag(cost) == pytest.approx([0.0] * len(LABELS))
    assert cost[matrix.conditions.index("cond_a")].tolist() == pytest.approx(
        [0.0 if c == "cond_a" else -3 / N_CLASSES for c in matrix.conditions])
    assert cost[matrix.conditions.index("cond_b")] == pytest.approx(0.0)


def test_a_missing_off_diagonal_cell_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="score_matrix"):
        _matrix(tmp_path, skip=[("cond_a", "cond_b")])


# Guards -- they run before any number is reported

def test_two_conditions_on_one_dataset_fail_the_checksum_guard(tmp_path):
    """The failure this exists for: a config error points two arms at the same file."""
    with pytest.raises(GuardFailure, match="datasets differ"):
        _matrix(tmp_path, shared_dataset=True)


def test_a_different_split_seed_fails(tmp_path):
    with pytest.raises(GuardFailure, match="split_seed shared"):
        _matrix(tmp_path, dataset_attrs={"cond_b": {"split_seed": 999}})


def test_a_different_subset_seed_fails(tmp_path):
    with pytest.raises(GuardFailure, match="subset_seed shared"):
        _matrix(tmp_path, dataset_attrs={"cond_a": {"subset_seed": 999}})


def test_a_different_frame_count_fails(tmp_path):
    with pytest.raises(GuardFailure, match="frame count shared"):
        _matrix(tmp_path, dataset_attrs={"cond_b": {"n_frames": N_FRAMES // 2}})


def test_misaligned_test_rows_fail(tmp_path):
    """Attrs can agree while the scored frames do not; the stored vectors settle it."""
    true, snr = _labels()
    order = np.random.default_rng(0).permutation(true.size)
    with pytest.raises(GuardFailure, match="test rows aligned"):
        _matrix(tmp_path, rows_for={("cond_a", "cond_b"): (true[order], snr[order])})


def test_the_report_states_what_each_arm_actually_read(tmp_path):
    matrix = _matrix(tmp_path)
    assert matrix.conditions == LABELS and matrix.seeds == SEEDS
    assert [matrix.facts[c].condition for c in LABELS] == list(LABELS)
    assert len({matrix.facts[c].checksum for c in LABELS}) == len(LABELS)
