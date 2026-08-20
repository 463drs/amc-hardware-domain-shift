"""Tests for the offline prediction path (src.predict).

The focus is verify_split(): it is the only thing standing between a drifted config and an
offline number that silently describes a different experiment. Its previous version read the
stored config at the wrong nesting level, found None for every key, and passed everything --
so these tests pin that a mismatch, a missing section and a missing key all RAISE, and that the
three benign representation differences do not.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.config import Config, resolve_config_path
from src.predict import (
    Cell,
    CellDir,
    discover_cells,
    expected_cells,
    verify_cells,
    verify_split,
)


def _cfg() -> Config:
    return Config.from_yaml(resolve_config_path("debug"))


def _meta(cfg: Config, seed: int = 0, condition: str | None = None, **data_overrides) -> dict:
    """What load_best_models writes: the W&B run config, nested, plus the resolved identity."""
    condition = condition if condition is not None else cfg.experiment.condition
    # Round-trip through JSON exactly as meta.json does -- this is what turns the `split` tuple
    # into a list, one of the representation differences verify_split must tolerate.
    stored = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    stored["data"].update(data_overrides)
    stored["condition"] = condition
    stored["seed"] = seed
    stored["run_name"] = f"{condition}_{seed}"
    return {"run_id": "abc", "run_name": f"{condition}_{seed}",
            "config": stored, "summary": {}}


def _cell(cfg: Config, **data_overrides) -> CellDir:
    return CellDir(
        cell=Cell(condition=cfg.experiment.condition, seed=0),
        path=Path("runs") / cfg.experiment.condition / "seed0",
        meta=_meta(cfg, **data_overrides),
    )


# Passes: the stored config genuinely matches, in whatever representation it came back as.

def test_matching_config_passes():
    cfg = _cfg()
    verify_split(cfg, [_cell(cfg)])          # must not raise


def test_split_list_vs_tuple_is_not_a_difference():
    """asdict gives a tuple, meta.json gives a list; str() of those never matched."""
    cfg = _cfg()
    cell = _cell(cfg, split=[0.7, 0.15, 0.15])
    assert isinstance(cell.meta["config"]["data"]["split"], list)
    assert isinstance(cfg.data.split, tuple)
    verify_split(cfg, [cell])                # must not raise


def test_absolute_path_prefix_is_not_a_difference():
    """Trained on Kaggle, evaluated locally: same dataset, different absolute prefix."""
    cfg = _cfg()
    name = Path(cfg.data.path).name
    verify_split(cfg, [_cell(cfg, path=f"/kaggle/input/radioml/{name}")])


def test_preload_is_not_a_difference():
    """preload picks HOW frames are read, never which frames or their values."""
    cfg = _cfg()
    verify_split(cfg, [_cell(cfg, preload=not cfg.data.preload)])


# Fails loudly: anything that makes the offline number describe a different experiment.

@pytest.mark.parametrize("key,value", [
    ("subset_seed", 999),          # different frames selected from each cell
    ("split_seed", 999),           # same frames, different train/test partition -> leakage
    ("frames_per_pair", 7),
    ("snr_min", -4),
    ("snr_max", 28),
    ("split", [0.8, 0.1, 0.1]),
    ("path", "some_other_subset.hdf5"),
    ("normalization", "none"),     # same frames, different values fed to the model
])
def test_drifted_data_config_raises(key, value):
    cfg = _cfg()
    with pytest.raises(RuntimeError, match="differs from the training run"):
        verify_split(cfg, [_cell(cfg, **{key: value})])


def test_all_differences_are_reported_at_once():
    cfg = _cfg()
    with pytest.raises(RuntimeError) as excinfo:
        verify_split(cfg, [_cell(cfg, subset_seed=999, split_seed=888)])
    message = str(excinfo.value)
    assert "2 key(s)" in message
    assert "subset_seed" in message and "split_seed" in message


# Unverifiable is an error, not a pass -- this is the exact bug the old version had.

def test_missing_data_section_raises():
    cfg = _cfg()
    cell = _cell(cfg)
    del cell.meta["config"]["data"]
    with pytest.raises(RuntimeError, match="no 'config.data' section"):
        verify_split(cfg, [cell])


def test_flat_legacy_config_raises_instead_of_silently_passing():
    """A top-level (un-nested) config is what the old lookup assumed; it verified nothing."""
    cfg = _cfg()
    cell = _cell(cfg)
    cell.meta["config"] = {"subset_seed": 999, "split_seed": 888}   # no "data" key
    with pytest.raises(RuntimeError, match="no 'config.data' section"):
        verify_split(cfg, [cell])


def test_missing_single_key_raises():
    cfg = _cfg()
    cell = _cell(cfg)
    del cell.meta["config"]["data"]["subset_seed"]
    with pytest.raises(RuntimeError, match="missing"):
        verify_split(cfg, [cell])


def test_every_cell_is_checked_not_just_the_first():
    cfg = _cfg()
    good, bad = _cell(cfg), _cell(cfg, subset_seed=999)
    with pytest.raises(RuntimeError, match="differs from the training run"):
        verify_split(cfg, [good, bad])


# Run identity comes from the logged config, not from splitting run_name on "_".

def _write_cell(root: Path, condition: str, seed: int, meta: dict) -> None:
    d = root / condition / f"seed{seed}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_discover_cells_reads_identity_from_config(tmp_path):
    cfg = _cfg()
    _write_cell(tmp_path, "debug", 2, _meta(cfg, seed=2))
    (cell,) = discover_cells(tmp_path)
    assert cell.cell == Cell(condition="debug", seed=2)


def test_discover_cells_handles_condition_with_underscores(tmp_path):
    """rtl_sdr_gain0 used to parse as condition 'rtl', seed 'sdr' -> int('sdr') ValueError."""
    cfg = _cfg()
    _write_cell(tmp_path, "rtl_sdr_gain0", 11, _meta(cfg, seed=11, condition="rtl_sdr_gain0"))
    (cell,) = discover_cells(tmp_path)
    assert cell.cell == Cell(condition="rtl_sdr_gain0", seed=11)


def test_discover_cells_rejects_metadata_without_identity(tmp_path):
    """Old metadata carried identity only in the name; guessing it would be a silent mismatch."""
    cfg = _cfg()
    meta = _meta(cfg, seed=3)
    del meta["config"]["condition"], meta["config"]["seed"]
    _write_cell(tmp_path, "debug", 3, meta)
    with pytest.raises(RuntimeError, match="no top-level 'condition'/'seed'"):
        discover_cells(tmp_path)


# runs/ holds every downloaded config side by side, so a neighbour is not this cell's problem.

def test_a_neighbouring_conditions_cell_is_not_an_extra(tmp_path):
    cfg = _cfg()
    found = [CellDir(cell=Cell(condition=cfg.experiment.condition, seed=s), path=tmp_path,
                     meta=_meta(cfg, seed=s)) for s in cfg.train.seeds]
    found.append(CellDir(cell=Cell(condition="phase_noise_100_ex", seed=0), path=tmp_path,
                         meta=_meta(cfg, condition="phase_noise_100_ex")))
    verify_cells(expected_cells(cfg), found)


def test_an_unexpected_seed_of_this_condition_still_fails(tmp_path):
    """The filter must scope by condition only -- a seed the config never declared is a defect."""
    cfg = _cfg()
    found = [CellDir(cell=Cell(condition=cfg.experiment.condition, seed=s), path=tmp_path,
                     meta=_meta(cfg, seed=s)) for s in list(cfg.train.seeds) + [999]]
    with pytest.raises(RuntimeError, match="cell mismatch"):
        verify_cells(expected_cells(cfg), found)
