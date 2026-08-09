import wandb

import json
from typing import Any
from dataclasses import dataclass

from src.config import Config
from pathlib import Path


@dataclass(frozen=True)
class RunRef:
    id: str
    name: str
    config: dict[str, Any]
    _raw: Any 

def get_runs_by_config(config: Config) -> list[RunRef]:
    """Get all runs of this config's condition, matched on the logged `condition` key.

    Not name.startswith(): "baseline" also prefixes "baseline_v2", and the name is not authoritative.
    """
    api = wandb.Api()
    runs: list[RunRef] = []
    for r in api.runs(config.experiment.project):
        cfg = dict(r.config)
        if cfg.get("condition") == config.experiment.condition:
            runs.append(RunRef(id=r.id, name=r.name, config=cfg, _raw=r))
    return runs

def download_best(run: RunRef, dest: Path) -> Path:
    """Download best artifacts from given run"""
    for art in run._raw.logged_artifacts():
        if "best" in art.aliases:
            return Path(art.download(root=str(dest)))
    raise FileNotFoundError(f"no artifact with alias 'best' in {run.name}")
    
def download_checkpoints_by_config(config: Config):
    """Download all checkpoints for given config"""
    for run in get_runs_by_config(config):
        if "seed" not in run.config:
            raise RuntimeError(
                f"run {run.name!r} ({run.id}) logs no top-level 'seed'; it predates src.train "
                f"recording it and cannot be placed in the matrix. Retrain the cell."
            )
        seed = run.config["seed"]
        destination = Path(f"runs/{config.experiment.condition}/seed{seed}")
        download_best(run, destination)

        meta = {
            "run_id": run.id,
            "run_name": run.name,
            "config": run.config,
            "summary": dict(run._raw.summary),
        }
        (destination/ "meta.json").write_text(json.dumps(meta, indent=2, default=str))

