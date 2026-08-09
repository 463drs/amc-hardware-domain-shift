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
    """Get all runs based on config"""
    api = wandb.Api()
    return [
        RunRef(id=r.id, name=r.name, config=dict(r.config), _raw=r)
        for r in api.runs(config.experiment.project)
        if r.name.startswith(config.experiment.condition)
    ]

def download_best(run: RunRef, dest: Path) -> Path:
    """Download best artifacts from given run"""
    for art in run._raw.logged_artifacts():
        if "best" in art.aliases:
            return Path(art.download(root=str(dest)))
    raise FileNotFoundError(f"no artifact with alias 'best' in {run.name}")
    
def download_checkpoints_by_config(config: Config):
    """Download all checkpoints for given config"""
    for run in get_runs_by_config(config):
        seed = run.name.split('_')[1]
        destination = Path(f"runs/{config.experiment.condition}/seed{seed}")
        download_best(run, destination)

        meta = {
            "run_id": run.id,
            "run_name": run.name,
            "config": run.config,
            "summary": dict(run._raw.summary),
        }
        (destination/ "meta.json").write_text(json.dumps(meta, indent=2, default=str))

