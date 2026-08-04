"""
    CLI orchestration script

    python scripts/orchestration.py --config configs/baseline.yaml
"""

from pathlib import Path


import argparse
import sys
import torch
import queue
import subprocess
import os
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO_ROOT = Path(__file__).resolve().parents[1]

from src.config import Config, TrainConfig, resolve_config_path

def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI. A function (not module-level) so the flag contract stays
    importable and unit-testable without argparse consuming the caller's argv."""
    parser = argparse.ArgumentParser(description="Train an AMC model from a YAML config.")
    parser.add_argument("--config", required=False, help="config name or path (see configs/)")

    # --resume and --fresh are opposite intents; let argparse reject the combination rather
    # than resolving it silently.
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument("--resume", nargs="?", const=True, default=None, metavar="PATH",
                        help="continue an existing run: bare --resume auto-selects the newer "
                             "of last.pt/best.pt; --resume PATH resumes that exact checkpoint")
    intent.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoints and train from epoch 1")

    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    config = Config.from_yaml(resolve_config_path(args.config))
    seeds = config.train.seeds
    num_gpus = torch.cuda.device_count()
    num_workers = max(1, num_gpus)
    task_queue = queue.Queue()
    for seed in seeds:
        task_queue.put(seed)

    def worker(gpu_id: int):
        env = os.environ.copy()
        if num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        while not task_queue.empty():
            try:
                seed = task_queue.get_nowait()
            except queue.Empty:
                break
            if seed:
                print(f'starting train at device: {gpu_id}')
            cmd = [
                sys.executable,
                "-u",
                "scripts/train.py",
                "--config", args.config,
                "--seed", str(seed),
            ]
            
            subprocess.run(cmd, env=env, check=True)
            task_queue.task_done()
    
    threads = []
    
    for gpu_id in range(num_workers):
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
