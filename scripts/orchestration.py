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
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO_ROOT = Path(__file__).resolve().parents[1]

from src.telegram_notification import (
    format_elapsed,
    load_telegram_credentials,
    send_telegram_notification,
)
from src.config import Config, TrainConfig, resolve_config_path


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI. A function (not module-level) so the flag contract stays
    importable and unit-testable without argparse consuming the caller's argv."""
    parser = argparse.ArgumentParser(
        description="Train every train.seeds entry of a config, one subprocess per GPU.")
    # required: a missing --config previously reached Path(None) and died with a bare TypeError.
    parser.add_argument("--config", required=True, help="config name or path (see configs/)")

    # --resume and --fresh are opposite intents; let argparse reject the combination rather
    # than resolving it silently. Forwarded verbatim to each scripts/train.py subprocess, so the
    # per-seed behaviour is exactly what running that command by hand would do.
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument("--resume", nargs="?", const=True, default=None, metavar="PATH",
                        help="continue existing runs: bare --resume auto-selects the newer "
                             "of last.pt/best.pt per seed; --resume PATH resumes that exact "
                             "checkpoint (only meaningful with a single seed)")
    intent.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoints and train from epoch 1, makes run id not determenistic")
    # Console output only.
    parser.add_argument("--no-progress", "--no_progress", dest="no_progress",
                        action="store_true",
                        help="suppress the per-epoch tqdm bars; keep the per-epoch summary lines")
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    config = Config.from_yaml(resolve_config_path(args.config))
    seeds = config.train.seeds
    num_gpus = torch.cuda.device_count()
    num_workers = max(1, num_gpus)
    task_queue = queue.Queue()

    start_time_sec = time.time()
    # Loaded here, not at import: importing this module must have no side effects (tests do).
    bot_token, chat_id = load_telegram_credentials()

    send_telegram_notification(
        bot_token, chat_id,
        text=f"Starting training: \nconfig: {args.config}\nseeds: {list(seeds)}")

    # No synthesized run id: run identity is left to src.train, which derives it from
    # condition + seed. Stamping a launch time here gave every restart a NEW run name, which
    # defeated both the resume contract and the deterministic W&B id (a restarted session would
    # spawn a duplicate run instead of continuing its curve). --fresh still adds a timestamp,
    # there, deliberately, because that is the one case that must not collide.
    failures: list[tuple[int, str]] = []
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

            print(f'starting train at device: {gpu_id}')
            cmd = [
                sys.executable,
                "-u",
                str(_REPO_ROOT / "scripts" / "train.py"),
                "--config", args.config,
                "--seed", str(seed)
            ]
            if args.no_progress:
                cmd.append("--no-progress")
            # Forward the intent flags so the orchestrated run behaves exactly like the same
            # command typed by hand -- without this, --resume was silently dropped.
            if args.fresh:
                cmd.append("--fresh")
            elif args.resume is not None:
                cmd.append("--resume")
                if args.resume is not True:      # bare --resume vs --resume PATH
                    cmd.append(str(args.resume))
            try:
                subprocess.run(cmd, env=env, check=True, cwd=str(_REPO_ROOT))
            except subprocess.CalledProcessError as e:
                # Recorded, not announced: the child already sent a failure message with the
                # exception itself. Tracking it here still catches a child killed outright
                # (OOM, SIGKILL) that never got to notify -- the final summary names it.
                failures.append((seed, str(e)))
                print(f"seed {seed} failed on device {gpu_id}: {e}")
            finally:
                task_queue.task_done()

    threads = []
    
    for gpu_id in range(num_workers):
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    formatted_time = format_elapsed(time.time() - start_time_sec)

    # Report what actually happened: announcing "complete" after every seed crashed is exactly
    # the kind of silent success that costs a night of GPU time.
    if failures:
        failed_seeds = ", ".join(str(s) for s, _ in sorted(failures))
        send_telegram_notification(
            bot_token, chat_id,
            text=f"All runs FINISHED WITH FAILURES:\nconfig: {args.config}\n"
                 f"failed seeds: {failed_seeds} ({len(failures)}/{len(seeds)})\n"
                 f"time elapsed: {formatted_time}"
            )
        sys.exit(1)

    send_telegram_notification(
        bot_token, chat_id,
        text=f"All {len(seeds)} runs complete:\nconfig: {args.config}\n"
             f"time elapsed: {formatted_time}"
        )