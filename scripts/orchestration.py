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
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO_ROOT = Path(__file__).resolve().parents[1]

from src.telegram_notification import send_telegram_notification
from src.config import Config, TrainConfig, resolve_config_path

try:
    from kaggle_secrets import UserSecretsClient # type: ignore
    user_secrets = UserSecretsClient()
    os.environ["TELEGRAM_BOT_TOKEN"] = user_secrets.get_secret("TELEGRAM_BOT_TOKEN")
    os.environ["TELEGRAM_CHAT_ID"] = user_secrets.get_secret("TELEGRAM_CHAT_ID")
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

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
                        help="ignore any existing checkpoints and train from epoch 1, makes run id not determenistic")

    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    config = Config.from_yaml(resolve_config_path(args.config))
    seeds = config.train.seeds
    num_gpus = torch.cuda.device_count()
    num_workers = max(1, num_gpus)
    task_queue = queue.Queue()

    start_time_sec = time.time()
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    send_telegram_notification(bot_token, chat_id, text=f"Starting training: \nconfig: {args.config}")
    
    for seed in seeds:
        run_id = f"{config.experiment.condition}_{seed}_{start_time}"
        task_queue.put((seed, run_id))

    def worker(gpu_id: int):
        env = os.environ.copy()
        if num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        while not task_queue.empty():
            try:
                seed, run_id = task_queue.get_nowait()
            except queue.Empty:
                break

            print(f'starting train at device: {gpu_id}')
            cmd = [
                sys.executable,
                "-u",
                "scripts/train.py",
                "--config", args.config,
                "--seed", str(seed),
                "--run_id", run_id
            ]
            if args.fresh:
                cmd.append("--fresh")
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                send_telegram_notification(
                    bot_token, chat_id,
                    text=f"Training failed: \ndevice: {gpu_id}\nseed: {seed}\nconfig: {args.config}\nExit code: {e}"
                )
            finally:
                task_queue.task_done()

    threads = []
    
    for gpu_id in range(num_workers):
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    elapsed_sec = int(time.time() - start_time_sec)
    hours, remainder = divmod(elapsed_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    send_telegram_notification(
        bot_token, chat_id, 
        text=f"Training complete:\nconfig: {args.config}\ntime elapsed: {formatted_time}"
        )