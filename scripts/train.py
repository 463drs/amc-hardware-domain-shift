"""CLI entry point for training. All logic lives in src.train.

    python scripts/train.py --config baseline --seed 10
    python scripts/train.py --config baseline --seed 10 --resume
    python scripts/train.py --config baseline --seed 10 --fresh
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.telegram_notification import (
    format_elapsed,
    load_telegram_credentials,
    send_telegram_notification,
)
from src.train import train


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI. A function (not module-level) so the flag contract stays
    importable and unit-testable without argparse consuming the caller's argv."""
    parser = argparse.ArgumentParser(description="Train an AMC model from a YAML config.")
    parser.add_argument("--config", required=True, help="config name or path (see configs/)")
    # type=int is load-bearing, not cosmetic: without it argparse yields a str, and set_seed()
    # dies on np.random.seed(str) before the first batch (torch.manual_seed happens to accept
    # one, so the failure looks unrelated to the flag).
    parser.add_argument("--seed", required=True, type=int,
                        help="training seed: model init, shuffling, dropout")
    # Both spellings accepted so existing callers (scripts/orchestration.py, notebook cells)
    # keep working either way.
    parser.add_argument("--run-id", "--run_id", dest="run_id", default=None, metavar="NAME",
                        help="override the run NAME (default: <condition>_<seed>); use it to "
                             "keep a retrain separate from an existing run")
    # --resume and --fresh are opposite intents; let argparse reject the combination rather
    # than resolving it silently.
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument("--resume", nargs="?", const=True, default=None, metavar="PATH",
                        help="continue an existing run: bare --resume auto-selects the newer "
                             "of last.pt/best.pt; --resume PATH resumes that exact checkpoint")
    intent.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoints and train from epoch 1")

    return parser


def _completion_text(summary: dict, config: str, elapsed: float) -> str:
    return (
        f"Training complete\n"
        f"run: {summary['run_name']}\n"
        f"config: {config}\n"
        f"best {summary['best_metric_name']}: {summary['best_metric']:.4f} "
        f"(epoch {summary['best_epoch']} of {summary['epochs_run']} run)\n"
        f"elapsed: {format_elapsed(elapsed)}"
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    # Read once, before training: a 2-hour run should not fail to notify because the environment
    # changed underneath it. Missing credentials make every send a no-op.
    bot_token, chat_id = load_telegram_credentials()
    started = time.time()
    try:
        # Forward the intent verbatim. resolve_resume_target() owns the three-state contract
        # (refuse if a checkpoint exists / resume it / discard it), and hardcoding it here would
        # override the user: it previously forced fresh=True, so --resume could never take effect
        # and every launch started a new timestamped run.
        summary = train(args.config, seed=args.seed, resume=args.resume, fresh=args.fresh,
                        run_id=args.run_id)
    except BaseException as exc:
        # Notify then re-raise: the exit code is what orchestration.py reads, so it must survive.
        send_telegram_notification(
            bot_token, chat_id,
            text=(f"Training FAILED\nconfig: {args.config}\nseed: {args.seed}\n"
                  f"elapsed: {format_elapsed(time.time() - started)}\n"
                  f"{type(exc).__name__}: {exc}"),
        )
        raise
    send_telegram_notification(
        bot_token, chat_id,
        text=_completion_text(summary, args.config, time.time() - started),
    )
