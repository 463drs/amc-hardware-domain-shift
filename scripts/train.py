"""CLI entry point for training. All logic lives in src.train.

    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config baseline --resume
    python scripts/train.py --config baseline --resume outputs/baseline_0/last.pt
    python scripts/train.py --config baseline --fresh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.train import main


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI. A function (not module-level) so the flag contract stays
    importable and unit-testable without argparse consuming the caller's argv."""
    parser = argparse.ArgumentParser(description="Train an AMC model from a YAML config.")
    parser.add_argument("--config", required=True, help="config name or path (see configs/)")

    # --resume and --fresh are opposite intents; let argparse reject the combination rather
    # than resolving it silently.
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument("--resume", nargs="?", const=True, default=None, metavar="PATH",
                        help="continue an existing run: bare --resume auto-selects the newer "
                             "of last.pt/best.pt; --resume PATH resumes that exact checkpoint")
    intent.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoints and train from epoch 1")

    parser.add_argument("--allow-config-change", action="store_true",
                        help="permit resuming when the config differs from the checkpoint's "
                             "(warns with a diff instead of refusing)")
    parser.add_argument("--run-id", default=None,
                        help="override the run name (default: <condition>_<seed>)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.config, resume=args.resume, fresh=args.fresh,
         run_id=args.run_id, allow_config_change=args.allow_config_change)
