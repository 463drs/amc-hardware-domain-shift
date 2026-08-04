"""CLI entry point for training. All logic lives in src.train.

    python scripts/train.py --config configs/baseline.yaml
"""

import sys
from pathlib import Path

#resume disabled untill first learning crash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.train import train


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI. A function (not module-level) so the flag contract stays
    importable and unit-testable without argparse consuming the caller's argv."""
    parser = argparse.ArgumentParser(description="Train an AMC model from a YAML config.")
    parser.add_argument("--config", required=True, help="config name or path (see configs/)")
    parser.add_argument("--seed", required=True, help="seed to lock rng")
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
    train(args.config, seed=args.seed, resume=None, fresh=True)
