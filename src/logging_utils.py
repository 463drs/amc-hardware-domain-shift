"""Central logging configuration for the project.

One place owns format, level, and destinations; every module gets its logger via
`get_logger(__name__)` and never calls `print()`. `configure_logging()` is called exactly
once, from `train.main()`, so a single run has consistent, timestamped output on the console
and (optionally) in a per-run log file. Module loggers propagate up to the root logger that
this configures, which is what "one logger configured in main, reused everywhere" means in
practice: per-module names for context, a single configuration point.

Named `logging_utils` (not `logging`) so it never shadows the stdlib `logging` module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Console/file line format. The %(name)s is the module logger (e.g. "src.data"), which is
# why the old "[data]" prefixes in the print calls are redundant once routed through here.
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are noisy at DEBUG/INFO and carry no signal for this project.
_QUIET_LIBRARIES = ("matplotlib", "h5py", "PIL")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Use at module top level: ``logger = get_logger(__name__)``."""
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger once: a console handler plus an optional per-run file handler.

    Call this early in `main()`. Every module obtains its own logger via `get_logger`; those
    propagate to the root logger configured here, so there is a single place that owns format,
    level, and destinations.

    Idempotent: existing handlers are cleared first, so calling it again (e.g. across tests or
    a second run in the same process) never double-logs.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Drop handlers from a previous call or from a library's import-time basicConfig so lines
    # are not duplicated and our format/level actually win.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Keep noisy libraries from drowning the run log at our INFO level.
    for lib in _QUIET_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)
