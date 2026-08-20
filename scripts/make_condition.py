"""Apply one condition's R_theta to the frozen subset and write a new standalone HDF5.
  make_condition("phase_noise", config="baseline_100")  ==  --condition x --config y"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_config_path, resolve_data_path
from src.data import KEY_X, KEY_Y, KEY_Z
from src.distortions import RNG_SCHEME, apply_to_frame, build_compose, condition_key

# Frames processed per HDF5 read/write batch. Bounds memory only -- output is
# independent of it, which the reproducibility test asserts.
_BATCH = 4096

_DEFAULT_CONDITIONS = "conditions.yaml"
# Attributes copied verbatim from the source so the output stays a drop-in replacement
# for data.path (same subset, same split, same class order).
_INHERITED_ATTRS = (
    "frames_per_pair", "subset_seed", "snr_min", "snr_max", "split", "split_seed",
    "n_classes", "frame_length", "class_order", "source_file", "source_n_frames",
)


def _load_conditions_file(path: str | Path | None = None) -> dict:
    """Read and validate the whole conditions document."""
    resolved = resolve_config_path(str(path) if path is not None else _DEFAULT_CONDITIONS)
    with Path(resolved).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "conditions" not in raw:
        raise ValueError(f"{resolved} must be a mapping with a top-level 'conditions' key.")
    return raw


def load_conditions(path: str | Path | None = None) -> dict:
    """Read the condition -> operator-spec mapping from YAML."""
    return _load_conditions_file(path)["conditions"]


def load_sample_rate_hz(path: str | Path | None = None) -> float | None:
    """The nominal f_s the file's datasheet figures are converted against; None if undeclared.

    It belongs to the file, not to a condition: one rate converts every per-Hz figure, and it
    is recorded in each generated dataset so the assumption is readable off the file itself."""
    raw = _load_conditions_file(path).get("sample_rate_hz")
    if raw is None:
        return None
    try:
        # float(): YAML 1.1 reads an unsigned exponent (1.024e6) as a string, not a number.
        sample_rate_hz = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"sample_rate_hz must be a number, got {raw!r}") from None
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    return sample_rate_hz


def verify_output(path: str | Path) -> tuple[bool, str]:
    """Is a generated file complete? Returns (ok, reason), checking structure then checksum.
    Attributes are written last, so a missing checksum is itself a truncation signal."""
    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    try:
        with h5py.File(path, "r") as f:
            for key in ("content_checksum", "n_frames", "condition"):
                if key not in f.attrs:
                    return False, f"missing attribute {key!r} -- the write did not complete"
            n_frames = int(f.attrs["n_frames"])
            if f[KEY_X].shape[0] != n_frames:
                return False, f"X has {f[KEY_X].shape[0]} rows, metadata declares {n_frames}"
            stored = str(f.attrs["content_checksum"])
            # Chunked, but blake2b over a byte stream is independent of how it is chunked,
            # so this matches the writer regardless of _BATCH changing between runs.
            digest = hashlib.blake2b(digest_size=16)
            for start in range(0, n_frames, _BATCH):
                digest.update(np.ascontiguousarray(f[KEY_X][start:start + _BATCH]).tobytes())
    except OSError as exc:
        return False, f"unreadable HDF5 ({exc})"

    actual = f"blake2b16:{digest.hexdigest()}"
    if actual != stored:
        return False, f"checksum mismatch (stored {stored}, actual {actual})"
    return True, "complete"


def default_output_name(source: Path, condition: str, compress: bool) -> str:
    """Signature name: the source subset plus the condition that was injected."""
    suffix = "_gz" if compress else ""
    return f"{source.stem}_cond-{condition}{suffix}.hdf5"


def make_condition(
    condition: str,
    config: str | Path | None = None,
    path: str | Path | None = None,
    conditions: str | Path | None = None,
    out: str | Path | None = None,
    out_dir: str | Path | None = None,
    compress: bool = False,
    overwrite: bool = False,
    verify: bool = True,
    verbose: bool = True,
) -> Path:
    """Generate the dataset for one condition; returns the output path.
    The source HDF5 comes from `config` (its data.path) or an explicit `path`."""
    table = load_conditions(conditions)
    if condition not in table:
        raise ValueError(f"Unknown condition {condition!r}. Available: {sorted(table)}.")
    sample_rate_hz = load_sample_rate_hz(conditions)
    compose = build_compose(table[condition], sample_rate_hz=sample_rate_hz)

    src_path = Path(resolve_data_path(config, path)[0])
    if out is not None:
        out_path = Path(out)
    else:
        base = Path(out_dir) if out_dir is not None else src_path.parent
        out_path = base / default_output_name(src_path, condition, compress)

    if out_path.resolve() == src_path.resolve():
        raise ValueError("Output path must differ from the source dataset.")
    if out_path.exists() and not overwrite:
        # Existence alone is NOT readiness: verify before reusing a file in a frozen dataset.
        ok, reason = verify_output(out_path) if verify else (True, "verification disabled")
        if not ok:
            raise RuntimeError(
                f"{out_path} exists but is INCOMPLETE ({reason}). An interrupted run most "
                f"likely left it. Delete it, or pass overwrite=True / --overwrite."
            )
        if verbose:
            print(f"[make_condition] {out_path} already exists ({reason}); skipping.")
        return out_path

    theta = compose.params()
    if verbose:
        print(f"[make_condition] source   : {src_path}")
        print(f"[make_condition] output   : {out_path}")
        print(f"[make_condition] condition: {condition}  (identity={compose.is_identity})")
        print(f"[make_condition] f_s      : {sample_rate_hz if sample_rate_hz else 'not declared'}")
        print(f"[make_condition] theta    : {json.dumps(theta)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if compress else None
    # Streaming checksum of the output X, in write order -- the file's content identity.
    digest = hashlib.blake2b(digest_size=16)
    # Publish atomically: build a sibling .partial and rename only once the write completes,
    # so a crash can never leave a truncated file that the exists-check would call ready.
    tmp_path = out_path.with_name(out_path.name + ".partial")

    try:
        with h5py.File(src_path, "r") as fin, h5py.File(tmp_path, "w") as fout:
            x_in = fin[KEY_X]
            n_frames, frame_len, n_iq = x_in.shape
            if n_iq != 2:
                raise ValueError(f"expected X of shape (N, T, 2), got {x_in.shape}")

            x_out = fout.create_dataset(
                KEY_X, shape=x_in.shape, dtype=x_in.dtype, compression=compression,
            )
            # Labels are untouched: R_theta changes the observation, never the class or SNR.
            fout.create_dataset(KEY_Y, data=fin[KEY_Y][:], compression=compression)
            fout.create_dataset(KEY_Z, data=fin[KEY_Z][:], compression=compression)

            # Sequential 1:1 with the source: the fixed stratified split indices stay valid.
            # Nothing is normalized here -- src/data.py applies unit_power at LOAD, after this.
            for start in range(0, n_frames, _BATCH):
                stop = min(start + _BATCH, n_frames)
                batch = np.asarray(x_in[start:stop])
                for i in range(stop - start):
                    # Global index, so the key is independent of batching and of any re-run.
                    batch[i] = apply_to_frame(compose, batch[i], condition, start + i)
                x_out[start:stop] = batch
                digest.update(np.ascontiguousarray(batch).tobytes())
                if verbose:
                    print(f"\r[make_condition] frames: {stop}/{n_frames}", end="", flush=True)
            if verbose:
                print()

            for key in _INHERITED_ATTRS:
                if key in fin.attrs:
                    fout.attrs[key] = fin.attrs[key]

            fout.attrs["condition"] = condition
            fout.attrs["theta"] = json.dumps(theta)
            fout.attrs["theta_is_identity"] = bool(compose.is_identity)
            fout.attrs["rng_scheme"] = RNG_SCHEME
            if sample_rate_hz is not None:
                # The rate theta's per-Hz figures were converted against: without it, a sigma_w
                # in the metadata is a number with no stated assumption behind it.
                fout.attrs["sample_rate_hz"] = sample_rate_hz
            fout.attrs["condition_key"] = str(condition_key(condition))
            fout.attrs["content_checksum"] = f"blake2b16:{digest.hexdigest()}"
            fout.attrs["n_frames"] = int(n_frames)
            fout.attrs["frame_length"] = int(frame_len)
            fout.attrs["normalization_applied"] = "none"  # unit_power happens at load time
            fout.attrs["subset_file"] = src_path.name
            fout.attrs["conditions_file"] = str(
                resolve_config_path(str(conditions) if conditions is not None else _DEFAULT_CONDITIONS)
            )
            fout.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    except BaseException:
        # BaseException, not Exception: Ctrl-C is the likeliest way a long run dies.
        tmp_path.unlink(missing_ok=True)
        raise

    # Atomic on POSIX and Windows alike, and it replaces an existing target.
    os.replace(tmp_path, out_path)

    if verbose:
        size_gb = out_path.stat().st_size / 1e9
        print(f"[make_condition] wrote {out_path} ({n_frames} frames, {size_gb:.2f} GB)")
        print(f"[make_condition] checksum blake2b16:{digest.hexdigest()}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one condition's dataset.")
    parser.add_argument("--condition", default=None, help="Condition name in the conditions YAML.")
    parser.add_argument("--config", default=None,
                        help="Config name (e.g. 'baseline_100') or path; its data.path is the source.")
    parser.add_argument("--path", default=None, help="Explicit source HDF5 path.")
    parser.add_argument("--conditions", default=None,
                        help=f"Conditions YAML name or path (default: {_DEFAULT_CONDITIONS}).")
    parser.add_argument("--out", default=None, help="Explicit output HDF5 path.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the auto-named output (default: the source's directory).")
    parser.add_argument("--compress", action="store_true", help="gzip-compress the output.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite an existing output.")
    parser.add_argument("--list", action="store_true", help="List available conditions and exit.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the checksum check when reusing an existing output.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only check an existing output for completeness; generate nothing.")
    args = parser.parse_args()

    if args.list:
        print("\n".join(sorted(load_conditions(args.conditions))))
        return
    if args.condition is None:
        parser.error("--condition is required (use --list to see the available names).")

    if args.verify_only:
        src = Path(resolve_data_path(args.config, args.path)[0])
        target = Path(args.out) if args.out else (
            (Path(args.out_dir) if args.out_dir else src.parent)
            / default_output_name(src, args.condition, args.compress))
        ok, reason = verify_output(target)
        print(f"[make_condition] {target}: {'OK' if ok else 'INCOMPLETE'} -- {reason}")
        sys.exit(0 if ok else 1)

    make_condition(
        condition=args.condition, config=args.config, path=args.path,
        conditions=args.conditions, out=args.out, out_dir=args.out_dir,
        compress=args.compress, overwrite=args.overwrite,
        verify=not args.no_verify, verbose=True,
    )


if __name__ == "__main__":
    main()
