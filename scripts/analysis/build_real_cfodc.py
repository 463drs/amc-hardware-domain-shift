"""Build the DIAGNOSTIC CFO-corrected copy of the frozen real set: same rows, labels and bins, each frame
derotated by its capture's CFO. Evaluated separately; it never replaces the primary set.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import domain_compare as dc     # noqa: E402

OUT_PATH = "data/real_captures_2db_noise-bw_cfodc.hdf5"
LOCK_HZ = 300.0            # a per-frame estimate this close to the capture's line counts as locked
LOCK_FRAC = 0.9            # per-frame derotation only when this share of a capture's frames lock
K_NEIGHBOURS = 4           # line-bearing captures nearest in time that stand in for a line-less one
LINE_FRAMES = 1024         # recording frames summed for the capture line; lines show clearly by ~1000
DIAGNOSTIC_NOTE = ("DIAGNOSTIC set: the frozen real set with per-frame CFO removed. Evaluated "
                   "separately; it NEVER replaces the primary set for any headline number.")


def estimate(c: dict, meta: dict, stored: np.ndarray, captures: Path) -> dict:
    """Capture line from every frame of its recording, then the per-frame lock on the stored rows."""
    path = captures / c["file"]
    frames, _ = dc.front_end(path, min(c["frames_available"], LINE_FRAMES), meta["front_end"]["k_shift_bins"],
                             meta["notch"]["centres_hz"].values(), meta["notch"]["half_bins"])
    cfo, order, prom = dc.capture_line(frames)
    rec = {"file": c["file"], "class": c["class"], "bin": c["bin"], "row_start": c["row_start"],
           "n_frames": c["n_frames"], "mtime": path.stat().st_mtime,
           "line_order": order, "line_prominence_db": round(prom, 2),
           "line_ok": prom >= dc.LINE_MIN_DB, "line_cfo_hz": round(cfo, 1)}
    if rec["line_ok"]:
        pf = dc.cfo_per_frame(stored, order)
        rec["lock_frac"] = round(float(np.mean(np.abs(pf - cfo) <= LOCK_HZ)), 3)
    return rec


def nearest_line_cfo(rec: dict, pool: list[dict], exclude: str | None = None) -> tuple[float, list, float]:
    """Median CFO of the K line-bearing captures closest in capture time -> (cfo, files, max gap s)."""
    near = sorted((r for r in pool if r["file"] != exclude), key=lambda r: abs(r["mtime"] - rec["mtime"]))
    near = near[:K_NEIGHBOURS]
    return (float(np.median([r["line_cfo_hz"] for r in near])), [r["file"] for r in near],
            float(max(abs(r["mtime"] - rec["mtime"]) for r in near)))


def derotation(rec: dict, stored: np.ndarray) -> tuple[np.ndarray, str]:
    """Per-frame CFO vector for one capture's stored rows, and the mode that produced it."""
    cfo = np.full(len(stored), rec["cfo_hz"])
    if rec["line_ok"] and rec["lock_frac"] >= LOCK_FRAC:
        pf = dc.cfo_per_frame(stored, rec["line_order"])
        locked = np.abs(pf - rec["line_cfo_hz"]) <= LOCK_HZ
        cfo[locked] = pf[locked]
        return cfo, "per-frame"
    return cfo, "per-capture"


def main() -> int:
    p = argparse.ArgumentParser(description="Build the diagnostic CFO-corrected real set.")
    p.add_argument("--src", default=dc.REAL_PATH, help="The frozen real set.")
    p.add_argument("--captures", default=dc.CAPTURES_DIR)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src, out, captures = dc._abs(args.src), dc._abs(args.out), dc._abs(args.captures)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to rewrite it")
        return 1
    with h5py.File(src, "r") as f:
        meta = json.loads(f.attrs["metadata"])
        attrs = dict(f.attrs)
        caps = [c for c in meta["captures"] if c["n_frames"]]
        recs = []
        for i, c in enumerate(caps):
            a = c["row_start"]
            x = f["X"][a:a + c["n_frames"]]
            recs.append(estimate(c, meta, x[..., 0] + 1j * x[..., 1].astype(np.float64), captures))
            print(f"\r  estimating {i + 1}/{len(caps)}", end="", flush=True)
    print()

    pool = [r for r in recs if r["line_ok"]]
    if not pool:
        raise ValueError("no capture carries a usable line; nothing to anchor the fallback")
    # Leave-one-out: how well do time neighbours predict a capture whose line we DO know?
    loo = np.array([nearest_line_cfo(r, pool, exclude=r["file"])[0] - r["line_cfo_hz"] for r in pool])
    for r in recs:
        if r["line_ok"]:
            r["cfo_hz"], r["cfo_source"] = r["line_cfo_hz"], f"x^{r['line_order']} line"
        else:
            cfo, files, gap = nearest_line_cfo(r, pool)
            r |= {"cfo_hz": round(cfo, 1), "cfo_source": "time neighbours",
                  "neighbours": files, "neighbour_max_gap_s": round(gap, 1)}

    tmp = out.with_suffix(".partial")
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fo:
        for k in ("Y", "Z"):
            fo.create_dataset(k, data=fin[k][:])
        xo = fo.create_dataset("X", shape=fin["X"].shape, dtype=fin["X"].dtype)
        n = np.arange(dc.FRAME_LEN)
        for i, r in enumerate(recs):
            a, b = r["row_start"], r["row_start"] + r["n_frames"]
            x = fin["X"][a:b]
            z = x[..., 0].astype(np.float64) + 1j * x[..., 1]
            cfo, r["mode"] = derotation(r, z)
            z = z * np.exp(-2j * np.pi * cfo[:, None] * n / dc.FS_RX)
            r["frame_cfo_median_hz"] = round(float(np.median(cfo)), 1)
            r["frame_cfo_iqr_hz"] = round(float(np.subtract(*np.percentile(cfo, [75, 25]))), 1)
            if r["line_ok"]:     # what line is left after derotation, same order
                r["residual_hz"] = round(dc.capture_line(z, (r["line_order"],))[0], 1)
            xo[a:b] = np.stack((z.real, z.imag), axis=-1).astype(x.dtype)
            print(f"\r  writing {i + 1}/{len(recs)}", end="", flush=True)
        print()

        modes = {m: sum(r["mode"] == m for r in recs) for m in ("per-frame", "per-capture")}
        sources = {s: sum(r["cfo_source"] == s for r in recs) for s in sorted({r["cfo_source"] for r in recs})}
        meta["diagnostic"] = {
            "note": DIAGNOSTIC_NOTE, "primary_set": src.name,
            "primary_created_utc": str(attrs.get("created_utc")),
            "identical": "rows, Y, Z and every |x| equal the primary set; only the phase differs",
            "method": {
                "capture_line": f"CFO of each capture from |FFT(x**p)|^2 summed over all its frames in "
                                f"captures/, p in {list(dc.LINE_ORDERS)}, best prominence; usable when "
                                f">= {dc.LINE_MIN_DB} dB over the search-window median",
                "per_frame": f"frames whose own x**p estimate is within {LOCK_HZ:g} Hz of the capture "
                             f"line, in captures where >= {LOCK_FRAC:.0%} lock; else the capture line",
                "no_line": f"median line CFO of the {K_NEIGHBOURS} line-bearing captures nearest in "
                           "capture time (file mtime); the per-frame centroid is NOT used, it reads "
                           "~2 kHz of in-band tilt and FM's intended +32 kHz as CFO",
                "leave_one_out_error_hz": {
                    "median_abs": round(float(np.median(np.abs(loo))), 1),
                    "p95_abs": round(float(np.percentile(np.abs(loo), 95)), 1),
                    "max_abs": round(float(np.abs(loo).max()), 1), "n": int(loo.size)}},
            "counts": {"captures": len(recs), "modes": modes, "sources": sources},
            "captures": recs}
        for k, v in attrs.items():
            fo.attrs[k] = v
        fo.attrs["metadata"] = json.dumps(meta, indent=1)
        fo.attrs["diagnostic"] = DIAGNOSTIC_NOTE
        fo.attrs["primary_set"] = src.name
        fo.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    tmp.replace(out)

    # Only the phase may differ: labels bit-equal, magnitudes equal to float32 rounding.
    with h5py.File(src, "r") as a, h5py.File(out, "r") as b:
        same_yz = all(np.array_equal(a[k][:], b[k][:]) for k in ("Y", "Z"))
        worst = 0.0
        for s in range(0, a["X"].shape[0], 16384):
            xa, xb = a["X"][s:s + 16384], b["X"][s:s + 16384]
            worst = max(worst, float(np.abs(np.hypot(*xa.transpose(2, 0, 1))
                                            - np.hypot(*xb.transpose(2, 0, 1))).max()))
    res = [abs(r["residual_hz"]) for r in recs if "residual_hz" in r]
    print(f"wrote {out}")
    print(f"  captures {len(recs)}: {sources}; modes {modes}")
    print(f"  leave-one-out neighbour error |Hz|: median {np.median(np.abs(loo)):.0f}, "
          f"p95 {np.percentile(np.abs(loo), 95):.0f}, max {np.abs(loo).max():.0f}  ({loo.size} captures)")
    print(f"  residual line after derotation |Hz|: median {np.median(res):.0f}, max {np.max(res):.0f}")
    print(f"  [{'PASS' if same_yz else 'FAIL'}] Y and Z identical to the primary set")
    print(f"  [{'PASS' if worst < 1e-5 else 'FAIL'}] |x| identical per sample (max diff {worst:.1e})")
    return 0 if same_yz and worst < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
