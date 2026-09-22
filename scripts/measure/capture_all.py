"""Capture all 24 TX classes over an SNR ladder. DRIVES THE HARDWARE.

Grid design, from the measured sweep (QPSK, backoff 114.3 LSB, RTL -g 25.4, 433.9 MHz):

    SNR ~= 1.07 * x - 2.0 dB      over the whole -x 0..47 range, slope 1.09 dB/dB

so the HackRF TX VGA alone reaches -2.0 .. +48.2 dB. Digital backoff is therefore only
needed BELOW -2 dB, and is used only there: every dB of digital attenuation costs a dB of
snr_tx_ceiling_db, so the rung with the higher ceiling wins wherever the two overlap.

    SNR >= -2 dB   backoff 114.3 (ceiling 50.9 dB), -x solves the model
    SNR <  -2 dB   -x 0, backoff attenuated by the shortfall (ceiling 50.9 dB - that)

At a fixed -x the per-class SNR still varies with PAPR (a 12.3 dB spread, since the files are
peak-normalized), so each capture is labelled by the -x used and the SNR must be MEASURED per
file with check_capture.py. --equalize-papr instead trims -x per class so every class lands on
the same rung; it is off by default because it varies the HackRF drive level class by class.

  python scripts/measure/capture_all.py --out captures --snr 28 20 10 0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_capture import RML_FRAMES, measure
from make_tx import avg_psd
from src.data import MODULATION_CLASSES
from validate_tx import load_tx

# Denser where the primary metric lives. RadioML runs to -20 dB, but a model is at chance
# there, so the ladder stops at -10; below 0 dB is coarse, 0..+20 is every 2 dB.
DEFAULT_SNR = (-10, -6, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28)

HACKRF = Path("D:/programs/anaconda/envs/sdr/Library/bin")
RTL = Path("E:/x64/rtl_sdr.exe")
TX_CENTRE_HZ = 433_000_000
F_SHIFT_HZ = 200_000           # RX tuned this far above the signal, keeping DC off it
FS_TX, FS_RX = 2_048_000, 1_024_000
RX_GAIN = 25.4
SNR_SLOPE, SNR_INTERCEPT = 1.07, -2.0     # measured; see the module docstring
X_MIN, X_MAX = 0, 44           # 47 puts the RX at -0.1 dBFS, too close to clipping
REF_BACKOFF = 127 * 0.9
SETTLE_S = 2.0                 # TX settling before the RX starts
TX_MARGIN_S = 6.0              # TX runs this much longer than the capture, then exits itself
# A capture this far below its rung recorded silence: the TX passed the streaming check and
# then stalled. Seen once in 384. Such a file is discarded and the capture retried.
DEGENERATE_SNR_DB = -50.0


def rung(snr_db: float) -> tuple[int, float]:
    """(-x, backoff LSB) for one SNR rung, preferring the setting with the higher ceiling."""
    x = round((snr_db - SNR_INTERCEPT) / SNR_SLOPE)
    if x >= X_MIN:
        return min(x, X_MAX), REF_BACKOFF
    # Rounded so the value naming the TX directory is reproducible from the printed grid.
    return X_MIN, round(REF_BACKOFF * 10 ** ((snr_db - (SNR_SLOPE * X_MIN + SNR_INTERCEPT))
                                             / 20), 1)


def wait_device(timeout_s: int = 20) -> None:
    """Block until the HackRF answers, so a previous run cannot collide with this one."""
    for _ in range(timeout_s):
        out = subprocess.run([HACKRF / "hackrf_info.exe"], capture_output=True, text=True)
        if "Serial number" in out.stdout:
            return
        time.sleep(1)
    raise RuntimeError("HackRF did not become available")


def backoff_db(backoff: float) -> float:
    """Digital attenuation relative to the reference backoff; 1 dB here costs 1 dB of ceiling."""
    return round(20 * math.log10(backoff / REF_BACKOFF), 1)


def capture_one(tx_file: Path, out_file: Path, x_gain: int, seconds: float,
                rx_centre_hz: int) -> dict:
    """Transmit tx_file and record `seconds` of it. The TX is bounded by -n so it ends itself;
    nothing is force-killed, which is what wedged the USB stack during the trials."""
    wait_device()
    log = out_file.with_suffix(".txlog")
    n_tx = int((seconds + TX_MARGIN_S) * FS_TX)
    tx = subprocess.Popen(
        [HACKRF / "hackrf_transfer.exe", "-t", str(tx_file), "-f", str(TX_CENTRE_HZ),
         "-s", str(FS_TX), "-x", str(x_gain), "-a", "0", "-R", "-n", str(n_tx)],
        stdout=log.open("w"), stderr=subprocess.STDOUT)
    time.sleep(SETTLE_S)
    if "dBfs" not in log.read_text(errors="ignore"):
        tx.terminate()
        raise RuntimeError(f"TX not streaming for {tx_file.name} at -x {x_gain}")
    subprocess.run([RTL, "-f", str(rx_centre_hz), "-s", str(FS_RX), "-g", str(RX_GAIN),
                    "-n", str(int(seconds * FS_RX)), str(out_file)], capture_output=True)
    tx.wait(timeout=TX_MARGIN_S + 10)
    log.unlink(missing_ok=True)
    return {"file": out_file.name, "x_gain": x_gain, "bytes": out_file.stat().st_size}


def main() -> int:
    p = argparse.ArgumentParser(description="Capture all 24 classes over an SNR ladder.")
    p.add_argument("--out", default="captures", help="Output root; one subdirectory per rung.")
    p.add_argument("--tx-dir", default="tx", help="Directory of <class>.bin at REF_BACKOFF.")
    p.add_argument("--snr", type=float, nargs="+", default=list(DEFAULT_SNR),
                   help="Target SNR rungs in dB.")
    p.add_argument("--first-class", default="QPSK",
                   help="Class run across the whole ladder first, to verify the grid.")
    p.add_argument("--no-notch", action="store_true",
                   help="Measure without notching the spur bins (the .bin files are raw "
                        "either way; notching is a framing-stage choice).")
    p.add_argument("--seconds", type=float, default=3.0, help="Capture length per class.")
    p.add_argument("--classes", nargs="*", default=None, help="Subset (default: all 24).")
    p.add_argument("--equalize-papr", action="store_true",
                   help="Trim -x per class by its PAPR so every class lands on the same rung.")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit.")
    args = p.parse_args()

    tx_dir = Path(args.tx_dir)
    manifest = json.loads((tx_dir / "manifest.json").read_text(encoding="utf-8"))
    papr = {c["class"]: c["papr_db"] for c in manifest["classes"]}
    ceiling = {c["class"]: c["snr_tx_ceiling_db"] for c in manifest["classes"]}
    names = list(args.classes) if args.classes else list(MODULATION_CLASSES)
    ref_papr = papr["QPSK"]        # the sweep that calibrated the model was run on QPSK
    rx_centre = TX_CENTRE_HZ + int(manifest["f_off_hz"]) + F_SHIFT_HZ
    k_off = manifest["f_off_cycles"]
    worst_ceiling = min(ceiling.values())

    print(f"RX centre {rx_centre / 1e6:.3f} MHz (TX {TX_CENTRE_HZ / 1e6:.3f} + f_off "
          f"{manifest['f_off_hz'] / 1e3:.0f}k + shift {F_SHIFT_HZ / 1e3:.0f}k)")
    print(f"{'SNR':>6}{'-x':>5}{'backoff':>9}{'ceiling':>9}{'margin':>8}")
    grid, tight = {}, False
    for target in args.snr:
        x, backoff = rung(target)
        ceil = worst_ceiling + backoff_db(backoff)
        grid[target] = (x, backoff)
        tight |= ceil - target < 10
        print(f"{target:>6.0f}{x:>5d}{backoff:>9.1f}{ceil:>9.1f}{ceil - target:>8.1f} dB")
    if tight:
        print("!! a rung is within 10 dB of its TX ceiling; the quantization floor will show")
    if args.dry_run:
        return 0

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    header = {"tx_centre_hz": TX_CENTRE_HZ, "rx_centre_hz": rx_centre,
              "f_shift_hz": F_SHIFT_HZ, "fs_rx_hz": FS_RX, "rx_gain_db": RX_GAIN,
              "seconds": args.seconds, "snr_model": [SNR_SLOPE, SNR_INTERCEPT],
              "equalize_papr": args.equalize_papr, "notched": not args.no_notch}
    records: list[dict] = []

    def src_dir(backoff: float) -> Path:
        d = tx_dir if backoff == REF_BACKOFF else Path(f"{tx_dir}_b{backoff:g}")
        if not d.exists():
            raise FileNotFoundError(f"{d} missing; run make_tx.py --backoff {backoff:g}")
        return d

    def flush() -> None:      # rewritten after every capture, so a long run survives a crash
        (root / "captures.json").write_text(
            json.dumps(header | {"captures": records}, indent=2), encoding="utf-8")

    # Zeros first, for every rung: each class capture is scored against its own rung's
    # reference, so they all have to exist before any class can be measured.
    for target, (x, backoff) in grid.items():
        out_file = root / f"snr{target:+.0f}" / "zeros.bin"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if not out_file.exists():
            capture_one(src_dir(backoff) / "zeros.bin", out_file, x, args.seconds, rx_centre)
        print(f"  zeros {target:+5.0f} dB  -x {x}")

    # Class-major: the first class walks the whole ladder before any other starts, so a bad
    # grid shows up in one pass instead of after every class has been captured at one rung.
    ordered = ([args.first_class] + [c for c in names if c != args.first_class]
               if args.first_class in names else names)
    for name in ordered:
        for target, (x, backoff) in grid.items():
            out_dir = root / f"snr{target:+.0f}"
            out_file = out_dir / f"{name}.bin"
            xg = x if not args.equalize_papr else \
                max(X_MIN, min(X_MAX, round(x + papr.get(name, ref_papr) - ref_papr)))
            # The band is taken from the reference-backoff TX file at every rung, so it cannot
            # drift as digital attenuation lifts that file's own quantization floor.
            _, tx_frames = load_tx(tx_dir / f"{name}.bin", k_off, RML_FRAMES)
            tx_psd = avg_psd(tx_frames, normalize=False)
            for attempt in (1, 2):
                if not out_file.exists():
                    capture_one(src_dir(backoff) / f"{name}.bin", out_file, xg, args.seconds,
                                rx_centre)
                rec, _ = measure(out_file, out_dir / "zeros.bin", tx_psd,
                                 manifest["f_off_hz"], float(F_SHIFT_HZ), 512,
                                 not args.no_notch)
                if rec["snr_db"] > DEGENERATE_SNR_DB or attempt == 2:
                    break
                print(f"  !! {name} {target:+.0f} dB recorded silence (TX stalled); retrying")
                out_file.unlink()
            rec["attempts"] = attempt
            rec |= {"class": name, "file": str(out_file.relative_to(root)),
                    "target_snr_db": target, "x_gain": xg, "backoff_lsb": backoff,
                    "tx_ceiling_db": round(ceiling[name] + backoff_db(backoff), 1)}
            records.append(rec)
            flush()
            warn = ""
            if rec["clipped_frac"] > 1e-5:
                warn = f"   !! RX CLIPPING {rec['clipped_frac']:.3%}"
            elif rec["snr_db"] < DEGENERATE_SNR_DB:
                warn = "   !! STILL SILENT after retry"
            # An offset of a few dB from target is expected, not a fault: it is the class's
            # PAPR shifting its whole ladder, which is why SNR is measured rather than assumed.
            print(f"  {name:<11}{target:>+5.0f} dB  -x {xg:<3d} measured {rec['snr_db']:>6.1f} dB"
                  f"  notch {rec['notched_bins']:>2d}  clip {rec['clipped_frac']:.3%}{warn}")

    flush()
    clipping = [r for r in records if r["clipped_frac"] > 1e-5]
    silent = [r for r in records if r["snr_db"] < DEGENERATE_SNR_DB]
    good = [r["snr_db"] for r in records if r["snr_db"] >= DEGENERATE_SNR_DB]
    retried = sum(r.get("attempts", 1) > 1 for r in records)
    print(f"\n{len(records)} captures -> {root}/captures.json")
    print(f"  measured SNR span {min(good):.1f} .. {max(good):.1f} dB"
          + (f"   ({retried} retried after a TX stall)" if retried else ""))
    for label, bad in (("RX clipping", clipping), ("silent after retry", silent)):
        if bad:
            print(f"  !! {len(bad)} file(s) with {label}: "
                  + ", ".join(f"{r['class']}@{r['target_snr_db']:+.0f}" for r in bad[:6]))
    return 1 if clipping or silent else 0


if __name__ == "__main__":
    sys.exit(main())
