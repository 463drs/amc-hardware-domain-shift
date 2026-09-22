"""Check one trial RTL-SDR capture of a TX class file. Offline -- reads files, drives nothing.

Geometry this assumes, which is what f_off is for: the RX is tuned to the SIGNAL, i.e.
RX LO = TX LO + f_off. The wanted signal therefore sits at 0 kHz in the RX baseband -- the
same place validate_tx.py puts tx/<class>.bin after removing f_off, and where RadioML frames
sit, so all three overlay without further shifting. Everything the HackRF leaks lands
elsewhere, folded into +-fs/2 by the 1.024 MS/s sampling:

    LO leakage   at -f_off      from the RX LO  ->  alias +324 kHz at f_off = 700 kHz
    TX image     at -2*f_off                    ->  alias -376 kHz
    RTL DC spike at 0 kHz, i.e. ON TOP of the signal -- hence measured over the central bins.

Positions are derived from the manifest's f_off and fs, not hardcoded, and each is then
located as the local maximum within SEARCH_KHZ of that bin so a small tuning error cannot
hide a spur behind the search grid.

Preconditions: both captures taken at the SAME fixed tuner gain, RTL and tuner AGC off.
Exits non-zero if any spur is above SPUR_LIMIT_DBC.

  python scripts/measure/check_capture.py --capture cap_BPSK.bin --noise noise_cap.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_tx import FRAME_LEN, FS_RX, SPEC, avg_psd, occupied_mask, to_frames
from src.config import resolve_data_path
from src.data import read_labels_and_snr
from validate_tx import _DEFAULT_DATA, load_radioml, load_tx

_DEFAULT_OUT_DIR = "outputs/tx_capture"
SKIP_S = 1.0               # discard the tuner / USB start-up transient, as in enob_chain.py
MID_SCALE = 127.5          # rtl_sdr writes unsigned bytes centred here
SEARCH_KHZ = 5.0           # a spur is sought within this much of its predicted bin
SPUR_HALF_BINS = 2         # spur power is integrated over +-this many bins around its peak
SPUR_LIMIT_DBC = -40.0     # PASS/FAIL threshold
NOISE_MARGIN_DB = 3.0      # a spur must clear the noise in its window by this to be called
NOTCH_HALF_BINS = 2        # bins zeroed either side of a spur when --notch is given
N_FRAMES = 512             # capture frames averaged; RadioML has far fewer, see RML_FRAMES
RML_FRAMES = 128


def load_capture(path: Path, skip_s: float = SKIP_S) -> tuple[np.ndarray, float]:
    """uint8 interleaved I/Q -> (complex samples about mid-scale, clipped sample fraction)."""
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw[: raw.size - raw.size % 2][2 * int(skip_s * FS_RX):]
    if raw.size < 2 * FRAME_LEN:
        raise ValueError(f"{path}: fewer than one frame after skipping {skip_s} s")
    clipped = float(np.count_nonzero((raw == 0) | (raw == 255)) / raw.size)
    iq = raw.astype(np.float64) - MID_SCALE
    return iq[0::2] + 1j * iq[1::2], clipped


def alias_hz(f_hz: float, fs_hz: float = FS_RX) -> float:
    """Fold a frequency into the sampled band [-fs/2, fs/2)."""
    return (f_hz + fs_hz / 2) % fs_hz - fs_hz / 2


def shift_bins(f_shift_hz: float) -> int:
    """f_shift in whole FFT bins, refusing anything that would not shift back exactly."""
    k = f_shift_hz * FRAME_LEN / FS_RX
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"--f-shift {f_shift_hz} Hz is {k:.4f} bins; it must be a whole "
                         f"number of bins (a multiple of {FS_RX / FRAME_LEN:.0f} Hz)")
    return int(round(k))


def apply_shift(x: np.ndarray, k_shift: int) -> np.ndarray:
    """Bring a signal captured at -f_shift back to 0 Hz. Exact, since k_shift is whole bins."""
    if not k_shift:
        return x
    return x * np.exp(2j * np.pi * k_shift * np.arange(x.size) / FRAME_LEN)


def spur_dbc(psd: np.ndarray, centre_hz: float, in_band_power: float,
             search_khz: float) -> tuple[float, float]:
    """Locate a spur near centre_hz and return (level in dBc, where it was actually found).

    The search makes the measurement robust to a reference-oscillator offset between the two
    radios: the tone test in capture.md saw 300 kHz land at 298.3 kHz."""
    bin_hz = FS_RX / FRAME_LEN
    centre = int(round(centre_hz / bin_hz)) + FRAME_LEN // 2        # fftshifted index
    half = int(round(search_khz * 1e3 / bin_hz))
    lo, hi = max(centre - half, 0), min(centre + half, FRAME_LEN - 1)
    peak = lo + int(np.argmax(psd[lo:hi + 1]))
    a, b = max(peak - SPUR_HALF_BINS, 0), min(peak + SPUR_HALF_BINS, FRAME_LEN - 1)
    level = 10 * np.log10(psd[a:b + 1].sum() / in_band_power + 1e-30)
    return float(level), float((peak - FRAME_LEN // 2) * bin_hz)


def notch_frames(frames: np.ndarray, centres_hz, half_bins: int = NOTCH_HALF_BINS
                 ) -> tuple[np.ndarray, int]:
    """Zero a few bins at each spur position in every frame; returns (frames, bins notched).

    Deliberately NOT a band-pass to the signal: RadioML frames carry noise across the whole
    +-512 kHz, so narrowing the band would trade one domain mismatch for another."""
    mask = np.zeros(FRAME_LEN, dtype=bool)
    for f_hz in centres_hz:
        centre = int(round(f_hz * FRAME_LEN / FS_RX))        # unshifted FFT index
        for d in range(-half_bins, half_bins + 1):
            mask[(centre + d) % FRAME_LEN] = True
    spec = np.fft.fft(frames, axis=1)
    spec[:, mask] = 0
    return np.fft.ifft(spec, axis=1), int(mask.sum())


def spur_targets(f_off_hz: float, f_shift_hz: float) -> dict[str, float]:
    """Spur positions AFTER the shift back -- the frame the overlay and notching both use.
    The HackRF spurs return to their unshifted aliases; the RTL's DC moves out to +f_shift."""
    return {"HackRF LO leakage": alias_hz(-f_off_hz),
            "HackRF TX image": alias_hz(-2 * f_off_hz),
            "RTL DC spike": alias_hz(f_shift_hz)}


def measure(cap_path: Path, noise_path: Path, tx: np.ndarray, f_off_hz: float,
            f_shift_hz: float, n_frames: int, notch: bool) -> tuple[dict, dict]:
    """Every number this script reports, with no printing or plotting.

    Returns (record, psds): the record is JSON-ready for a capture manifest, psds carries the
    spectra the figure needs. Shared with capture_all.py so a campaign records exactly what a
    single check would have printed."""
    spur_hz = spur_targets(f_off_hz, f_shift_hz)
    k_shift = shift_bins(f_shift_hz)
    cap_x, cap_clipped = load_capture(cap_path)
    noise_x, noise_clipped = load_capture(noise_path)
    cap_f = to_frames(apply_shift(cap_x, k_shift), n_frames)
    noise_f = to_frames(apply_shift(noise_x, k_shift), n_frames)

    # The band comes from the TX reference, NOT from the capture: the 99 % rule needs a
    # signal-dominated spectrum, and on a noisy capture it just selects the whole band.
    freq = np.fft.fftshift(np.fft.fftfreq(FRAME_LEN, 1 / FS_RX)) / 1e3
    band = occupied_mask(tx / tx.sum())
    lo_khz, hi_khz = float(freq[band].min()), float(freq[band].max())

    notched = 0
    if notch:
        inside = [k for k, f in spur_hz.items() if lo_khz * 1e3 <= f <= hi_khz * 1e3]
        if inside:
            raise ValueError(f"refusing to notch inside the signal band "
                             f"({lo_khz:.0f}..{hi_khz:.0f} kHz): {inside}")
        cap_f, notched = notch_frames(cap_f, spur_hz.values())
        noise_f, _ = notch_frames(noise_f, spur_hz.values())
    cap = avg_psd(cap_f, normalize=False)
    noise = avg_psd(noise_f, normalize=False)

    in_band = float(cap[band].sum())
    noise_in_band = float(noise[band].sum())          # SNR is the excess over receiver noise
    snr_db = 10 * np.log10(max(in_band - noise_in_band, 1e-30) / noise_in_band)
    floor_cap = float(np.median(cap[~band]))
    floor_noise = float(np.median(noise[~band]))
    # A spur can only be called if it stands clear of the noise integrated over the same
    # window; at or below that the measurement is noise-limited and says nothing either way.
    noise_equiv = float(10 * np.log10(floor_cap * (2 * SPUR_HALF_BINS + 1) / in_band))

    spurs, failed, undet = [], 0, 0
    for label, centre in spur_hz.items():
        # The DC spike is read off the ZERO-SIGNAL capture: without a shift the wanted signal
        # sits on it, so those bins would be mostly signal. Still referenced to the signal
        # capture's in-band power, so the dBc figures stay comparable across spurs.
        on_noise = label.startswith("RTL")
        dbc, found = spur_dbc(noise if on_noise else cap, centre, in_band,
                              0.0 if on_noise else SEARCH_KHZ)
        if dbc <= SPUR_LIMIT_DBC:
            verdict, ok = "PASS", True
        elif dbc <= noise_equiv + NOISE_MARGIN_DB:
            verdict, ok = "NOT DETECTABLE (noise-limited)", True
            undet += 1
        else:
            verdict, ok = "FAIL", False
        failed += not ok
        spurs.append({"name": label, "dbc": round(dbc, 1), "expected_khz": centre / 1e3,
                      "found_khz": found / 1e3, "verdict": verdict, "ok": ok,
                      "on_zero_signal": on_noise})

    record = {"snr_db": round(float(snr_db), 1),
              "clipped_frac": cap_clipped, "noise_clipped_frac": noise_clipped,
              "band_lo_khz": lo_khz, "band_hi_khz": hi_khz, "notched_bins": notched,
              "floor_dbc_bin": round(float(10 * np.log10(floor_cap / in_band)), 1),
              "noise_floor_dbc_bin": round(float(10 * np.log10(floor_noise / in_band)), 1),
              "floor_delta_db": round(float(10 * np.log10(floor_cap / floor_noise)), 1),
              "spur_noise_floor_dbc": round(noise_equiv, 1), "spurs": spurs,
              "spurs_failed": failed, "spurs_undetectable": undet, "k_shift_bins": k_shift}
    return record, {"cap": cap, "noise": noise, "band": band, "freq": freq}


def plot_capture(name: str, freq: np.ndarray, cap: np.ndarray, noise: np.ndarray,
                 tx: np.ndarray, rml: np.ndarray, band: np.ndarray, spurs: list,
                 out_path: Path) -> None:
    """Two panels: the three-way PSD overlay with the spurs marked, and capture vs noise."""
    def norm_db(psd: np.ndarray) -> np.ndarray:
        return 10 * np.log10(psd / psd[band].sum() + 1e-30)     # each to its own in-band power

    def vs_capture(psd: np.ndarray) -> np.ndarray:
        return 10 * np.log10(psd / cap[band].sum() + 1e-30)     # a COMMON reference

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"{name}  --  RTL-SDR capture vs tx/{name}.bin vs RadioML @ 30 dB SNR")

    ax[0].plot(freq, norm_db(rml), lw=0.9, label="RadioML", color="0.6")
    ax[0].plot(freq, norm_db(tx), lw=0.9, label=f"tx/{name}.bin")
    ax[0].plot(freq, norm_db(cap), lw=0.9, label="capture")
    ax[0].axvspan(freq[band].min(), freq[band].max(), color="0.9", zorder=0)
    ax[0].axhline(SPUR_LIMIT_DBC, color="r", lw=0.8, ls="--")
    for s in spurs:
        ax[0].annotate(f"{s['name']}\n{s['dbc']:.0f} dBc", (s["found_khz"], s["dbc"]),
                       fontsize=7, ha="center", va="bottom",
                       color="r" if s["dbc"] > SPUR_LIMIT_DBC else "0.3")
    ax[0].set(ylabel="dBc (0 = in-band power)", title="PSD overlay, spurs marked")
    ax[0].legend(fontsize=8)

    # Both against the capture's in-band power, so the two noise floors are comparable.
    ax[1].plot(freq, vs_capture(cap), lw=0.9, label="capture")
    ax[1].plot(freq, vs_capture(noise), lw=0.9, label="zero-signal capture")
    ax[1].axvspan(freq[band].min(), freq[band].max(), color="0.9", zorder=0)
    ax[1].set(xlabel="kHz", ylabel="dBc", title="capture vs receiver noise floor")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Check one RTL-SDR capture of a TX class file.")
    p.add_argument("--capture", required=True, help="uint8 I/Q capture of one class.")
    p.add_argument("--noise", required=True, help="uint8 I/Q zero-signal capture, same gain.")
    p.add_argument("--class", dest="cls", default=None,
                   help="Class name; defaults to the capture filename stem.")
    p.add_argument("--tx-dir", default="tx", help="Directory holding <class>.bin + manifest.json.")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="Where the PNG goes.")
    p.add_argument("--config", default=None, help="Config name whose data.path supplies the HDF5.")
    p.add_argument("--path", default=None, help="Explicit RadioML HDF5 path.")
    p.add_argument("--snr", type=int, default=30, help="RadioML SNR slice to overlay.")
    p.add_argument("--n-frames", type=int, default=N_FRAMES, help="Capture frames to average.")
    p.add_argument("--f-shift", type=float, default=0.0,
                   help="How far ABOVE the signal the RX was tuned, in Hz (e.g. 200000 when "
                        "tuning 433.9 MHz for a 433.7 MHz signal). The capture is shifted back "
                        "by this before framing; must be a whole number of FFT bins.")
    p.add_argument("--notch", action="store_true",
                   help="Zero the bins at each spur position after the shift back, and report "
                        "the levels that survive. Refuses to notch inside the signal band.")
    args = p.parse_args()

    capture_path = Path(args.capture)
    name = args.cls or capture_path.stem
    if name not in SPEC:
        raise ValueError(f"{name!r} is not a RadioML class; pass --class explicitly.")

    tx_dir = Path(args.tx_dir)
    manifest = json.loads((tx_dir / "manifest.json").read_text(encoding="utf-8"))
    f_off, k_off = manifest["f_off_hz"], manifest["f_off_cycles"]

    _, tx_frames = load_tx(tx_dir / f"{name}.bin", k_off, RML_FRAMES)
    tx = avg_psd(tx_frames, normalize=False)
    rec, psd = measure(capture_path, Path(args.noise), tx, f_off, args.f_shift,
                       args.n_frames, args.notch)

    data_path = resolve_data_path(args.config, args.path, _DEFAULT_DATA)[0]
    class_idx, snr_all, _, _ = read_labels_and_snr(data_path)
    rml = avg_psd(load_radioml(data_path, name, args.snr, RML_FRAMES, (class_idx, snr_all)),
                  normalize=False)

    print(f"class {name}   f_off {f_off / 1e3:.1f} kHz   {args.n_frames} frames averaged")
    print(f"  clipped samples: capture {rec['clipped_frac']:.3%}, "
          f"zero-signal {rec['noise_clipped_frac']:.3%}")
    print(f"  occupied band  : {rec['band_lo_khz']:.0f} .. {rec['band_hi_khz']:.0f} kHz "
          f"({psd['band'].sum() * FS_RX / FRAME_LEN / 1e3:.0f} kHz wide)")
    if args.notch:
        print(f"  notched        : {rec['notched_bins']} of {FRAME_LEN} bins "
              f"({rec['notched_bins'] / FRAME_LEN:.2%}), {NOTCH_HALF_BINS * 2 + 1} per spur, "
              f"all out of band")
    print(f"  in-band SNR    : {rec['snr_db']:.1f} dB")
    print(f"  noise floor    : capture {rec['floor_dbc_bin']:.1f} dBc/bin, zero-signal "
          f"{rec['noise_floor_dbc_bin']:.1f} dBc/bin (delta {rec['floor_delta_db']:+.1f} dB)")
    if entry := next((c for c in manifest["classes"] if c["class"] == name), None):
        print(f"  TX file ceiling: {entry['snr_tx_ceiling_db']:.1f} dB "
              f"-- the capture SNR cannot exceed this")
    if rec["k_shift_bins"]:
        raw = {s["name"].split()[-1]: alias_hz(s["expected_khz"] * 1e3 - args.f_shift) / 1e3
               for s in rec["spurs"]}
        print(f"  f_shift {args.f_shift / 1e3:.0f} kHz = {rec['k_shift_bins']} bins; raw "
              f"baseband positions "
              + ", ".join(f"{n} {v:+.0f}k" for n, v in raw.items()))

    print(f"\n  noise floor over a {2 * SPUR_HALF_BINS + 1}-bin spur window: "
          f"{rec['spur_noise_floor_dbc']:.1f} dBc -- nothing below this is measurable")
    print(f"  {'spur':<20}{'expected':>10}{'found':>9}{'level':>10}   verdict "
          f"(limit {SPUR_LIMIT_DBC:.0f} dBc)")
    for s in rec["spurs"]:
        print(f"  {s['name']:<20}{s['expected_khz']:>9.0f}k{s['found_khz']:>8.0f}k"
              f"{s['dbc']:>9.1f} dBc   {s['verdict']}"
              f"{' (on zero-signal)' if s['on_zero_signal'] else ''}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_capture(name, psd["freq"], psd["cap"], psd["noise"], tx, rml, psd["band"],
                 rec["spurs"], out / f"capture_{name}.png")
    print(f"\n  figure -> {out / f'capture_{name}.png'}")
    failed, undet = rec["spurs_failed"], rec["spurs_undetectable"]
    if failed:
        print(f"  {failed} SPUR(S) OVER LIMIT")
    elif undet:
        print(f"  no spur over limit, but {undet} could not be measured -- raise the capture "
              f"level and repeat before trusting this")
    else:
        print("  ALL SPURS PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
