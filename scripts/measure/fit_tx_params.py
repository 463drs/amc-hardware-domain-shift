"""Fit the make_tx.py SPEC entries that the RadioML paper does not specify, against the
dataset itself. Offline -- no hardware. Prints values to paste back into make_tx.py.

Everything here is measured on short signals pushed through the SAME front end validate_tx.py
uses (remove f_off, decimate to 1.024 MS/s, 1024-sample frames, Hann-averaged PSD).

CAVEAT that applies to every number below: RadioML frames contain a channel as well as a
signal (Table I -- frequency-selective fading, carrier offset, symbol-rate offset). Fading
reshapes both the averaged PSD and the envelope statistics, so no measurement here can
separate a signal parameter from the channel it arrived through.

  alpha   SENSITIVITY CHECK ONLY, not a fit: make_tx.RRC_ALPHA is taken from the paper, which
          states alpha directly. This sweep shows how far the comparison metrics move across
          the range, and that they disagree about where the optimum is -- which is the point.
          Restricted to ALPHA_CLASSES, where PAPR is set by the pulse alone; APSK/QAM ring and
          grid geometry is still unverified and would leak into the estimate.
  analog  per-family comb + modulation parameters. Scored on TWO criteria that must BOTH hold:
          Wasserstein-1 distance between the TX and RadioML power-vs-frequency distributions
          (a dB residual is useless here -- the analog spectra are narrow, so it is dominated
          by RadioML's noise floor), and per-frame PAPR. W1 alone constrains only where the
          power sits in frequency, and says nothing about the envelope that produced it.
  offset  is RadioML's +32 kHz on FM and AM-DSB a class property or a CFO artefact of the
          particular frames sampled? Re-measures the median frequency across SNR slices and
          frame offsets: a class property stays put, an artefact moves.
  apsk    128APSK ring radii, via a single exponent matched to RadioML's per-frame PAPR.

  python scripts/measure/fit_tx_params.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_tx as mk
from src.config import resolve_data_path
from src.data import read_labels_and_snr
from validate_tx import _DEFAULT_DATA, load_radioml

N_FIT = 1 << 18            # TX samples per trial -> 128 frames after decimation
N_FRAMES = 128
ANALOG = ("AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM")
BIN_KHZ = mk.FS_RX / mk.FRAME_LEN / 1e3
# Constellation geometry is fixed and verified for these, so their PAPR moves with the pulse
# and nothing else. The APSK/QAM classes still carry "TODO: verify" geometry.
ALPHA_CLASSES = ("OOK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK")
# A fit is only accepted when BOTH criteria hold.
W1_OK_KHZ = 2.0
PAPR_OK_DB = 1.0
# An off-DC median counts as a class property only if it varies less than this across cells.
SHIFT_STABLE_KHZ = 2.0
# Scalarization for the search, in kHz per dB; the accept test uses the two separately.
PAPR_WEIGHT = 2.0


def tx_frames(x: np.ndarray) -> np.ndarray:
    """Short TX signal -> the frames validation would cut from it (no f_off applied)."""
    return mk.to_frames(mk.decimate_to_rx(x), N_FRAMES)


def tx_psd(x: np.ndarray) -> np.ndarray:
    """Short TX signal -> the averaged PSD validation would measure."""
    return mk.avg_psd(tx_frames(x))


def psd_rms_db(tx: np.ndarray, rml: np.ndarray) -> float:
    """Bias-removed dB residual over RadioML's occupied band -- validate_tx's psd_rms_db."""
    band = mk.occupied_mask(rml)
    diff = 10 * np.log10(tx[band] + 1e-20) - 10 * np.log10(rml[band] + 1e-20)
    return float(np.sqrt(np.mean((diff - diff.mean()) ** 2)))


def w1_khz(tx: np.ndarray, rml: np.ndarray) -> float:
    """Wasserstein-1 distance between two normalized power spectra, in kHz."""
    return float(np.abs(np.cumsum(tx) - np.cumsum(rml)).sum() * BIN_KHZ)


def median_khz(psd: np.ndarray) -> float:
    """Frequency splitting the power in half -- the shift W1 would want removed."""
    freq = np.fft.fftshift(np.fft.fftfreq(mk.FRAME_LEN, 1 / mk.FS_RX)) / 1e3
    return float(freq[int(np.searchsorted(np.cumsum(psd) / psd.sum(), 0.5))])


def bw_khz(psd: np.ndarray) -> float:
    """99 % occupied width, the same rule validation reports."""
    return float(mk.occupied_mask(psd).sum() * BIN_KHZ)


# Fits

def fit_alpha(rml: dict, rml_papr: dict, seed: int, grid: np.ndarray) -> dict:
    """Sweep alpha over ALPHA_CLASSES and score it four ways. A SENSITIVITY CHECK, not a fit.

    The four criteria disagree, and none of them settles alpha: occupied width and W1 depend
    on alpha, on the symbol rate AND on RadioML's fading, which the averaged PSD cannot
    separate; PAPR is likewise reshaped by frequency-selective fading. What the sweep is good
    for is showing how much each metric moves across the paper's stated range."""
    score = {}
    for alpha in grid:
        rms, bw_err, w1, papr_err = [], [], [], []
        for name in ALPHA_CLASSES:
            rng = np.random.default_rng([seed, mk.MODULATION_CLASSES.index(name)])
            frames = tx_frames(mk.gen_digital(mk.SPEC[name], N_FIT // mk.SPS_TX, rng,
                                              alpha=alpha))
            p = mk.avg_psd(frames)
            rms.append(psd_rms_db(p, rml[name]))
            bw_err.append(bw_khz(p) - bw_khz(rml[name]))
            w1.append(w1_khz(p, rml[name]))
            papr_err.append(mk.frame_papr_db(frames) - rml_papr[name])
        score[round(float(alpha), 3)] = (float(np.mean(rms)), float(np.mean(bw_err)),
                                         float(np.mean(w1)), float(np.mean(papr_err)))
    return score


def _analog_trial(mod: str, src: dict, carrier: float, depth: float, fdev: float,
                  seed: int, name: str, shift_khz: float = 0.0) -> np.ndarray:
    spec = {"mod": mod, "carrier": carrier, "depth": depth, "fdev_hz": fdev,
            "f_shift_hz": shift_khz * 1e3}
    rng = np.random.default_rng([seed, mk.MODULATION_CLASSES.index(name)])
    return mk.gen_analog(spec, N_FIT, rng, src=src)


def _analog_score(mod: str, src: dict, carrier: float, depth: float, fdev: float,
                  seed: int, name: str, shift_khz: float, rml: np.ndarray,
                  rml_papr: float) -> tuple[float, float]:
    """(W1 kHz, PAPR error dB) for one candidate against one class."""
    frames = tx_frames(_analog_trial(mod, src, carrier, depth, fdev, seed, name, shift_khz))
    return w1_khz(mk.avg_psd(frames), rml), mk.frame_papr_db(frames) - rml_papr


def fit_analog(family: str, classes: tuple, rml: dict, rml_papr: dict, seed: int,
               shift: dict) -> tuple[dict, float, float]:
    """Grid-search one analog family against its observed spectra AND envelope statistics.

    W1 alone pins only where the power sits in frequency; two very different modulations can
    share a spectrum and differ completely in envelope, so PAPR is scored alongside it and a
    candidate is accepted only if both criteria pass."""
    win = ["flat", "hann"]
    if family == "dsb":                # RadioML's DSB is a sharp line pair at +-31 kHz over a
        space = itertools.product(["dsb"], [0.0], [1.0], [0.0],   # pedestal, so the band may
                                  [8e3, 18e3, 24e3, 28e3, 30e3, 31e3],   # be very narrow
                                  [32e3, 33e3, 34e3, 36e3, 42e3, 56e3],
                                  [-0.25, 0.0, 0.25], win)
    elif family == "ssb":              # both one- and two-sided are tried: the name is not
        space = itertools.product(["ssb", "dsb"], [1.0],    # evidence for either
                                  [0.02, 0.05, 0.1, 0.2, 0.4], [0.0],
                                  [500.0, 1e3, 2e3],
                                  [3e3, 5e3, 10e3, 20e3, 40e3], [-0.5, 0.0], win)
    else:
        space = itertools.product(["fm"], [0.0], [1.0],
                                  [500.0, 1e3, 2e3, 3e3, 5e3, 8e3, 12e3],
                                  [200.0, 500.0, 1e3, 2e3], [1e3, 2e3, 4e3, 8e3], [-0.5, 0.0],
                                  win)

    best, best_cost = None, np.inf
    for mod, carrier, depth, fdev, f_lo, f_hi, slope, window in space:
        if f_hi <= f_lo:
            continue
        src = {"f_lo_hz": f_lo, "f_hi_hz": f_hi, "slope": slope, "window": window}
        scored = [_analog_score(mod, src, carrier, depth, fdev, seed, c, shift[c],
                                rml[c], rml_papr[c]) for c in classes]
        w1 = float(np.mean([a for a, _ in scored]))
        pe = float(np.mean([abs(b) for _, b in scored]))
        cost = w1 + PAPR_WEIGHT * pe
        if cost < best_cost:
            best, best_cost = (mod, carrier, depth, fdev, src, w1, pe), cost
    mod, carrier, depth, fdev, src, w1, pe = best
    return {"mod": mod, "carrier": carrier, "depth": depth, "fdev_hz": fdev, "src": src}, w1, pe


def offset_stability(data_path: str, labels: tuple, snrs: tuple, offsets: tuple) -> dict:
    """median_khz per analog class over several SNR slices and frame offsets.

    Decides whether RadioML's off-DC analog energy is a CLASS property or just the carrier
    offset of the frames that happened to be sampled: a property of the modulation is the
    same in every cell, a CFO artefact wanders between them."""
    table = {}
    for c in ANALOG:
        row = []
        for snr in snrs:
            for off in offsets:
                frames = load_radioml(data_path, c, snr, N_FRAMES, labels, offset=off)
                row.append(((snr, off), median_khz(mk.avg_psd(frames))))
        table[c] = row
    return table


def fit_apsk(name: str, seed: int) -> list:
    """128APSK radii as r_j ~ (j+1)**p. Returns the whole (p, radii, PAPR) curve.

    PAPR is a necessary, not sufficient, constraint -- many ring layouts share one PAPR --
    so the curve is reported rather than an argmin, and p is chosen under the extra
    requirement that the outer/inner radius ratio stay in the DVB-S2X range (<= ~4)."""
    counts = [n for n, _ in mk.SPEC[name]["const"][1]["rings"]]
    curve = []
    for p in np.linspace(0.2, 1.6, 15):
        radii = (np.arange(1, len(counts) + 1, dtype=float)) ** p
        radii /= radii[0]
        spec = {"const": ("apsk", {"rings": tuple(zip(counts, radii))}), "shape": "rrc"}
        rng = np.random.default_rng([seed, mk.MODULATION_CLASSES.index(name)])
        frames = mk.to_frames(mk.decimate_to_rx(mk.gen_digital(spec, N_FIT // mk.SPS_TX, rng)),
                              N_FRAMES)
        curve.append((float(p), tuple(round(float(r), 2) for r in radii),
                      mk.frame_papr_db(frames)))
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit unspecified make_tx.py SPEC values to RadioML.")
    ap.add_argument("--config", default=None, help="Config name whose data.path supplies the HDF5.")
    ap.add_argument("--path", default=None, help="Explicit RadioML HDF5 path.")
    ap.add_argument("--snr", type=int, default=30, help="RadioML SNR slice used as the reference.")
    ap.add_argument("--seed", type=int, default=1234, help="Seed for the trial signals.")
    args = ap.parse_args()

    data_path = resolve_data_path(args.config, args.path, _DEFAULT_DATA)[0]
    class_idx, snr_all, _, _ = read_labels_and_snr(data_path)
    labels = (class_idx, snr_all)
    names = list(dict.fromkeys(list(ALPHA_CLASSES) + list(ANALOG) + ["128APSK"]))
    rml_frames = {n: load_radioml(data_path, n, args.snr, N_FRAMES, labels) for n in names}
    rml = {n: mk.avg_psd(f) for n, f in rml_frames.items()}
    rml_papr = {n: mk.frame_papr_db(f) for n, f in rml_frames.items()}
    freq = np.fft.fftshift(np.fft.fftfreq(mk.FRAME_LEN, 1 / mk.FS_RX)) / 1e3

    print("=== RadioML analog spectra as observed (the names are not descriptive) ===")
    print(f"{'class':<11}{'99% bw kHz':>11}{'peak kHz':>10}{'median kHz':>12}"
          f"{'PAPR dB':>9}{'upper/lower dB':>16}")
    for c in ANALOG:
        p = rml[c]
        half = mk.FRAME_LEN // 2
        asym = 10 * np.log10(p[half:].sum() / p[:half].sum())
        print(f"{c:<11}{bw_khz(p):>11.1f}{freq[int(np.argmax(p))]:>10.1f}{median_khz(p):>12.1f}"
              f"{rml_papr[c]:>9.2f}{asym:>16.1f}")

    print("\n=== is the off-DC analog energy a class property or a CFO artefact? ===")
    snrs, offsets = (30, 20, 10), (0, 128, 256)
    table = offset_stability(data_path, labels, snrs, offsets)
    head = "".join(f"{f'{snr}dB/{off}':>11}" for snr in snrs for off in offsets)
    print(f"{'class':<11}{head}{'spread':>9}")
    stable = {}
    for c, row in table.items():
        vals = [v for _, v in row]
        spread = max(vals) - min(vals)
        stable[c] = spread
        print(f"{c:<11}" + "".join(f"{v:>11.1f}" for v in vals) + f"{spread:>9.1f}")
    print(f"  a class property holds its median across every cell; a CFO artefact wanders."
          f"  (threshold {SHIFT_STABLE_KHZ} kHz)")
    # Only a median that survives every cell is treated as belonging to the modulation and
    # carried into SPEC as f_shift_hz; anything that wanders is the frames' own carrier offset.
    shift = {c: (median_khz(rml[c]) if stable[c] <= SHIFT_STABLE_KHZ else 0.0) for c in ANALOG}
    for c in ANALOG:
        if shift[c]:
            print(f"    {c}: keeping f_shift_hz = {shift[c] * 1e3:.0f}")

    print(f"\n=== alpha sensitivity over {len(ALPHA_CLASSES)} classes "
          f"(NOT a fit -- make_tx uses the paper's {mk.RRC_ALPHA}) ===")
    score = fit_alpha(rml, rml_papr, args.seed, np.concatenate(
        (np.arange(0.01, 0.20, 0.02), np.arange(0.20, 0.96, 0.05))))
    print(f"  {'alpha':>6}{'psd_rms dB':>12}{'bw error kHz':>14}{'W1 kHz':>9}"
          f"{'PAPR err dB':>13}")
    for a in sorted(score):
        rms, bw, w1, pe = score[a]
        tag = "  <- min rms" if a == min(score, key=lambda k: score[k][0]) else ""
        tag += "  <- min |bw err|" if a == min(score, key=lambda k: abs(score[k][1])) else ""
        tag += "  <- min W1" if a == min(score, key=lambda k: score[k][2]) else ""
        tag += "  <- min |PAPR err|" if a == min(score, key=lambda k: abs(score[k][3])) else ""
        tag += "  <- in use" if abs(a - mk.RRC_ALPHA) < 1e-9 else ""
        print(f"  {a:>6.2f}{rms:>12.3f}{bw:>14.1f}{w1:>9.2f}{pe:>13.2f}{tag}")

    print("\n=== analog families (accepted only if W1 <= "
          f"{W1_OK_KHZ} kHz AND |PAPR err| <= {PAPR_OK_DB} dB) ===")
    for fam, classes in (("dsb", ("AM-DSB-WC", "AM-DSB-SC")),
                         ("ssb", ("AM-SSB-WC", "AM-SSB-SC")), ("fm", ("FM",))):
        fit, w1, pe = fit_analog(fam, classes, rml, rml_papr, args.seed, shift)
        src = fit["src"]
        verdict = "ACCEPT" if w1 <= W1_OK_KHZ and pe <= PAPR_OK_DB else "REJECT"
        print(f"  {fam}: mod={fit['mod']} carrier={fit['carrier']} depth={fit['depth']} "
              f"fdev={fit['fdev_hz']:.0f} Hz   [{verdict}]")
        print(f"       src = {{'f_lo_hz': {src['f_lo_hz']:.0f}, "
              f"'f_hi_hz': {src['f_hi_hz']:.0f}, 'slope': {src['slope']}, "
              f"'window': '{src['window']}'}}")
        for c in classes:
            cw1, cpe = _analog_score(fit["mod"], src, fit["carrier"], fit["depth"],
                                     fit["fdev_hz"], args.seed, c, shift[c], rml[c],
                                     rml_papr[c])
            trial = _analog_trial(fit["mod"], src, fit["carrier"], fit["depth"],
                                  fit["fdev_hz"], args.seed, c, shift[c])
            print(f"       {c:<11} W1 {cw1:6.2f} kHz   PAPR err {cpe:+6.2f} dB   "
                  f"bw tx/rml {bw_khz(tx_psd(trial)):.1f}/{bw_khz(rml[c]):.1f} kHz")

    print("\n=== 128APSK ring radii: PAPR vs radius exponent ===")
    print(f"  RadioML per-frame PAPR = {rml_papr['128APSK']:.2f} dB")
    for pp, radii, papr in fit_apsk("128APSK", args.seed):
        print(f"  p {pp:.2f}  outer/inner {radii[-1]:5.2f}  PAPR {papr:5.2f} dB  "
              f"delta {papr - rml_papr['128APSK']:+5.2f}  radii {radii}")


if __name__ == "__main__":
    main()
