"""Generate HackRF TX files for the 24 RadioML 2018.01A classes.

The signal definition follows RadioML; only the hardware path differs. RadioML declares
f_s = 1.024 MS/s at sps = 8 -> 128 kBd; the HackRF runs at 2.048 MS/s, so sps = 16 here and
the symbol rate is unchanged. Every file is a seamless loop for `hackrf_transfer -R`.

Also holds the DSP primitives and the RX-side front end (decimate, PSD, occupied band) that
validate_tx.py and fit_tx_params.py share, so all three measure a signal the same way.

Two properties of the output that affect how a capture may be read:

  * Every file is peak-normalized to the same backoff, so per-class MEAN power varies with
    PAPR -- currently a 12.3 dB spread, from 0.0 dB (FM and GMSK, constant-envelope) to
    12.3 dB (AM-DSB-SC). One fixed `hackrf_transfer -x` therefore does NOT give a fixed SNR
    across classes: the mean power differs by that much, so at constant noise the per-class
    SNR does too. SNR must be measured per captured file, never assumed from the TX gain.
    The same spread sets snr_tx_ceiling_db below, which tracks it (47.2 dB at the worst PAPR
    against 70.8 dB for the constant-envelope classes).
  * AM-SSB-WC/SC are pairwise identical by construction, and so are AM-DSB-WC/SC: RadioML's
    own WC/SC pairs are spectrally indistinguishable (see ANALOG_SRC), so they are generated
    from the same parameters and differ only by seed. Per-class recall for those four is only
    meaningful PER PAIR -- a classifier separating WC from SC would be reading the seed.

  python scripts/measure/make_tx.py --out-dir tx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import MODULATION_CLASSES

FS_TX = 2_048_000          # HackRF sample rate
SPS_TX = 16                # 2.048 MS/s / 128 kBd -- same symbol rate as RadioML's sps = 8
SYMBOL_RATE = FS_TX // SPS_TX
DURATION_S = 2.0
RRC_SPAN = 16              # symbols; tails are below -80 dB at alpha = 0.25
GMSK_SPAN = 4              # symbols; the Gaussian frequency pulse is negligible beyond +-2 T
MOD_INDEX = 0.5            # GMSK is h = 1/2 CPM by definition

# RX-side front end, shared with validate_tx.py: RadioML's own rate and frame length.
FS_RX = FS_TX // 2
SPS_RX = SPS_TX // 2
FRAME_LEN = 1024
OCC_FRAC = 0.99            # occupied band = shortest window holding this fraction of the power

# Pulse shape: O'Shea 2018 (arXiv:1712.04578) Sec. III -- "Digital signals are shaped with a
# root-raised cosine pulse shaping filter with a range of roll-off values (alpha)"; Table I
# gives alpha ~ U(0.1, 0.4). One file needs one number, so: the stated distribution mean.
#
# NOT fitted to the dataset, deliberately. RadioML frames carry a channel as well as a signal
# (Table I: frequency-selective fading H, carrier offset, symbol-rate offset), and fading
# reshapes BOTH the averaged PSD and the envelope statistics. So neither an occupied-width
# match nor a PAPR match can separate alpha from the symbol rate from the channel -- any
# "fitted alpha" read off an averaged PSD is really a fit to alpha x Rs x fading combined, and
# the three are not identifiable from that measurement. The paper states alpha directly, which
# the dataset cannot; the paper wins.
#
# fit_tx_params.py still sweeps alpha, as a SENSITIVITY CHECK only: it shows how far the
# comparison metrics move across the range, not which value is correct.
RRC_ALPHA = 0.25

# Modulating source for the analog classes. The paper's Fig. 2 shows only an unspecified
# "Audio Source", so this is a deterministic band-limited tone comb: exactly periodic over the
# file (every tone sits on an integer FFT bin) and zero-mean, which keeps the loop seamless.
#
# DATASET PROPERTY, not a bug here: RadioML's analog class NAMES do not describe their
# spectra, as measured off the frames by fit_tx_params.py:
#   AM-DSB-WC/SC  67 kHz wide, symmetric, a sharp line pair at +-31 kHz over a pedestal with
#                 a NOTCH at DC -- suppressed carrier, despite "WC". The two are alike.
#   AM-SSB-WC/SC  near-tonal: 99 % of the power inside 3 kHz, dominated by a carrier line
#                 (despite "SC") and only 7 dB of sideband asymmetry, so it is fitted as
#                 two-sided ("mod": "dsb") -- the name is not evidence for single-sideband.
#   FM            11 kHz wide and parked 32 kHz ABOVE DC (56 dB one-sided), not centred.
# O'Shea 2018 Sec. V-E independently reports WC/SC confusion as a top error source, which is
# what spectrally identical pairs would produce.
#
# The off-DC energy was then tested for whether it BELONGS to the class or is merely the
# carrier offset of the frames that happened to be sampled, by re-measuring the median
# frequency at SNR 30/20/10 dB x three disjoint frame slices. The answer differs by family:
#   FM       +32.0 kHz in all nine cells, spread 0.0 kHz -- a class property, kept as f_shift_hz.
#   AM-DSB   wanders across a 50 kHz spread (+26 to -24 kHz) -- a CFO artefact of whichever
#            frames were drawn, NOT a class property, so no shift is applied and none should be.
#
# Each family is grid-fitted on TWO criteria that must both hold: Wasserstein-1 spectral
# distance AND per-frame PAPR. W1 alone fixes only where the power sits in frequency and says
# nothing about the envelope that produced it. Residuals are in the trailing comments.
ANALOG_SRC = {
    "dsb": {"f_lo_hz": 31_000.0, "f_hi_hz": 32_000.0,   # W1 0.50 kHz, PAPR +0.82 dB
            "slope": -0.25, "window": "hann"},
    "ssb": {"f_lo_hz": 2_000.0, "f_hi_hz": 3_000.0,     # W1 0.01 kHz, PAPR -0.17 dB
            "slope": 0.0, "window": "hann"},
    "fm":  {"f_lo_hz": 500.0, "f_hi_hz": 1_000.0,       # W1 0.13 kHz, PAPR -0.72 dB
            "slope": 0.0, "window": "hann"},
}

# One spec per class, in the RadioML class order (src.data.MODULATION_CLASSES).
# "const": constellation family + args; "shape": pulse. Analog classes carry "mod" (how the
# carrier is modulated) and "src" (which ANALOG_SRC comb feeds it) separately, so a class can
# be given the spectrum RadioML actually has rather than the one its name implies.
SPEC: dict[str, dict] = {
    # --- linear digital: RRC(RRC_ALPHA) per O'Shea 2018 Sec. III + Table I -----------------
    # ASK/OOK levels are unipolar {0..M-1}: RadioML's 4ASK/8ASK carry a strong discrete
    # carrier line at DC, which symmetric PAM cannot produce.  # verified against the data
    "OOK":       {"const": ("ask", {"m": 2}),                   "shape": "rrc"},
    "4ASK":      {"const": ("ask", {"m": 4}),                   "shape": "rrc"},
    "8ASK":      {"const": ("ask", {"m": 8}),                   "shape": "rrc"},
    "BPSK":      {"const": ("psk", {"m": 2}),                   "shape": "rrc"},
    "QPSK":      {"const": ("psk", {"m": 4, "rot": np.pi / 4}), "shape": "rrc"},
    "8PSK":      {"const": ("psk", {"m": 8}),                   "shape": "rrc"},
    "16PSK":     {"const": ("psk", {"m": 16}),                  "shape": "rrc"},
    "32PSK":     {"const": ("psk", {"m": 32}),                  "shape": "rrc"},
    # APSK ring populations are the DVB-S2 / S2X ones. Their radii are code-rate dependent
    # there and the paper names none, so 16/32APSK use the usual DVB-S2 gamma and the
    # 64/128APSK radii are picked for roughly uniform point density.  # TODO: verify
    # 128APSK is 0.7 dB below RadioML's per-frame PAPR. fit_tx_params.py shows closing that
    # needs an ~11:1 outer/inner ring ratio, far outside any DVB-S2X layout, so the gap is
    # NOT attributable to the radii and they are left alone.
    "16APSK":    {"const": ("apsk", {"rings": ((4, 1.0), (12, 2.85))}), "shape": "rrc"},
    "32APSK":    {"const": ("apsk", {"rings": ((4, 1.0), (12, 2.84), (16, 5.27))}),
                  "shape": "rrc"},
    "64APSK":    {"const": ("apsk", {"rings": ((8, 1.0), (16, 2.0), (20, 3.0), (20, 4.0))}),
                  "shape": "rrc"},
    "128APSK":   {"const": ("apsk", {"rings": ((6, 1.0), (18, 2.0), (32, 3.2), (36, 4.0),
                                               (36, 4.8))}), "shape": "rrc"},
    # Square QAM for even log2(M), the standard cross constellation for odd.  # TODO: verify
    "16QAM":     {"const": ("qam", {"m": 16}),  "shape": "rrc"},
    "32QAM":     {"const": ("qam", {"m": 32}),  "shape": "rrc"},
    "64QAM":     {"const": ("qam", {"m": 64}),  "shape": "rrc"},
    "128QAM":    {"const": ("qam", {"m": 128}), "shape": "rrc"},
    "256QAM":    {"const": ("qam", {"m": 256}), "shape": "rrc"},
    # --- analog: fitted to the observed spectrum per family, see ANALOG_SRC ----------------
    # The WC/SC pairs share parameters because RadioML's do not differ spectrally; only the
    # per-class seed makes the two files different realizations of the same process.
    "AM-SSB-WC": {"mod": "dsb", "src": "ssb", "carrier": 1.0, "depth": 0.2},
    "AM-SSB-SC": {"mod": "dsb", "src": "ssb", "carrier": 1.0, "depth": 0.2},
    "AM-DSB-WC": {"mod": "dsb", "src": "dsb", "carrier": 0.0, "depth": 1.0},
    "AM-DSB-SC": {"mod": "dsb", "src": "dsb", "carrier": 0.0, "depth": 1.0},
    "FM":        {"mod": "fm",  "src": "fm",  "fdev_hz": 5_000.0, "f_shift_hz": 32_000.0},
    # --- CPM / offset ----------------------------------------------------------------------
    "GMSK":      {"const": ("psk", {"m": 2}), "shape": "gmsk", "bt": 0.3},  # TODO: verify
    "OQPSK":     {"const": ("psk", {"m": 4, "rot": np.pi / 4}), "shape": "rrc", "offset": True},
}

DIGITAL = tuple(c for c in MODULATION_CLASSES if "mod" not in SPEC[c])
# Linear digital classes: the RRC-shaped ones, excluding GMSK (CPM) and OQPSK (staggered).
LINEAR = tuple(c for c in DIGITAL if SPEC[c]["shape"] == "rrc" and not SPEC[c].get("offset"))


# Constellations

def _qam_points(m: int) -> np.ndarray:
    """Square QAM for even log2(M); the standard cross constellation (32, 128, ...) for odd."""
    if int(round(np.log2(m))) % 2 == 0:
        s = int(round(np.sqrt(m)))
        lv = np.arange(-(s - 1), s, 2, dtype=float)
        return (lv[:, None] + 1j * lv[None, :]).ravel()
    k = int(round(np.log2(m / 32) / 2))            # cross QAM: side 6*2^k, 2^k-square corners
    s, c = 6 * 2**k, 2**k
    lv = np.arange(-(s - 1), s, 2, dtype=float)
    i, q = np.meshgrid(lv, lv, indexing="ij")
    corner = (np.abs(i) > lv[s - c - 1]) & (np.abs(q) > lv[s - c - 1])
    return (i + 1j * q)[~corner]


def constellation(name: str, **kw) -> np.ndarray:
    """Unit-average-power constellation points for one family."""
    if name == "ask":
        pts = np.arange(kw["m"], dtype=float).astype(complex)      # unipolar, keeps a DC line
    elif name == "psk":
        pts = np.exp(1j * (2 * np.pi * np.arange(kw["m"]) / kw["m"] + kw.get("rot", 0.0)))
    elif name == "apsk":
        pts = np.concatenate([r * np.exp(1j * (2 * np.pi * np.arange(n) / n + np.pi / n))
                              for n, r in kw["rings"]])
    elif name == "qam":
        pts = _qam_points(kw["m"])
    else:
        raise ValueError(f"unknown constellation family {name!r}")
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


# Pulses and circular filtering

def rrc_taps(alpha: float, sps: int, span: int) -> np.ndarray:
    """Unit-energy root-raised-cosine taps, span symbols wide, sampled at sps."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = ((np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha)))
             / (np.pi * t * (1 - (4 * alpha * t) ** 2)))
    h[t == 0] = 1 - alpha + 4 * alpha / np.pi
    sing = np.abs(np.abs(4 * alpha * t) - 1) < 1e-9          # removable pole at t = T/(4 alpha)
    h[sing] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
    return h / np.sqrt(np.sum(h ** 2))


def gauss_taps(bt: float, sps: int, span: int) -> np.ndarray:
    """GMSK frequency pulse (Gaussian-filtered rectangle), normalized to unit area."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    a = 2 * np.pi * bt / np.sqrt(np.log(2)) / np.sqrt(2)     # Q(x) = erfc(x / sqrt2) / 2
    g = 0.5 * (erfc(a * (t - 0.5)) - erfc(a * (t + 0.5)))
    return g / g.sum()


def circ_filter(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Circular convolution with a centred odd-length filter, so the output stays periodic."""
    half = len(taps) // 2
    padded = np.concatenate((x[-half:], x, x[:half]))
    return fftconvolve(padded, taps, mode="valid")


def upsample(sym: np.ndarray, sps: int) -> np.ndarray:
    """Symbols -> impulse train at sps samples per symbol."""
    out = np.zeros(sym.size * sps, dtype=complex)
    out[::sps] = sym
    return out


# Baseband generators

def gen_gmsk(spec: dict, n_sym: int, rng: np.random.Generator) -> np.ndarray:
    """h = 1/2 CPM with a Gaussian frequency pulse; phase is integrated circularly."""
    d = rng.choice(np.array([-1.0, 1.0]), n_sym)
    if int(d.sum()) % 4:                      # total phase = (pi/2)*sum(d) must be a 2*pi
        d[0] = -d[0]                          # multiple; one flip moves the sum by +-2
    freq = circ_filter(upsample(d.astype(complex), SPS_TX),
                       gauss_taps(spec["bt"], SPS_TX, GMSK_SPAN))
    return np.exp(1j * np.pi * MOD_INDEX * np.cumsum(freq.real))


def gen_digital(spec: dict, n_sym: int, rng: np.random.Generator,
                alpha: float | None = None) -> np.ndarray:
    """Random i.i.d. symbols -> circular pulse shaping, so the result is periodic in n_sym."""
    if spec["shape"] == "gmsk":
        return gen_gmsk(spec, n_sym, rng)
    pts = constellation(spec["const"][0], **spec["const"][1])
    sym = pts[rng.integers(0, pts.size, n_sym)]
    taps = rrc_taps(RRC_ALPHA if alpha is None else alpha, SPS_TX, RRC_SPAN)
    if spec.get("offset"):                    # OQPSK: Q delayed by half a symbol
        i = circ_filter(upsample(sym.real.astype(complex), SPS_TX), taps)
        q = circ_filter(upsample(sym.imag.astype(complex), SPS_TX), taps)
        return i + 1j * np.roll(q, SPS_TX // 2)
    return circ_filter(upsample(sym, SPS_TX), taps)


def analog_source(n: int, p: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic tone comb -> (real m, analytic m). Zero-mean and exactly periodic in n.

    One tone on EVERY FFT bin in [f_lo, f_hi], so the comb is a band-limited continuum whose
    shape does not depend on the file length -- fitted parameters transfer between lengths."""
    lo = max(int(round(p["f_lo_hz"] * n / FS_TX)), 1)
    bins = np.arange(lo, int(round(p["f_hi_hz"] * n / FS_TX)) + 1)
    amp = bins.astype(float) ** p["slope"]
    if p.get("window") == "hann":      # smooth hump across the band, not a flat block
        u = (bins - bins.min()) / max(bins.max() - bins.min(), 1)
        amp = amp * (0.5 - 0.5 * np.cos(2 * np.pi * u))
    spec = np.zeros(n, dtype=complex)
    spec[bins] = n * amp * np.exp(2j * np.pi * rng.random(bins.size))
    m_a = np.fft.ifft(spec)                   # positive bins only -> already analytic
    return m_a.real / np.abs(m_a.real).max(), m_a / np.abs(m_a.real).max()


def gen_analog(spec: dict, n: int, rng: np.random.Generator,
               src: dict | None = None) -> np.ndarray:
    """AM (DSB/SSB, with or without carrier) or FM of the synthetic modulating signal."""
    m, m_a = analog_source(n, ANALOG_SRC[spec["src"]] if src is None else src, rng)
    if spec["mod"] == "fm":
        out = np.exp(2j * np.pi * spec["fdev_hz"] * np.cumsum(m) / FS_TX)
    else:
        base = m_a if spec["mod"] == "ssb" else m.astype(complex)
        out = spec["carrier"] + spec["depth"] * base
    # RadioML parks its FM and AM-DSB energy off DC; snapped to whole cycles to stay seamless.
    k = round(spec.get("f_shift_hz", 0.0) * n / FS_TX)
    return out * np.exp(2j * np.pi * k * np.arange(n) / n)


def gen_baseband(name: str, n: int, seed: int) -> np.ndarray:
    """One class's complex baseband at FS_TX, length n, before the frequency shift."""
    spec = SPEC[name]
    rng = np.random.default_rng([seed, MODULATION_CLASSES.index(name)])
    return gen_analog(spec, n, rng) if "mod" in spec else gen_digital(spec, n // SPS_TX, rng)


# RX-side front end, shared with validate_tx.py and fit_tx_params.py

def decimate_to_rx(x: np.ndarray, k_off: int = 0) -> np.ndarray:
    """Remove f_off and halve the rate. f_off is an integer bin and the signal is periodic,
    so a bin roll plus a brickwall keep is exact and free of edge transients."""
    n = x.size
    spec = np.roll(np.fft.fft(x), -k_off)
    m = n // 2
    keep = np.concatenate((spec[:m // 2], spec[-(m - m // 2):])) / 2.0
    return np.fft.ifft(keep)


def to_frames(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Cut a decimated stream into at most n_frames frames of FRAME_LEN samples."""
    take = min(n_frames, x.size // FRAME_LEN)
    return x[:take * FRAME_LEN].reshape(take, FRAME_LEN)


def avg_psd(frames: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Hann-windowed periodogram averaged over frames; normalized to unit total power."""
    win = np.hanning(frames.shape[1])
    p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(frames * win, axis=1), axes=1)) ** 2, axis=0)
    return p / p.sum() if normalize else p


def frame_papr_db(frames: np.ndarray) -> float:
    """Median per-frame PAPR, so one outlier frame does not set the number."""
    peak = np.max(np.abs(frames) ** 2, axis=1)
    return float(np.median(10 * np.log10(peak / np.mean(np.abs(frames) ** 2, axis=1))))


def occupied_mask(psd: np.ndarray, frac: float = OCC_FRAC) -> np.ndarray:
    """Shortest contiguous bin window holding `frac` of the power. Unlike a peak-relative
    threshold this survives discrete carrier lines (ASK, AM-WC) and one-sided SSB spectra."""
    cum = np.concatenate(([0.0], np.cumsum(psd)))
    target = frac * cum[-1]
    lo, best = 0, (0, psd.size - 1)
    for hi in range(psd.size):
        while cum[hi + 1] - cum[lo + 1] >= target:
            lo += 1
        if cum[hi + 1] - cum[lo] >= target and hi - lo < best[1] - best[0]:
            best = (lo, hi)
    mask = np.zeros(psd.size, dtype=bool)
    mask[best[0]:best[1] + 1] = True
    return mask


# File writing

def snr_ceiling_db(ideal: np.ndarray, quant: np.ndarray, k_off: int,
                   n_frames: int = 128) -> float:
    """In-band SNR of the int8 file itself: signal vs quantization + dither error inside the
    signal's own occupied band, measured through the same front end as validation.

    This caps the SNR any real capture of this file can reach. A zero-signal noise capture
    does NOT contain it -- the error only exists while the DAC is driven."""
    sig = to_frames(decimate_to_rx(ideal, k_off), n_frames)
    err = to_frames(decimate_to_rx(quant - ideal, k_off), n_frames)
    p_sig = avg_psd(sig, normalize=False)
    band = occupied_mask(p_sig / p_sig.sum())
    return float(10 * np.log10(p_sig[band].sum()
                               / avg_psd(err, normalize=False)[band].sum()))


def gen_unit(name: str, n: int, k_off: int, seed: int) -> np.ndarray:
    """One class's baseband, shifted to +f_off and scaled to unit peak PER COMPONENT.

    Split out from the quantization so a backoff sweep reuses one pulse-shaping pass."""
    x = gen_baseband(name, n, seed) * np.exp(2j * np.pi * k_off * np.arange(n) / n)
    return x / max(np.abs(x.real).max(), np.abs(x.imag).max())   # int8 clips per component


def make_class(name: str, n: int, k_off: int, backoff: float, seed: int,
               x_unit: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Build one class's int8 I/Q loop plus its manifest entry."""
    x = (gen_unit(name, n, k_off, seed) if x_unit is None else x_unit) * backoff
    papr_db = float(10 * np.log10(np.abs(x).max() ** 2 / np.mean(np.abs(x) ** 2)))

    rng = np.random.default_rng([seed, MODULATION_CLASSES.index(name), 1])

    def tpdf() -> np.ndarray:                 # +-1 LSB triangular dither, as in make_tone.py
        return rng.uniform(-0.5, 0.5, n) + rng.uniform(-0.5, 0.5, n)

    i, q = np.round(x.real + tpdf()), np.round(x.imag + tpdf())
    clipped = int(np.count_nonzero((i < -128) | (i > 127) | (q < -128) | (q > 127)))
    i, q = np.clip(i, -128, 127), np.clip(q, -128, 127)
    iq = np.empty(2 * n, dtype=np.int8)
    iq[0::2], iq[1::2] = i, q

    entry = {"class": name, "spec": SPEC[name], "papr_db": round(papr_db, 2),
             "peak_component": round(float(max(np.abs(x.real).max(), np.abs(x.imag).max())), 2),
             "peak_envelope": round(float(np.abs(x).max()), 2), "clipped": clipped,
             "snr_tx_ceiling_db": round(snr_ceiling_db(x, i + 1j * q, k_off), 2),
             "digital": "mod" not in SPEC[name]}
    return iq, entry


def main() -> None:
    p = argparse.ArgumentParser(description="Generate HackRF TX loops for the 24 RadioML classes.")
    p.add_argument("--out-dir", default="tx", help="Directory for <class>.bin and manifest.json.")
    p.add_argument("--f-off", type=float, default=700_000.0,
                   help="Offset from the LO in Hz; snapped to an integer cycle count per file.")
    p.add_argument("--backoff", type=float, nargs="+", default=[127 * 0.9],
                   help="Peak amplitude in LSB. Several values write one subdirectory each "
                        "(b<value>), reusing a single pulse-shaping pass; digital attenuation "
                        "costs 1 dB of snr_tx_ceiling_db per dB.")
    p.add_argument("--duration", type=float, default=DURATION_S, help="Loop length in seconds.")
    p.add_argument("--seed", type=int, default=1234, help="Base seed; per-class streams derive it.")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Subset of classes (default: all 24).")
    p.add_argument("--zeros", action="store_true",
                   help="Write only zeros.bin -- a silent file of the same length, for the "
                        "zero-signal reference capture. Leaves the class files untouched.")
    args = p.parse_args()

    n = int(round(args.duration * FS_TX))
    if n % SPS_TX:
        raise ValueError(f"{args.duration} s at {FS_TX} S/s is not a whole number of symbols")
    k_off = int(round(args.f_off * n / FS_TX))
    f_off = k_off * FS_TX / n
    names = list(args.classes) if args.classes else list(MODULATION_CLASSES)
    unknown = [c for c in names if c not in SPEC]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; expected any of {list(SPEC)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.zeros:
        # Literal zeros, not dithered: no intentional signal. LO leakage is still radiated
        # while sending it, so it is the noise actually present during a class capture.
        np.zeros(2 * n, dtype=np.int8).tofile(out / "zeros.bin")
        print(f"zeros.bin -> {out}/  ({2 * n / 1e6:.1f} MB, {args.duration} s, "
              f"same length as every class file)")
        return

    print(f"f_off = {f_off:.1f} Hz ({k_off} cycles / {args.duration} s), {n} samples/file, "
          f"alpha = {RRC_ALPHA}, backoff {', '.join(f'{b:.1f}' for b in args.backoff)} LSB")

    manifests = {b: {"fs_hz": FS_TX, "sps": SPS_TX, "symbol_rate_hz": SYMBOL_RATE,
                     "duration_s": args.duration, "n_samples": n, "f_off_hz": f_off,
                     "f_off_cycles": k_off, "backoff_lsb": b, "seed": args.seed,
                     "backoff_db": round(20 * np.log10(b / (127 * 0.9)), 2),
                     "rrc_alpha": RRC_ALPHA, "rrc_span_sym": RRC_SPAN, "analog_src": ANALOG_SRC,
                     "format": "int8 interleaved I/Q, TPDF dithered, seamless loop "
                               "(hackrf_transfer -R)",
                     "classes": []} for b in args.backoff}
    # One directory per backoff when several are asked for; flat for a single one.
    dirs = {b: (out if len(args.backoff) == 1 else out / f"b{b:g}") for b in args.backoff}

    print(f"\n{'class':<11}{'PAPR dB':>9}{'peak':>8}{'clipped':>9}" +
          "".join(f"{f'ceil@{b:g}':>13}" for b in args.backoff))
    for name in names:
        x_unit = gen_unit(name, n, k_off, args.seed)      # one pulse-shaping pass per class
        row = []
        for b in args.backoff:
            iq, entry = make_class(name, n, k_off, b, args.seed, x_unit=x_unit)
            dirs[b].mkdir(parents=True, exist_ok=True)
            iq.tofile(dirs[b] / f"{name}.bin")
            manifests[b]["classes"].append(entry)
            row.append(entry)
        print(f"{name:<11}{row[0]['papr_db']:>9.2f}{row[0]['peak_envelope']:>8.1f}"
              f"{sum(e['clipped'] for e in row):>9d}" +
              "".join(f"{e['snr_tx_ceiling_db']:>13.1f}" for e in row))

    for b, mf in manifests.items():
        (dirs[b] / "manifest.json").write_text(json.dumps(mf, indent=2, default=str),
                                               encoding="utf-8")
        ceil = [c["snr_tx_ceiling_db"] for c in mf["classes"]]
        print(f"\nbackoff {b:6.1f} LSB ({mf['backoff_db']:+5.1f} dB) -> {dirs[b]}/  "
              f"clipped {sum(c['clipped'] for c in mf['classes'])}, "
              f"SNR ceiling {min(ceil):.1f}..{max(ceil):.1f} dB")


if __name__ == "__main__":
    main()
