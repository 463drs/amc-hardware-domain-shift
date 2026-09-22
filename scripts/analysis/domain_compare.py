"""Descriptive real-vs-RadioML comparison per (class, SNR bin): each figure function returns (Figure, numbers).
Reads the two HDF5 files (and captures/ for raw samples) only; no model, checkpoint or run directory is touched.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy.signal import spectrogram as _stft
from scipy.stats import kurtosis, ks_2samp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "measure"))

from build_real_hdf5 import front_end                                  # noqa: E402
from check_capture import load_capture                                 # noqa: E402
from make_tx import FRAME_LEN, FS_RX, RRC_ALPHA, avg_psd, occupied_mask  # noqa: E402
from validate_tx import symbol_samples                                  # noqa: E402
from src.data import MODULATION_CLASSES, read_labels_and_snr            # noqa: E402
from src.distortions import PercentileReference, Quantize, to_complex  # noqa: E402

REAL_PATH = "data/real_captures_2db_noise-bw.hdf5"
CAPTURES_DIR = "captures"
N_FRAMES = 384             # frames per domain per cell; the synthetic subset holds 384 per cell
BAND_REF_SNR = 30          # the occupied band is read off RadioML at this SNR, whatever bin is shown
GUARD_KHZ = 30.0           # kept clear of the band on each side when reading the noise floor
MAX_CFO_HZ = 15_000.0      # CFO search limit; x8 stays below the 128 kHz symbol-rate lines
CFO_NFFT = 16384
CFO_IQR_OK_HZ = 1000.0     # wider per-frame IQR = the estimator found no line (e.g. real 8PSK, 9 ADC codes)
FREQ_KHZ = np.fft.fftshift(np.fft.fftfreq(FRAME_LEN, 1 / FS_RX)) / 1e3
REAL_COLOR, SYNTH_COLOR = "#eb6834", "#2a78d6"    # dataviz slots 2 and 1, CVD-validated pair
REAL_LABEL, SYNTH_LABEL = "real (RTL-SDR)", "RadioML"
DENSITY_CMAP = "Greys"

# Power whose spectrum carries a line at order*CFO; 0 = spectral centroid (no usable line).
CFO_ORDER = {c: 4 for c in MODULATION_CLASSES} | {
    "OOK": 1, "4ASK": 1, "8ASK": 1, "AM-SSB-WC": 1, "AM-SSB-SC": 1,
    "BPSK": 2, "AM-DSB-WC": 2, "AM-DSB-SC": 2, "8PSK": 8,
    "16PSK": 0, "32PSK": 0, "FM": 0, "GMSK": 0, "OQPSK": 0}


@dataclass(frozen=True)
class Sources:
    """Where the two domains live; relative paths resolve against the repo root."""
    real: str = REAL_PATH
    synthetic: str | None = None       # None -> the real file's reference_file, under data/
    captures: str = CAPTURES_DIR
    quantized: str | None = None       # None -> the synthetic file's _cond-quantization sibling


SOURCES = Sources()


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# Loading

@lru_cache(maxsize=4)
def _real_meta(path: str) -> tuple[dict, str]:
    with h5py.File(_abs(path), "r") as f:
        return json.loads(f.attrs["metadata"]), str(f.attrs["reference_file"])


def synthetic_path(src: Sources = SOURCES) -> Path:
    """The synthetic reference the real file was built against, unless overridden."""
    return _abs(src.synthetic or Path("data") / _real_meta(src.real)[1])


@lru_cache(maxsize=4)
def _synth_labels(path: str) -> tuple[np.ndarray, np.ndarray]:
    cls, snr, _, _ = read_labels_and_snr(path)
    return cls, snr


def real_bins(src: Sources = SOURCES) -> list[int]:
    return list(_real_meta(src.real)[0]["snr"]["usable_bins"])


def _check(cls: str, snr_bin: int, src: Sources) -> None:
    if cls not in MODULATION_CLASSES:
        raise ValueError(f"unknown class {cls!r}")
    if snr_bin not in real_bins(src):
        raise ValueError(f"bin {snr_bin} not in the real file; usable bins {real_bins(src)}")


def _spread(rows: np.ndarray, n: int) -> np.ndarray:
    """n rows evenly spaced over `rows`, so every contributing capture/slice is sampled."""
    if rows.size <= n:
        return rows
    return rows[np.unique(np.linspace(0, rows.size - 1, n).round().astype(int))]


def _unit_power(x: np.ndarray) -> np.ndarray:
    """Per-frame unit mean power, as src.data's unit_power normalizer."""
    return x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True) + 1e-12)


def real_frames(cls: str, snr_bin: int, n_frames: int = N_FRAMES,
                src: Sources = SOURCES) -> np.ndarray:
    """(n, 1024) complex real frames of one cell, as stored (already unit power)."""
    _check(cls, snr_bin, src)
    caps = [c for c in _real_meta(src.real)[0]["captures"]
            if c["class"] == cls and c["bin"] == snr_bin and c["n_frames"]]
    rows = np.concatenate([np.arange(c["row_start"], c["row_start"] + c["n_frames"])
                           for c in caps])
    with h5py.File(_abs(src.real), "r") as f:
        x = f["X"][np.sort(_spread(rows, n_frames)).tolist()]
    return x[..., 0].astype(np.float64) + 1j * x[..., 1]


def synthetic_frames(cls: str, snr_bin: int, n_frames: int = N_FRAMES,
                     src: Sources = SOURCES, normalize: bool = True) -> np.ndarray:
    """(n, 1024) complex RadioML frames of one cell; raw unless `normalize`."""
    path = synthetic_path(src)
    lab, snr = _synth_labels(str(path))
    rows = np.nonzero((lab == MODULATION_CLASSES.index(cls)) & (snr == snr_bin))[0]
    if rows.size == 0:
        raise ValueError(f"{cls} at {snr_bin} dB is not in {path}")
    with h5py.File(path, "r") as f:
        x = f["X"][_spread(rows, n_frames).tolist()]
    x = x[..., 0].astype(np.float64) + 1j * x[..., 1]
    return _unit_power(x) if normalize else x


def _real_adc_cell(cls: str, snr_bin: int, n_frames: int, src: Sources
                   ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """(front-end frames, the same samples as raw ADC values, clipped fraction per capture), all in LSB.
    Raw = uint8 - 127.5 before shift and notch, so its values sit exactly on the ADC codes."""
    _check(cls, snr_bin, src)
    meta = _real_meta(src.real)[0]
    caps = [c for c in meta["captures"]
            if c["class"] == cls and c["bin"] == snr_bin and c["n_frames"]]
    per = math.ceil(n_frames / len(caps))
    fe, raw, clipped = [], [], []
    for c in caps:
        path, m = _abs(src.captures) / c["file"], min(per, c["n_frames"])
        fr, clip = front_end(path, m, meta["front_end"]["k_shift_bins"],
                             meta["notch"]["centres_hz"].values(), meta["notch"]["half_bins"])
        fe.append(fr)
        raw.append(load_capture(path)[0][:m * FRAME_LEN].reshape(m, FRAME_LEN))
        clipped.append(clip)
    return np.concatenate(fe)[:n_frames], np.concatenate(raw)[:n_frames], clipped


def raw_real_frames(cls: str, snr_bin: int, n_frames: int = N_FRAMES,
                    src: Sources = SOURCES) -> tuple[np.ndarray, dict]:
    """Real frames before unit_power (ADC LSB), rebuilt from captures/ by the build's own front end,
    plus the clipped fraction and how many 8-bit I codes those samples span."""
    fe, raw, clipped = _real_adc_cell(cls, snr_bin, n_frames, src)
    return fe, {"clipped_frac_real": float(np.mean(clipped)),
                "adc_codes_used_real": np.unique(raw.real).size}


@lru_cache(maxsize=64)
def _band_cached(cls: str, synth: str) -> np.ndarray:
    return occupied_mask(avg_psd(synthetic_frames(cls, BAND_REF_SNR, src=Sources(synthetic=synth))))


def occupied_band(cls: str, src: Sources = SOURCES) -> np.ndarray:
    """99 % band of the class, from RadioML at BAND_REF_SNR so it does not widen with noise."""
    return _band_cached(cls, str(synthetic_path(src))).copy()


def notch_mask(src: Sources = SOURCES) -> np.ndarray:
    """fftshifted bins the real front end zeroes; excluded from every out-of-band statistic."""
    bins = _real_meta(src.real)[0]["notch"]["bins_khz"]
    return np.isin(np.round(FREQ_KHZ).astype(int), np.round(bins).astype(int))


# Numbers, figure-free

def _db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(p + 1e-20)


def _tilt(f_khz: np.ndarray, y_db: np.ndarray) -> tuple[float, np.ndarray]:
    """Linear fit of y over f -> (dB change across the span of f, fitted line)."""
    slope, icpt = np.polyfit(f_khz, y_db, 1)
    return float(slope * (f_khz.max() - f_khz.min())), slope * f_khz + icpt


def psd_stats(real: np.ndarray, synth: np.ndarray, band: np.ndarray,
              notch: np.ndarray | None = None) -> dict:
    """Bias-removed in-band dB mismatch (rms, max, tilt), each domain's own tilt, and each
    out-of-band floor in dB re the in-band mean per bin (the SNR-definition gap)."""
    pr, ps = avg_psd(real), avg_psd(synth)
    oob = ~_dilate(band, GUARD_KHZ) & (~notch if notch is not None else True)
    diff = _db(pr[band]) - _db(ps[band])
    bias = float(diff.mean())
    f = FREQ_KHZ[band]
    return {"band_lo_khz": float(f.min()), "band_hi_khz": float(f.max()),
            "psd_bias_db": bias,
            "psd_rms_db": float(np.sqrt(np.mean((diff - bias) ** 2))),
            "psd_max_db": float(np.abs(diff - bias).max()),
            "psd_tilt_db": _tilt(f, diff)[0],
            "tilt_real_db": _tilt(f, _db(pr[band]))[0],
            "tilt_synth_db": _tilt(f, _db(ps[band]))[0],
            "floor_real_dbc": float(np.median(_db(pr[oob])) - _db(pr[band].mean())),
            "floor_synth_dbc": float(np.median(_db(ps[oob])) - _db(ps[band].mean()))}


def frame_papr_db(frames: np.ndarray) -> np.ndarray:
    """Per-frame PAPR in dB."""
    p = np.abs(frames) ** 2
    return 10 * np.log10(p.max(axis=1) / p.mean(axis=1))


def papr_stats(real: np.ndarray, synth: np.ndarray) -> dict:
    out = {}
    for tag, x in (("real", real), ("synth", synth)):
        papr, env = frame_papr_db(x), np.abs(_unit_power(x))
        out |= {f"papr_median_{tag}_db": float(np.median(papr)),
                f"papr_p10_{tag}_db": float(np.percentile(papr, 10)),
                f"papr_p90_{tag}_db": float(np.percentile(papr, 90)),
                f"envelope_std_{tag}": float(env.std())}
    out["papr_diff_db"] = out["papr_median_real_db"] - out["papr_median_synth_db"]
    return out


def cfo_per_frame(frames: np.ndarray, order: int, band: np.ndarray | None = None,
                  max_cfo_hz: float = MAX_CFO_HZ) -> np.ndarray:
    """Per-frame CFO in Hz: peak of the x**order spectrum / order, or the band's spectral centroid (order 0)."""
    if order == 0:
        keep = _dilate(band, GUARD_KHZ) if band is not None else np.ones(FRAME_LEN, bool)
        p = np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(FRAME_LEN), axis=1),
                                   axes=1)) ** 2
        floor = np.median(p[:, ~keep], axis=1, keepdims=True) if (~keep).any() else 0.0
        w = np.clip(p[:, keep] - floor, 0, None)
        return (w * FREQ_KHZ[keep] * 1e3).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-30)
    y = frames ** order * np.hanning(FRAME_LEN)
    spec = np.abs(np.fft.fftshift(np.fft.fft(y, CFO_NFFT, axis=1), axes=1)) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(CFO_NFFT, 1 / FS_RX))
    idx = np.nonzero(np.abs(f) <= order * max_cfo_hz)[0]
    k = idx[0] + np.argmax(spec[:, idx], axis=1)
    # Parabolic interpolation on the log peak for sub-bin resolution.
    a, b, c = (np.log(spec[np.arange(len(k)), k + d] + 1e-30) for d in (-1, 0, 1))
    delta = 0.5 * (a - c) / np.where(a - 2 * b + c == 0, -1e-30, a - 2 * b + c)
    return (f[k] + np.clip(delta, -0.5, 0.5) * FS_RX / CFO_NFFT) / order


LINE_ORDERS = (1, 2, 4, 8)
LINE_MIN_DB = 3.0          # averaged x**p peak over its search-window median; noise alone gives < 1 dB


def capture_line(frames: np.ndarray, orders=LINE_ORDERS, max_cfo_hz: float = MAX_CFO_HZ
                 ) -> tuple[float, int, float]:
    """One CFO for a whole capture: |FFT(x**p)|^2 summed over its frames, best order by line
    prominence -> (cfo_hz, order, prominence_db). Averaging finds lines single frames cannot."""
    f = np.fft.fftshift(np.fft.fftfreq(CFO_NFFT, 1 / FS_RX))
    best = (0.0, 0, -np.inf)
    for p in orders:
        spec = np.zeros(CFO_NFFT)
        for i in range(0, len(frames), 256):
            y = frames[i:i + 256] ** p * np.hanning(FRAME_LEN)
            spec += (np.abs(np.fft.fftshift(np.fft.fft(y, CFO_NFFT, axis=1), axes=1)) ** 2).sum(0)
        idx = np.nonzero(np.abs(f) <= p * max_cfo_hz)[0]
        k = int(idx[np.argmax(spec[idx])])
        prom = float(10 * np.log10(spec[k] / np.median(spec[idx])))
        if prom > best[2]:
            a, b, c = np.log(spec[k - 1:k + 2] + 1e-30)
            den = a - 2 * b + c
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den else 0.0
            best = (float((f[k] + delta * FS_RX / CFO_NFFT) / p), p, prom)
    return best


def derotate_frames(frames: np.ndarray, order: int, band: np.ndarray) -> np.ndarray:
    """Remove each frame's estimated CFO, then (order > 0) its carrier phase, mod 2π/order."""
    n = np.arange(frames.shape[1])
    y = frames * np.exp(-2j * np.pi * cfo_per_frame(frames, order, band)[:, None] * n / FS_RX)
    if order:
        y = y * np.exp(-1j * np.angle(np.sum(y ** order, axis=1, keepdims=True)) / order)
    return y


def _dilate(mask: np.ndarray, khz: float) -> np.ndarray:
    n = int(round(khz * 1e3 * FRAME_LEN / FS_RX))
    return np.convolve(mask.astype(int), np.ones(2 * n + 1, int), mode="same") > 0


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def cfo_stats(cfo_real: np.ndarray, cfo_synth: np.ndarray, order: int) -> dict:
    out: dict = {"cfo_method": "centroid" if order == 0 else f"x^{order} line"}
    for tag, v in (("real", cfo_real), ("synth", cfo_synth)):
        q25, q75 = np.percentile(v, [25, 75])
        out |= {f"cfo_median_{tag}_hz": float(np.median(v)),
                f"cfo_iqr_{tag}_hz": float(q75 - q25),
                f"cfo_reliable_{tag}": bool(q75 - q25 <= CFO_IQR_OK_HZ),
                f"cfo_sigma_{tag}_hz": _robust_sigma(v)}
    out["cfo_ks"] = float(ks_2samp(cfo_real, cfo_synth).statistic)
    return out


def noise_stats(real: np.ndarray, synth: np.ndarray, band: np.ndarray,
                notch: np.ndarray, guard_khz: float = GUARD_KHZ) -> tuple[dict, dict]:
    """Out-of-band floor per domain, 0 dB = its median: tilt, edge droop, ripple about the fit."""
    oob = ~_dilate(band, guard_khz) & ~notch
    f = FREQ_KHZ[oob]
    edge = np.abs(f) >= 0.9 * FS_RX / 2e3
    stats, curves = {}, {}
    for tag, x in (("real", real), ("synth", synth)):
        y = _db(avg_psd(x)[oob])
        y -= np.median(y)
        tilt, fit = _tilt(f, y)
        stats |= {f"noise_tilt_{tag}_db": tilt,
                  f"noise_edge_droop_{tag}_db": float(y[edge].mean() - np.median(y[~edge])),
                  f"noise_ripple_{tag}_db": float(np.sqrt(np.mean((y - fit) ** 2)))}
        curves[tag] = (y, fit)
    stats["oob_bins"] = int(oob.sum())
    return stats, {"oob": oob, **curves}


def sample_stats(x: np.ndarray, tag: str) -> dict:
    """Pre-normalization I/Q statistics: DC, gain ratio, I-Q correlation, kurtosis, frame-power spread."""
    i, q = x.real.ravel(), x.imag.ravel()
    rms = float(np.sqrt(np.mean(i ** 2 + q ** 2) / 2))
    frame_rms = np.sqrt(np.mean(np.abs(x) ** 2, axis=1))
    return {f"rms_{tag}": rms,
            f"dc_i_{tag}": float(i.mean() / rms), f"dc_q_{tag}": float(q.mean() / rms),
            f"iq_gain_ratio_{tag}_db": float(20 * np.log10(i.std() / q.std())),
            f"iq_corr_{tag}": float(np.corrcoef(i, q)[0, 1]),
            f"kurtosis_i_{tag}": float(kurtosis(i)), f"kurtosis_q_{tag}": float(kurtosis(q)),
            f"frame_power_spread_{tag}_db": float(np.std(20 * np.log10(frame_rms)))}


# Figures

def _figure(ncols: int, figsize=None, **kw) -> tuple[Figure, np.ndarray]:
    fig = Figure(figsize=figsize or rcParams["figure.figsize"], layout="constrained")
    return fig, np.atleast_1d(fig.subplots(1, ncols, **kw))


def _title(fig: Figure, show: bool, text: str) -> None:
    if show:
        fig.suptitle(text)


def _hist(ax, v, bins, color, label, ls="-") -> None:
    ax.hist(v, bins=bins, density=True, histtype="step", color=color, ls=ls, lw=1.2, label=label)


def psd_overlay(cls: str, snr_bin: int, *, n_frames: int = N_FRAMES, src: Sources = SOURCES,
                figsize=None, title: bool = True) -> tuple[Figure, dict]:
    """Averaged PSD of both domains (left) and their in-band dB difference with its linear fit (right)."""
    real, synth = real_frames(cls, snr_bin, n_frames, src), synthetic_frames(cls, snr_bin, n_frames, src)
    band, notch = occupied_band(cls, src), notch_mask(src)
    nums = psd_stats(real, synth, band, notch) | {"n_real": len(real), "n_synth": len(synth)}
    pr, ps = _db(avg_psd(real)), _db(avg_psd(synth))
    pr[notch] = np.nan                                  # zeroed bins, not a spectrum

    fig, (a0, a1) = _figure(2, figsize, gridspec_kw={"width_ratios": [3, 2]})
    a0.axvspan(nums["band_lo_khz"], nums["band_hi_khz"], color="0.92", lw=0, zorder=0)
    a0.plot(FREQ_KHZ, ps, color=SYNTH_COLOR, label=SYNTH_LABEL)
    a0.plot(FREQ_KHZ, pr, color=REAL_COLOR, ls="--", label=REAL_LABEL)
    a0.set(xlabel="frequency, kHz", ylabel="PSD, dB (unit total power)", xlim=(-512, 512))
    a0.legend(loc="upper right")

    f = FREQ_KHZ[band]
    diff = pr[band] - ps[band] - nums["psd_bias_db"]
    a1.plot(f, diff, color="0.2", lw=0.8, label="real − RadioML")
    a1.plot(f, _tilt(f, diff)[1], color="0.2", ls=":", label=f"fit, {nums['psd_tilt_db']:+.2f} dB")
    a1.axhline(0, color="0.6", lw=0.6)
    a1.set(xlabel="frequency, kHz", ylabel="in-band difference, dB")
    a1.set_title(f"rms {nums['psd_rms_db']:.2f} dB, max {nums['psd_max_db']:.2f} dB", loc="left")
    a1.legend(loc="upper center")
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: averaged PSD ({len(real)} frames each)")
    return fig, nums


def envelope_papr(cls: str, snr_bin: int, *, n_frames: int = N_FRAMES, src: Sources = SOURCES,
                  figsize=None, title: bool = True) -> tuple[Figure, dict]:
    """Per-frame PAPR distribution (left) and pooled envelope |x| at unit power (right)."""
    real, synth = real_frames(cls, snr_bin, n_frames, src), synthetic_frames(cls, snr_bin, n_frames, src)
    nums = papr_stats(real, synth)
    pr, ps = frame_papr_db(real), frame_papr_db(synth)

    fig, (a0, a1) = _figure(2, figsize)
    bins = np.linspace(min(pr.min(), ps.min()), max(pr.max(), ps.max()), 40)
    _hist(a0, ps, bins, SYNTH_COLOR, SYNTH_LABEL)
    _hist(a0, pr, bins, REAL_COLOR, REAL_LABEL, "--")
    a0.set(xlabel="per-frame PAPR, dB", ylabel="density")
    a0.set_title(f"median diff {nums['papr_diff_db']:+.2f} dB", loc="left")
    a0.legend(loc="upper right")

    er, es = np.abs(real).ravel(), np.abs(synth).ravel()
    bins = np.linspace(0, np.percentile(np.concatenate((er, es)), 99.9), 80)
    _hist(a1, es, bins, SYNTH_COLOR, SYNTH_LABEL)
    _hist(a1, er, bins, REAL_COLOR, REAL_LABEL, "--")
    a1.set(xlabel="envelope |x| (unit power)", ylabel="density")
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: PAPR and envelope")
    return fig, nums


def constellation(cls: str, snr_bin: int, *, n_frames: int = N_FRAMES, src: Sources = SOURCES,
                  alpha: float = RRC_ALPHA, derotate: bool = True, figsize=None,
                  title: bool = True) -> tuple[Figure, dict]:
    """RRC matched filter + best-phase symbol-rate sampling, 2-D density per domain.
    `derotate` strips per-frame CFO and carrier phase; False pools Table I's random θ_c into a ring."""
    frames = {"real": real_frames(cls, snr_bin, n_frames, src),
              "synth": synthetic_frames(cls, snr_bin, n_frames, src)}
    if derotate:
        order, band = CFO_ORDER[cls], occupied_band(cls, src)
        frames = {k: derotate_frames(x, order, band) for k, x in frames.items()}
    sym = {k: symbol_samples(x, alpha, matched=True) for k, x in frames.items()}
    sym = {k: s / np.sqrt(np.mean(np.abs(s) ** 2)) for k, s in sym.items()}
    nums: dict = {"derotated": derotate, "n_symbols": len(sym["real"])}
    for k, s in sym.items():
        m = np.abs(s)
        nums |= {f"ring_spread_{k}": float(m.std() / m.mean()),
                 f"m4_coherence_{k}": float(np.abs(np.mean(s ** 4)) / np.mean(m ** 4))}

    lim = float(np.percentile(np.abs(np.concatenate(list(sym.values()))), 99.5)) * 1.05
    edges = np.linspace(-lim, lim, 121)
    fig, axes = _figure(2, figsize, sharex=True, sharey=True)
    for ax, (k, label) in zip(axes, (("synth", SYNTH_LABEL), ("real", REAL_LABEL))):
        ax.hist2d(sym[k].real, sym[k].imag, bins=edges, cmap=DENSITY_CMAP, norm=LogNorm(),
                  rasterized=True)
        ax.set(xlabel="I", aspect="equal")
        ax.set_title(f"{label}: ring spread {nums[f'ring_spread_{k}']:.2f}", loc="left")
        ax.grid(False)
    axes[0].set_ylabel("Q")
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: symbol-rate samples"
                       + (", CFO and phase removed" if derotate else ", as the model sees them"))
    return fig, nums


def sample_histogram(cls: str, snr_bin: int, *, n_frames: int = N_FRAMES,
                     src: Sources = SOURCES, figsize=None, title: bool = True
                     ) -> tuple[Figure, dict]:
    """I/Q value distributions before unit_power (each domain in its own units) and per-frame RMS spread."""
    raw, extra = raw_real_frames(cls, snr_bin, n_frames, src)
    synth = synthetic_frames(cls, snr_bin, n_frames, src, normalize=False)
    nums = sample_stats(raw, "real") | sample_stats(synth, "synth") | extra

    fig, axes = _figure(3, figsize)
    for ax, x, color, label, unit in ((axes[0], synth, SYNTH_COLOR, SYNTH_LABEL, "stored units"),
                                      (axes[1], raw, REAL_COLOR, REAL_LABEL, "ADC LSB")):
        lim = np.percentile(np.abs(np.concatenate((x.real, x.imag), axis=None)), 99.9)
        bins = np.linspace(-lim, lim, 81)
        _hist(ax, x.real.ravel(), bins, color, "I")
        _hist(ax, x.imag.ravel(), bins, color, "Q", ":")
        ax.set(xlabel=f"value, {unit}", yticks=[])
        ax.set_title(f"{label}\nI solid, Q dotted", loc="left")
    axes[0].set_ylabel("density")

    rel = {k: 20 * np.log10(np.sqrt(np.mean(np.abs(x) ** 2, axis=1))) for k, x in
           (("synth", synth), ("real", raw))}
    rel = {k: r - np.median(r) for k, r in rel.items()}
    w = 1.1 * max(np.percentile(np.abs(r), 99.5) for r in rel.values())
    for k, color, label, ls in (("synth", SYNTH_COLOR, SYNTH_LABEL, "-"),
                                ("real", REAL_COLOR, REAL_LABEL, "--")):
        _hist(axes[2], rel[k], np.linspace(-w, w, 41), color, label, ls)
    axes[2].set(xlabel="frame RMS re median, dB", yticks=[])
    axes[2].set_title("frame RMS spread\n(removed by unit_power)", loc="left")
    axes[2].legend(loc="upper left")
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: I/Q values before normalization")
    return fig, nums


def spectrogram(cls: str, snr_bin: int, *, frame: int = 0, src: Sources = SOURCES,
                nperseg: int = 64, figsize=None, title: bool = True) -> tuple[Figure, dict]:
    """STFT of one frame per domain on a shared dB scale; reports in-band power variation over time."""
    band = occupied_band(cls, src)
    x = {"synth": synthetic_frames(cls, snr_bin, frame + 1, src)[frame],
         "real": real_frames(cls, snr_bin, frame + 1, src)[frame]}
    spec: dict = {}
    nums: dict = {"frame": frame}
    for k, v in x.items():
        f, t, s = _stft(v, fs=FS_RX, window="hann", nperseg=nperseg,
                        noverlap=3 * nperseg // 4, return_onesided=False, mode="psd")
        f, s = np.fft.fftshift(f) / 1e3, np.fft.fftshift(s, axes=0)
        inb = (f >= FREQ_KHZ[band].min()) & (f <= FREQ_KHZ[band].max())
        if not inb.any():                              # band narrower than one STFT bin
            inb = np.abs(f - FREQ_KHZ[band].mean()) == np.abs(f - FREQ_KHZ[band].mean()).min()
        nums[f"inband_power_std_{k}_db"] = float(np.std(_db(s[inb].sum(axis=0))))
        spec[k] = (f, t * 1e6, _db(s / s.sum()))

    lo, hi = np.percentile(np.concatenate([v[2].ravel() for v in spec.values()]), [5, 99.9])
    fig, axes = _figure(2, figsize, sharey=True)
    for ax, (k, label) in zip(axes, (("synth", SYNTH_LABEL), ("real", REAL_LABEL))):
        f, t, s = spec[k]
        im = ax.pcolormesh(t, f, s, cmap=DENSITY_CMAP, vmin=lo, vmax=hi,
                           shading="nearest", rasterized=True)
        ax.set(xlabel="time, µs")
        ax.set_title(f"{label}\nband power σ {nums[f'inband_power_std_{k}_db']:.1f} dB", loc="left")
        ax.grid(False)
    axes[0].set_ylabel("frequency, kHz")
    fig.colorbar(im, ax=axes, label="dB (frame total = 0)", shrink=0.9)
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: spectrogram of frame {frame}")
    return fig, nums


def noise_color(cls: str, snr_bin: int, *, n_frames: int = N_FRAMES, src: Sources = SOURCES,
                guard_khz: float = GUARD_KHZ, figsize=None, title: bool = True
                ) -> tuple[Figure, dict]:
    """Out-of-band PSD per domain relative to its own median; a flat line at 0 dB is white noise."""
    real, synth = real_frames(cls, snr_bin, n_frames, src), synthetic_frames(cls, snr_bin, n_frames, src)
    nums, cur = noise_stats(real, synth, occupied_band(cls, src), notch_mask(src), guard_khz)
    f = FREQ_KHZ[cur["oob"]]
    gaps = np.concatenate(([False], np.diff(np.nonzero(cur["oob"])[0]) > 1))

    def broken(y):                      # NaN across excluded bins so no line bridges them
        return np.insert(y.astype(float), np.nonzero(gaps)[0], np.nan)

    fb = broken(f)
    fig, axes = _figure(1, figsize)
    ax = axes[0]
    for k, color, label, ls in (("synth", SYNTH_COLOR, SYNTH_LABEL, "-"),
                                ("real", REAL_COLOR, REAL_LABEL, "--")):
        y, fit = cur[k]
        ax.plot(fb, broken(y), color=color, ls=ls, lw=0.8,
                label=f"{label}: tilt {nums[f'noise_tilt_{k}_db']:+.2f} dB, "
                      f"edge {nums[f'noise_edge_droop_{k}_db']:+.2f} dB")
        ax.plot(fb, broken(fit), color=color, ls=":", lw=1.0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel="frequency, kHz (occupied band ± guard and notches removed)",
           ylabel="dB re out-of-band median", xlim=(-512, 512))
    ax.legend(loc="lower right")
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: out-of-band noise floor")
    return fig, nums


def cfo_estimate(cls: str, snr_bin: int, *, n_frames: int = N_FRAMES, src: Sources = SOURCES,
                 max_cfo_hz: float = MAX_CFO_HZ, figsize=None, title: bool = True
                 ) -> tuple[Figure, dict]:
    """Per-frame CFO histograms vs Table I's N(0, σ_clk); σ_clk is unstated in the paper, so it is fitted to RadioML.
    The centroid (no spectral line) also reads spectral asymmetry and in-band tilt, e.g. RadioML FM's +32 kHz."""
    order, band = CFO_ORDER[cls], occupied_band(cls, src)
    cr = cfo_per_frame(real_frames(cls, snr_bin, n_frames, src), order, band, max_cfo_hz)
    cs = cfo_per_frame(synthetic_frames(cls, snr_bin, n_frames, src), order, band, max_cfo_hz)
    nums = cfo_stats(cr, cs, order)

    fig, axes = _figure(1, figsize)
    ax = axes[0]
    both = np.concatenate((cr, cs)) / 1e3
    lo, hi = np.percentile(both, [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    _hist(ax, cs / 1e3, bins, SYNTH_COLOR, SYNTH_LABEL)
    _hist(ax, cr / 1e3, bins, REAL_COLOR, REAL_LABEL, "--")
    if order:
        sig = nums["cfo_sigma_synth_hz"] / 1e3
        g = np.linspace(bins[0], bins[-1], 400)
        ax.plot(g, np.exp(-0.5 * (g / sig) ** 2) / (sig * np.sqrt(2 * np.pi)), color="0.3",
                ls=":", label=f"Table I N(0, σ_clk), σ_clk fit {sig * 1e3:.0f} Hz")
    ax.set(xlabel="estimated CFO, kHz", ylabel="density")
    ax.set_title(f"{nums['cfo_method']}; median real {_cfo_txt(nums, 'real')}, "
                 f"RadioML {_cfo_txt(nums, 'synth')} Hz", loc="left")
    ax.legend(loc="upper right")
    _title(fig, title, f"{cls}, {snr_bin:+d} dB: per-frame carrier offset")
    return fig, nums


def _cfo_txt(nums: dict, tag: str) -> str:
    """Median CFO, bracketed when the per-frame spread says no line was found."""
    v = f"{nums[f'cfo_median_{tag}_hz']:+.0f}"
    return v if nums[f"cfo_reliable_{tag}"] else f"({v})"


FIGURES = {"psd_overlay": psd_overlay, "envelope_papr": envelope_papr,
           "constellation": constellation, "sample_histogram": sample_histogram,
           "spectrogram": spectrogram, "noise_color": noise_color, "cfo_estimate": cfo_estimate}


# ADC loading

ADC_BITS = 8
ADC_FS_LSB = 2.0 ** (ADC_BITS - 1)   # mid-rise: codes at +-0.5 .. +-127.5 LSB, RTL-SDR and Quantize alike
QUANT_PERCENTILE = 99.9              # the `quantization` condition's full-scale reference


def quantized_path(src: Sources = SOURCES) -> Path:
    """The synthetic `quantization` condition file, row-aligned with the synthetic subset."""
    s = synthetic_path(src)
    return _abs(src.quantized) if src.quantized else s.with_name(f"{s.stem}_cond-quantization{s.suffix}")


def _codes_per_frame(codes: np.ndarray) -> np.ndarray:
    """Distinct ADC codes per frame, I and Q pooled (one quantizer grid): (n, T) complex ints -> (n,)."""
    c = np.sort(np.concatenate((codes.real, codes.imag), axis=1), axis=1)
    return (np.diff(c, axis=1) != 0).sum(axis=1) + 1


def adc_metrics(x_lsb: np.ndarray, codes: np.ndarray, band: np.ndarray,
                notch: np.ndarray | None, tag: str, x_band: np.ndarray | None = None) -> dict:
    """ADC loading in LSB (FS = 128), median over frames; ENOB = bits a full-scale sine needs for this SQNR.
    In-band SQNR = band signal of `x_band` (default x_lsb; signal at 0 Hz) / (Δ²/6 · B/fs), Δ = 1 LSB."""
    br = np.concatenate((np.abs(x_lsb.real), np.abs(x_lsb.imag)), axis=1)
    rms = np.sqrt(np.mean(np.abs(x_lsb) ** 2, axis=1) / 2)            # per branch
    p999 = np.percentile(br, QUANT_PERCENTILE, axis=1)
    enob = (10 * np.log10(rms ** 2 / (1 / 12)) - 1.76) / 6.02
    # Band signal = in-band PSD minus the out-of-band floor, in LSB^2 (bins sum to mean |x|^2).
    win = np.hanning(FRAME_LEN)
    xb = x_lsb if x_band is None else x_band
    p = np.abs(np.fft.fftshift(np.fft.fft(xb * win, axis=1), axes=1)) ** 2
    p /= FRAME_LEN * np.sum(win ** 2)
    oob = ~_dilate(band, GUARD_KHZ) & (~notch if notch is not None else True)
    sig = p[:, band].sum(axis=1) - np.median(p[:, oob], axis=1) * band.sum()
    q_inband = (1 / 6) * band.sum() / FRAME_LEN
    sqnr = 10 * np.log10(np.where(sig > 0, sig, np.nan) / q_inband)
    return {f"rms_lsb_{tag}": float(np.median(rms)),
            f"peak_lsb_{tag}": float(np.median(br.max(axis=1))),
            f"p999_lsb_{tag}": float(np.median(p999)),
            f"headroom_{tag}_db": float(20 * np.log10(ADC_FS_LSB / np.median(p999))),
            f"codes_{tag}": float(np.median(_codes_per_frame(codes))),
            f"enob_{tag}_bits": float(np.median(enob)),
            f"sqnr_inband_{tag}_db": float(np.nanmedian(sqnr)) if np.isfinite(sqnr).any() else float("nan")}


def synthetic_adc_frames(cls: str, snr_bin: int, n_frames: int = N_FRAMES, src: Sources = SOURCES
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """`quantization`-condition frames in LSB, their codes, and measured/modelled error power.
    Full scale is recomputed per frame exactly as the condition did and checked against the file."""
    path = synthetic_path(src)
    lab, snr = _synth_labels(str(path))
    rows = _spread(np.nonzero((lab == MODULATION_CLASSES.index(cls)) & (snr == snr_bin))[0],
                   n_frames).tolist()
    with h5py.File(path, "r") as f:
        clean = f["X"][rows]
    with h5py.File(quantized_path(src), "r") as f:
        stored = f["X"][rows]
    clean = np.stack([to_complex(x) for x in clean])      # complex64, as make_condition feeds the chain
    stored = np.stack([to_complex(x) for x in stored])
    ref, quant = PercentileReference(QUANT_PERCENTILE), Quantize(ADC_BITS)
    fs = np.array([ref(x) for x in clean])
    again = np.stack([quant(x, None, fs=v) for x, v in zip(clean, fs)])
    # Alignment check: ~1e-5 of samples sit on a bin edge and float rounding moves them one code.
    off = np.abs(again - stored) / (fs / ADC_FS_LSB)[:, None]
    if (off > 1.001).any() or np.mean(off > 1e-3) > 1e-4:
        raise ValueError(f"{quantized_path(src).name} rows do not re-quantize from the subset; "
                         "row alignment or the condition's reference differs")
    clean, stored = clean.astype(np.complex128), stored.astype(np.complex128)
    delta = (fs / ADC_FS_LSB)[:, None]                          # one LSB, per frame
    k = lambda v: np.round((v + fs[:, None]) / delta - 0.5)     # stored values sit on bin centres
    err = np.mean(np.abs(stored - clean) ** 2 / delta ** 2) / (1 / 6)
    return stored / delta, k(stored.real) + 1j * k(stored.imag), float(err)


def adc_loading_row(cls: str, snr_bin: int, n_frames: int = N_FRAMES, src: Sources = SOURCES) -> dict:
    """Real (RTL-SDR codes) vs the synthetic `quantization` condition for one cell."""
    band, notch = occupied_band(cls, src), notch_mask(src)
    fe, raw, clipped = _real_adc_cell(cls, snr_bin, n_frames, src)
    # Loading from the raw ADC values; SQNR from the front-end frames, which carry the band at 0 Hz.
    real = adc_metrics(raw, raw + (127.5 + 127.5j), band, notch, "real", x_band=fe)
    xq, codes, err = synthetic_adc_frames(cls, snr_bin, n_frames, src)
    return ({"class": cls, "snr_bin": snr_bin} | real | adc_metrics(xq, codes, band, None, "synth")
            | {"clipped_frac_real": float(np.mean(clipped)), "qerr_model_ratio_synth": err})


def adc_loading_table(snr_bins=None, classes=MODULATION_CLASSES, n_frames: int = N_FRAMES,
                      src: Sources = SOURCES) -> list[dict]:
    """Rows for every class x bin (default: every usable real bin)."""
    bins = real_bins(src) if snr_bins is None else snr_bins
    return [adc_loading_row(c, b, n_frames, src) for c in classes for b in bins]


def format_adc_table(rows: list[dict]) -> str:
    """Fixed-width text of the ADC-loading rows, real | synthetic `quantization` side by side."""
    fmt = "{:<10}{:>5} |{:>7}{:>7}{:>7}{:>6}{:>6}{:>8} |{:>7}{:>7}{:>6}{:>6}{:>8}"
    lines: list[str] = [
        fmt.format("", "", "real", "", "", "", "", "", "quant", "", "", "", ""),
        fmt.format("class", "bin", "rms", "peak", "hdrm", "codes", "ENOB", "SQNRib",
                   "rms", "peak", "codes", "ENOB", "SQNRib"),
        fmt.format("", "dB", "LSB", "LSB", "dB", "", "bits", "dB", "LSB", "LSB", "", "bits", "dB")]
    for r in rows:
        lines.append(fmt.format(
            r["class"], f"{r['snr_bin']:+d}", f"{r['rms_lsb_real']:.2f}", f"{r['peak_lsb_real']:.1f}",
            f"{r['headroom_real_db']:.1f}", f"{r['codes_real']:.0f}", f"{r['enob_real_bits']:.2f}",
            f"{r['sqnr_inband_real_db']:.1f}", f"{r['rms_lsb_synth']:.1f}",
            f"{r['peak_lsb_synth']:.1f}", f"{r['codes_synth']:.0f}", f"{r['enob_synth_bits']:.2f}",
            f"{r['sqnr_inband_synth_db']:.1f}"))
    return "\n".join(lines)


# Aggregate table

AGGREGATE_COLUMNS = ("class", "snr_bin", "psd_rms_db", "psd_max_db", "psd_tilt_db",
                     "papr_median_real_db", "papr_median_synth_db", "papr_diff_db",
                     "cfo_method", "cfo_median_real_hz", "cfo_median_synth_hz",
                     "cfo_iqr_real_hz", "cfo_iqr_synth_hz", "cfo_reliable_real",
                     "cfo_reliable_synth", "noise_tilt_real_db", "noise_tilt_synth_db",
                     "floor_real_dbc", "floor_synth_dbc")


def aggregate_row(cls: str, snr_bin: int, n_frames: int = N_FRAMES,
                  src: Sources = SOURCES) -> dict:
    """One table row from a single load of the cell; no figures are built."""
    real, synth = real_frames(cls, snr_bin, n_frames, src), synthetic_frames(cls, snr_bin, n_frames, src)
    band, order = occupied_band(cls, src), CFO_ORDER[cls]
    s = (psd_stats(real, synth, band, notch_mask(src)) | papr_stats(real, synth)
         | cfo_stats(cfo_per_frame(real, order, band), cfo_per_frame(synth, order, band), order)
         | noise_stats(real, synth, band, notch_mask(src))[0])
    return {k: s.get(k) for k in AGGREGATE_COLUMNS} | {"class": cls, "snr_bin": snr_bin}


def aggregate_table(snr_bins=(16,), classes=MODULATION_CLASSES, n_frames: int = N_FRAMES,
                    src: Sources = SOURCES) -> list[dict]:
    """Rows for every class at each bin: PSD mismatch, PAPR difference, median CFO, noise-floor tilt."""
    return [aggregate_row(c, b, n_frames, src) for b in snr_bins for c in classes]


def format_table(rows: list[dict]) -> str:
    """Fixed-width text of the aggregate rows; (bracketed) CFO = no line found, see CFO_IQR_OK_HZ."""
    fmt = "{:<10}{:>5}{:>9}{:>9}{:>10}{:>11}{:>10}{:>10}{:>12}{:>12}{:>12}{:>12}"
    lines: list[str] = [fmt.format("class", "bin", "PSD rms", "PSD max", "PSD tilt", "PAPR diff",
                        "CFO real", "CFO RML", "floor tilt", "floor tilt", "floor", "floor"),
             fmt.format("", "dB", "dB", "dB", "dB", "dB", "Hz", "Hz", "real dB", "RML dB",
                        "real dBc", "RML dBc")]
    for r in rows:
        lines.append(fmt.format(
            r["class"], f"{r['snr_bin']:+d}", f"{r['psd_rms_db']:.2f}", f"{r['psd_max_db']:.2f}",
            f"{r['psd_tilt_db']:+.2f}", f"{r['papr_diff_db']:+.2f}",
            _cfo_txt(r, "real"), _cfo_txt(r, "synth"),
            f"{r['noise_tilt_real_db']:+.2f}", f"{r['noise_tilt_synth_db']:+.2f}",
            f"{r['floor_real_dbc']:+.1f}", f"{r['floor_synth_dbc']:+.1f}"))
    return "\n".join(lines)


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = _abs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)
    return path
