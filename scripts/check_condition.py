"""Verify a generated condition: theta is re-estimated FROM THE DATA and compared to the
theta the file declares, so the check is independent of the code that generated it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_condition import default_output_name
from src.config import resolve_data_path
from src.data import KEY_X, KEY_Y, KEY_Z, MODULATION_CLASSES, read_labels_and_snr
from src.distortions import to_complex

_DEFAULT_OUT_DIR = "notebooks/outputs"
# RadioML oversampling factor; used only to downsample for the constellation view.
_DEFAULT_SPS = 8
# Frames pooled per estimate. Estimators are per-frame, so this is a sample size, not a batch.
_DEFAULT_N_FRAMES = 64
# Bin the probe tone is placed on for the IRR spectrum panel (bin-exact -> no leakage).
_PROBE_BIN = 128
# Largest lag in the sigma_w variance-slope fit; well below the 1024-sample frame length.
_MAX_LAG = 64


# Loading a clean / impaired pair

def load_pair(condition, config=None, path=None, impaired=None, snr=30, n_frames=_DEFAULT_N_FRAMES):
    """Load row-aligned (clean, impaired) frames plus the impaired file's metadata.
    Drawn at one SNR (high by default: estimator variance falls with channel noise)."""
    clean_path = Path(resolve_data_path(config, path)[0])
    impaired_path = (Path(impaired) if impaired is not None
                     else clean_path.parent / default_output_name(clean_path, condition, False))
    if not impaired_path.exists():
        raise FileNotFoundError(f"{impaired_path} not found -- run make_condition.py first.")

    class_idx, snr_all, _, _ = read_labels_and_snr(clean_path)
    rows = np.nonzero(snr_all == snr)[0]
    if rows.size == 0:
        raise ValueError(f"SNR {snr} not in the subset. Available: {np.unique(snr_all).tolist()}")
    # Spread across the SNR slice so every class is represented, not just the first ones.
    rows = np.sort(rows[np.linspace(0, rows.size - 1, min(n_frames, rows.size)).astype(int)])

    with h5py.File(clean_path, "r") as fc, h5py.File(impaired_path, "r") as fi:
        if fi[KEY_X].shape != fc[KEY_X].shape:
            raise ValueError("clean and impaired files differ in shape -- frame order is not 1:1")
        idx = rows.tolist()
        clean, dirty = fc[KEY_X][idx], fi[KEY_X][idx]
        # Labels must be untouched: R_theta changes the observation, never the class or SNR.
        assert np.array_equal(fc[KEY_Y][idx], fi[KEY_Y][idx]), "labels drifted"
        assert np.array_equal(fc[KEY_Z][idx], fi[KEY_Z][idx]), "SNRs drifted"
        meta = {k: v for k, v in fi.attrs.items()}

    return clean, dirty, class_idx[rows], meta


# Parameter recovery -- each estimator inverts one operator

def lag_variance_fit(clean, dirty, max_lag=_MAX_LAG):
    """Median (slope, intercept) of var(phi[n+L]-phi[n]) against lag L, over the frames.
    slope = sigma_w^2; the intercept absorbs additive noise and any deterministic CFO ramp."""
    lags = np.arange(1, max_lag + 1)
    slopes, intercepts = [], []
    for c, d in zip(clean, dirty):
        # conj-product rather than a ratio: same phase difference, no divide-by-zero guard.
        phi = np.unwrap(np.angle(to_complex(d) * np.conj(to_complex(c))))
        variances = np.array([np.var(phi[lag:] - phi[:-lag]) for lag in lags])
        slope, intercept = np.polyfit(lags, variances, 1)
        slopes.append(slope)
        intercepts.append(intercept)
    return float(np.median(slopes)), float(np.median(intercepts))


def estimate_phase_noise_sigma(clean, dirty, max_lag=_MAX_LAG):
    """sigma_w from the SLOPE of the lag-variance fit, which is L*sigma_w^2 + const.
    Differencing at L=1 alone would report 0.097 for a true 0.010 under the full chain."""
    slope, _ = lag_variance_fit(clean, dirty, max_lag)
    return float(np.sqrt(max(slope, 0.0)))


def estimate_iq_imbalance(clean, dirty):
    """(gain_db, phase_deg, irr_db) from the least-squares fit y = alpha*x + beta*conj(x).
    alpha + conj(beta) == 1 and alpha - conj(beta) == g*exp(j*psi) invert the model exactly."""
    gains, phases, irrs = [], [], []
    for c, d in zip(clean, dirty):
        x, y = to_complex(c).astype(np.complex128), to_complex(d).astype(np.complex128)
        alpha, beta = np.linalg.lstsq(np.stack((x, np.conj(x)), axis=1), y, rcond=None)[0]
        g_psi = alpha - np.conj(beta)
        gains.append(20.0 * np.log10(np.abs(g_psi)))
        phases.append(np.rad2deg(np.angle(g_psi)))
        irrs.append(np.inf if beta == 0 else 20.0 * np.log10(np.abs(alpha) / np.abs(beta)))
    return float(np.median(gains)), float(np.median(phases)), float(np.median(irrs))


def estimate_quantization_bits(dirty):
    """(n_levels, bits, step) from the level spacing and the outermost OBSERVED level.
    b is a LOWER BOUND: a frame that never reaches the rails leaves the outer codes unseen."""
    levels, bits, steps = [], [], []
    for d in dirty:
        uniq = np.unique(np.concatenate((d[:, 0], d[:, 1])).astype(np.float64))
        if uniq.size < 3:
            continue
        diffs = np.diff(uniq)
        step = float(np.median(diffs[diffs > 0]))
        fs = float(np.abs(uniq).max()) + step / 2.0
        levels.append(uniq.size)
        steps.append(step)
        bits.append(np.log2(2.0 * fs / step))
    return int(np.median(levels)), float(np.median(bits)), float(np.median(steps))


def estimate_dc_offset(clean, dirty):
    """Per-branch DC shift as a fraction of the frame's peak, matching the operator's units."""
    frac_i, frac_q = [], []
    for c, d in zip(clean, dirty):
        fs = float(np.abs(d).max())
        frac_i.append((d[:, 0].mean() - c[:, 0].mean()) / fs)
        frac_q.append((d[:, 1].mean() - c[:, 1].mean()) / fs)
    return float(np.median(frac_i)), float(np.median(frac_q))


def active_operators(theta):
    """Names of the operators in theta that are not at their degenerate value."""
    live = {
        "PhaseNoise": lambda p: p["sigma_w"] != 0.0,
        "IQImbalance": lambda p: p["gain_db"] != 0.0 or p["phase_deg"] != 0.0,
        "DCOffset": lambda p: p["offset_i"] != 0.0 or p["offset_q"] != 0.0,
        "Quantize": lambda p: p["n_bits"] is not None,
    }
    return [name for name, params in theta.items() if name in live and live[name](params)]


def recover_theta(clean, dirty, meta):
    """Re-estimate every declared theta component and pair it with the declared value.
    Each estimator inverts ONE operator, so composite chains report APPROX, never FAIL."""
    theta = json.loads(meta.get("theta", "{}"))
    composite = len(active_operators(theta)) > 1
    rows = []

    def add(name, declared, recovered, tol):
        if composite:
            status = "APPROX"
        else:
            status = "PASS" if np.isfinite(recovered) and abs(recovered - declared) <= tol else "FAIL"
        rows.append({"parameter": name, "declared": declared, "recovered": recovered,
                     "status": status})

    def info(name, declared, recovered):
        rows.append({"parameter": name, "declared": declared, "recovered": recovered,
                     "status": "INFO"})

    if "PhaseNoise" in theta:
        add("sigma_w (rad/sample)", theta["PhaseNoise"]["sigma_w"],
            estimate_phase_noise_sigma(clean, dirty), 0.002)
    if "IQImbalance" in theta:
        gain, phase, irr = estimate_iq_imbalance(clean, dirty)
        add("gain_db (a)", theta["IQImbalance"]["gain_db"], gain, 0.05)
        add("phase_deg (psi)", theta["IQImbalance"]["phase_deg"], phase, 0.3)
        info("IRR (dB)", float("nan"), irr)
    if "DCOffset" in theta:
        off_i, off_q = estimate_dc_offset(clean, dirty)
        add("offset_i (frac FS)", theta["DCOffset"]["offset_i"], off_i, 0.004)
        add("offset_q (frac FS)", theta["DCOffset"]["offset_q"], off_q, 0.004)
    if "Quantize" in theta:
        n_levels, bits, step = estimate_quantization_bits(dirty)
        add("n_bits (b)", theta["Quantize"]["n_bits"], bits, 0.35)
        # Clipping at a percentile reference leaves the outer codes unused, so <= not ==.
        declared_levels = 2 ** theta["Quantize"]["n_bits"]
        rows.append({"parameter": "distinct levels", "declared": declared_levels,
                     "recovered": n_levels,
                     "status": "PASS" if n_levels <= declared_levels else "FAIL"})
        info("step (LSB)", float("nan"), step)
    return rows


def check_nothing_else_changed(clean, dirty, theta_is_identity):
    """Guards that hold for every condition: shape, dtype, finiteness, identity exactness."""
    checks = [
        ("shape preserved", clean.shape == dirty.shape),
        ("dtype preserved", clean.dtype == dirty.dtype),
        ("all finite", bool(np.all(np.isfinite(dirty)))),
    ]
    if theta_is_identity:
        checks.append(("identity theta -> bit-identical", bool(np.array_equal(clean, dirty))))
    else:
        checks.append(("non-identity theta -> data changed", not np.array_equal(clean, dirty)))
    return checks


# Figures

def plot_overlays(clean, dirty, class_idx, frame=0, sps=_DEFAULT_SPS, n_samples=200):
    """Constellation and time-domain overlays, clean vs impaired, same frame index."""
    c, d = clean[frame], dirty[frame]
    name = MODULATION_CLASSES[int(class_idx[frame])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].scatter(c[::sps, 0], c[::sps, 1], s=10, alpha=0.6, label="clean")
    axes[0].scatter(d[::sps, 0], d[::sps, 1], s=10, alpha=0.6, label="impaired")
    axes[0].set_title(f"constellation @ sps={sps} -- {name}")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
    axes[0].set_aspect("equal", adjustable="datalim"); axes[0].legend(fontsize=8)

    t = np.arange(n_samples)
    for ax, ch, label in ((axes[1], 0, "I"), (axes[2], 1, "Q")):
        ax.plot(t, c[:n_samples, ch], lw=1.0, label="clean")
        ax.plot(t, d[:n_samples, ch], lw=1.0, alpha=0.8, label="impaired")
        ax.set_title(f"{label} vs time"); ax.set_xlabel("sample"); ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_spectra(clean, dirty, frame=0):
    """Clean vs impaired spectrum of one frame, plus the difference (the injected energy)."""
    c, d = to_complex(clean[frame]), to_complex(dirty[frame])
    freqs = np.fft.fftshift(np.fft.fftfreq(c.size))

    def db(x): return 20.0 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(x))), 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(freqs, db(c), lw=0.8, label="clean")
    axes[0].plot(freqs, db(d), lw=0.8, alpha=0.8, label="impaired")
    axes[0].set_title("spectrum"); axes[0].legend(fontsize=8)
    axes[1].plot(freqs, db(d - c), lw=0.8, color="crimson")
    axes[1].set_title("injected component (impaired - clean)")
    for ax in axes:
        ax.set_xlabel("normalized frequency"); ax.set_ylabel("dB")
    fig.tight_layout()
    return fig


def plot_image_rejection(clean, meta, frame=0):
    """Probe making the IQ image a SEPARATE spectral line: baseband frames hide their image
    under the signal, so the same theta is applied to a tone at +f0 and read off at -f0."""
    from src.distortions import build_compose

    theta = json.loads(meta.get("theta", "{}"))
    if "IQImbalance" not in theta:
        return None
    op = build_compose([{"name": "iq_imbalance", "kwargs": theta["IQImbalance"]}])

    x = to_complex(clean[frame])
    n = x.size
    tone = np.exp(2j * np.pi * (_PROBE_BIN / n) * np.arange(n)).astype(np.complex64)
    y = op(tone, np.random.default_rng(0))

    spectrum = np.abs(np.fft.fft(y))
    measured = 20.0 * np.log10(spectrum[_PROBE_BIN] / spectrum[-_PROBE_BIN])
    declared = op.distortions[0].irr_db()

    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    db = 20.0 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(y))), 1e-12))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(freqs, db, lw=0.9)
    ax.axvline(_PROBE_BIN / n, color="tab:green", ls="--", lw=1, label="signal")
    ax.axvline(-_PROBE_BIN / n, color="crimson", ls="--", lw=1, label="image")
    ax.annotate(f"IRR = {measured:.2f} dB\n(analytic {declared:.2f} dB)",
                xy=(-_PROBE_BIN / n, db.max() - 30), xytext=(-0.45, db.max() - 12),
                arrowprops=dict(arrowstyle="->", color="crimson"), fontsize=9, color="crimson")
    ax.set_title(f"image rejection probe -- a={theta['IQImbalance']['gain_db']} dB, "
                 f"psi={theta['IQImbalance']['phase_deg']} deg")
    ax.set_xlabel("normalized frequency"); ax.set_ylabel("dB"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_quantizer(clean, dirty, meta, frame=0, n_steps=8):
    """Quantizer staircase and error sawtooth. At 8 bits the full-range curve looks straight,
    so the left panel zooms to a few LSBs and the right bounds the error by +/- step/2."""
    if "Quantize" not in json.loads(meta.get("theta", "{}")):
        return None
    c, d = clean[frame], dirty[frame]
    ci, di = c[:, 0].astype(np.float64), d[:, 0].astype(np.float64)
    _, _, step = estimate_quantization_bits(dirty[frame:frame + 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    centre = float(np.median(ci))
    window = np.abs(ci - centre) < n_steps * step / 2
    order = np.argsort(ci[window])
    axes[0].step(ci[window][order], di[window][order], where="mid", lw=1.0)
    axes[0].plot(ci[window][order], ci[window][order], "--", lw=0.8, color="grey", label="ideal")
    axes[0].set_xlabel("clean I"); axes[0].set_ylabel("quantized I")
    axes[0].set_title(f"transfer characteristic, {n_steps} LSB window (step={step:.2e})")
    axes[0].legend(fontsize=8)

    error = np.concatenate((di - ci, d[:, 1].astype(np.float64) - c[:, 1].astype(np.float64)))
    axes[1].plot(np.concatenate((ci, c[:, 1].astype(np.float64))), error, ".", ms=2, alpha=0.4)
    for sign in (-1, 1):
        axes[1].axhline(sign * step / 2, color="crimson", ls="--", lw=1)
    axes[1].set_xlabel("clean amplitude"); axes[1].set_ylabel("quantization error")
    axes[1].set_title(f"error vs input (dashed: +/- step/2), {np.unique(d).size} distinct levels")
    fig.tight_layout()
    return fig


# Report

def check_condition(condition, config=None, path=None, impaired=None, snr=30,
                    n_frames=_DEFAULT_N_FRAMES, frame=0, sps=_DEFAULT_SPS,
                    plot=True, out_dir=_DEFAULT_OUT_DIR, verbose=True):
    """Run the full numeric + visual check for one condition; returns a report dict."""
    clean, dirty, class_idx, meta = load_pair(condition, config, path, impaired, snr, n_frames)
    theta_is_identity = bool(meta.get("theta_is_identity", False))
    guards = check_nothing_else_changed(clean, dirty, theta_is_identity)
    recovered = recover_theta(clean, dirty, meta)

    if verbose:
        print(f"=== condition: {condition} ===")
        print(f"declared theta : {meta.get('theta')}")
        print(f"rng scheme     : {meta.get('rng_scheme')}")
        print(f"checksum       : {meta.get('content_checksum')}")
        print(f"frames checked : {len(clean)} at SNR {snr} dB\n")
        print("--- nothing else changed ---")
        for name, ok in guards:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print("\n--- theta recovered from the data ---")
        if any(row["status"] == "APPROX" for row in recovered):
            print("  (composite chain: each estimator inverts one operator, so all are biased)")
        print(f"  {'parameter':<22}{'declared':>12}{'recovered':>12}   status")
        for row in recovered:
            print(f"  {row['parameter']:<22}{row['declared']:>12.4g}{row['recovered']:>12.4g}"
                  f"   {row['status']}")

    figures = {}
    if plot:
        figures["overlays"] = plot_overlays(clean, dirty, class_idx, frame, sps)
        figures["spectra"] = plot_spectra(clean, dirty, frame)
        for name, fig in (("image_rejection", plot_image_rejection(clean, meta, frame)),
                          ("quantizer", plot_quantizer(clean, dirty, meta, frame))):
            if fig is not None:
                figures[name] = fig
        if out_dir is not None:
            directory = Path(out_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(directory / f"condition_{condition}_{name}.png", dpi=120)
            if verbose:
                print(f"\nsaved {len(figures)} figure(s) to {directory.resolve()}")

    return {"condition": condition, "meta": meta, "guards": guards,
            "recovered": recovered, "figures": figures,
            "passed": all(ok for _, ok in guards)
                      and not any(r["status"] == "FAIL" for r in recovered)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a generated condition dataset.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--config", default=None, help="Config name/path; its data.path is the clean file.")
    parser.add_argument("--path", default=None, help="Explicit clean HDF5 path.")
    parser.add_argument("--impaired", default=None, help="Explicit impaired HDF5 path.")
    parser.add_argument("--snr", type=int, default=30)
    parser.add_argument("--n-frames", type=int, default=_DEFAULT_N_FRAMES)
    parser.add_argument("--frame", type=int, default=0, help="Which loaded frame to plot.")
    parser.add_argument("--sps", type=int, default=_DEFAULT_SPS)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")  # console is headless
    report = check_condition(
        condition=args.condition, config=args.config, path=args.path, impaired=args.impaired,
        snr=args.snr, n_frames=args.n_frames, frame=args.frame, sps=args.sps,
        plot=not args.no_plot, out_dir=args.out_dir, verbose=True,
    )
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
