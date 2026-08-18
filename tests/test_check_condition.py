"""Tests for the parameter-recovery estimators: inject a known theta, check it comes back.
This is what makes the notebook's verdict independent of the code that applied theta."""

import numpy as np
import pytest

from scripts.check_condition import (
    active_operators,
    estimate_dc_offset,
    estimate_iq_imbalance,
    estimate_phase_noise_sigma,
    estimate_quantization_bits,
    lag_variance_fit,
)
from src.distortions import (
    DCOffset,
    FixedReference,
    IQImbalance,
    PeakReference,
    PhaseNoise,
    Quantize,
    frame_rng,
    to_iq,
)

FRAME_LEN = 1024
N_FRAMES = 24


def _clean(n=N_FRAMES, t=FRAME_LEN):
    """Stored-layout (N, T, 2) float32 frames at roughly unit power, like RadioML."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((n, t, 2)) / np.sqrt(2)).astype(np.float32)


def _apply(op, clean, condition="c"):
    from src.distortions import to_complex
    return np.stack([to_iq(op(to_complex(f), frame_rng(condition, i)))
                     for i, f in enumerate(clean)])


@pytest.mark.parametrize("sigma_w", [0.005, 0.01, 0.03])
def test_recovers_sigma_w(sigma_w):
    clean = _clean()
    dirty = _apply(PhaseNoise(sigma_w), clean)
    assert estimate_phase_noise_sigma(clean, dirty) == pytest.approx(sigma_w, rel=0.15)


@pytest.mark.parametrize("gain_db,phase_deg", [(0.2, 1.0), (0.4, 3.0), (0.6, 5.0)])
def test_recovers_gain_phase_and_irr(gain_db, phase_deg):
    clean = _clean()
    op = IQImbalance(gain_db, phase_deg)
    gain, phase, irr = estimate_iq_imbalance(clean, _apply(op, clean))
    assert gain == pytest.approx(gain_db, abs=0.01)
    assert phase == pytest.approx(phase_deg, abs=0.05)
    assert irr == pytest.approx(op.irr_db(), abs=0.1)


@pytest.mark.parametrize("n_bits", [4, 6, 8])
def test_recovers_bit_depth(n_bits):
    """Exact when the signal exercises the full range, which a per-frame reference ensures."""
    clean = _clean()
    dirty = _apply(Quantize(n_bits, PeakReference()), clean)
    n_levels, bits, _ = estimate_quantization_bits(dirty)
    assert bits == pytest.approx(n_bits, abs=0.2)
    assert n_levels <= 2 ** n_bits


def test_bit_depth_is_a_lower_bound_when_the_signal_never_clips():
    """With headroom the outer codes are unused, so full scale is not identifiable."""
    clean = _clean()
    # FS=3.0 against a ~0.707-sigma signal: the rails are ~4 sigma away and never reached.
    dirty = _apply(Quantize(8, FixedReference(3.0)), clean)
    _, bits, _ = estimate_quantization_bits(dirty)
    assert bits < 8
    assert bits > 7


def test_recovers_dc_offset():
    clean = _clean()
    dirty = _apply(DCOffset(0.02, -0.01, FixedReference(3.0)), clean)
    off_i, off_q = estimate_dc_offset(clean, dirty)
    # Recovered against the frame's own peak, so exact only in sign and rough magnitude.
    assert off_i > 0 and off_q < 0
    assert abs(off_i / off_q) == pytest.approx(2.0, rel=0.2)


def test_sigma_estimator_survives_added_quantization():
    """The lag-variance fit must not absorb white ADC noise into sigma_w."""
    clean = _clean()
    noisy = _apply(Quantize(8, FixedReference(3.0)), _apply(PhaseNoise(0.01), clean))
    assert estimate_phase_noise_sigma(clean, noisy) == pytest.approx(0.01, rel=0.25)


def _apply_cfo(frames, freq):
    """Deterministic carrier frequency offset: a per-sample phase ramp, as in R_0."""
    from src.distortions import to_complex
    ramp = np.exp(2j * np.pi * freq * np.arange(frames.shape[1]))
    return np.stack([to_iq((to_complex(f) * ramp).astype(np.complex64)) for f in frames])


@pytest.mark.parametrize("freq", [0.0, 1e-4, 1e-3, 5e-3])
def test_sigma_estimate_is_invariant_to_cfo(freq):
    """A CFO ramp is a CONSTANT in the phase increment, so it drops out of every lag variance.
    This is why the estimator survives R_0, which carries CFO -- locked down, not incidental."""
    clean = _clean()
    dirty = _apply_cfo(_apply(PhaseNoise(0.01), clean), freq)
    assert estimate_phase_noise_sigma(clean, dirty) == pytest.approx(0.01, rel=0.1)


def test_cfo_moves_neither_slope_nor_intercept():
    """Stronger than the sigma check: the whole fit is unmoved, not just its square root."""
    clean = _clean()
    dirty = _apply(PhaseNoise(0.01), clean)
    base_slope, base_intercept = lag_variance_fit(clean, dirty)
    slope, intercept = lag_variance_fit(clean, _apply_cfo(dirty, 2e-3))
    assert slope == pytest.approx(base_slope, rel=0.02)
    assert intercept == pytest.approx(base_intercept, abs=1e-4)


def test_quantization_noise_raises_the_intercept_not_the_slope():
    """Additive noise is uncorrelated across lags, so it lands entirely in the intercept."""
    clean = _clean()
    clean_phase = _apply(PhaseNoise(0.01), clean)
    noisy_phase = _apply(Quantize(6, FixedReference(3.0)), clean_phase)

    slope, intercept = lag_variance_fit(clean, clean_phase)
    noisy_slope, noisy_intercept = lag_variance_fit(clean, noisy_phase)
    assert noisy_slope == pytest.approx(slope, rel=0.25)     # sigma_w^2 survives
    assert noisy_intercept > intercept + 1e-4                # the ADC noise goes here


def test_active_operators_ignores_degenerate_entries():
    assert active_operators({"PhaseNoise": {"sigma_w": 0.0}}) == []
    assert active_operators({"IQImbalance": {"gain_db": 0.0, "phase_deg": 0.0}}) == []
    assert active_operators({"Quantize": {"n_bits": None}}) == []
    assert active_operators({"PhaseNoise": {"sigma_w": 0.01},
                             "Quantize": {"n_bits": 8}}) == ["PhaseNoise", "Quantize"]
