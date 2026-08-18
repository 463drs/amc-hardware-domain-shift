"""Tests for src.distortions: each operator is checked against the physics it claims (phase
variance, IRR from (a,psi), 2^b levels), not a golden array that re-encodes today's code."""

import numpy as np
import pytest

from src.distortions import (
    Compose,
    DCOffset,
    FixedReference,
    IQImbalance,
    PeakReference,
    PercentileReference,
    PhaseNoise,
    Quantize,
    build_compose,
    frame_rng,
    to_complex,
    to_iq,
)

FRAME_LEN = 1024


def _frame(seed=0, n=FRAME_LEN):
    """A random complex64 frame with realistic per-branch amplitude."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) / np.sqrt(2)


# Bin-exact tone frequency: k/N lands on an FFT bin, so there is no leakage into the image.
_TONE_BIN = 100


def _tone(n=FRAME_LEN):
    """A single complex exponential -- its image is unambiguous in the spectrum."""
    return np.exp(2j * np.pi * (_TONE_BIN / n) * np.arange(n)).astype(np.complex64)


def _all_operators():
    ref = PeakReference()
    return [
        PhaseNoise(0.01),
        IQImbalance(0.4, 3.0),
        DCOffset(0.01, -0.005, ref),
        Quantize(8, ref),
        Compose([PhaseNoise(0.01), IQImbalance(0.4, 3.0), Quantize(8, ref)]),
    ]


# Shape / dtype / finiteness

@pytest.mark.parametrize("op", _all_operators(), ids=lambda o: type(o).__name__)
def test_preserves_shape_dtype_and_finiteness(op):
    x = _frame()
    y = op(x, frame_rng("c", 0))
    assert y.shape == x.shape
    assert y.dtype == np.complex64
    assert np.all(np.isfinite(y.real)) and np.all(np.isfinite(y.imag))


# Degenerate parameters are exact identities

@pytest.mark.parametrize(
    "op",
    [PhaseNoise(0.0), IQImbalance(0.0, 0.0), DCOffset(0.0, 0.0), Quantize(None), Compose([])],
    ids=lambda o: type(o).__name__,
)
def test_degenerate_value_is_identity(op):
    x = _frame()
    assert op.is_identity
    # Bit-for-bit: is_identity must short-circuit, not multiply by one.
    assert op(x, frame_rng("c", 0)) is x


def test_compose_is_identity_iff_all_elements_are():
    ref = PeakReference()
    assert Compose([PhaseNoise(0.0), IQImbalance(0.0, 0.0)]).is_identity
    assert not Compose([PhaseNoise(0.0), Quantize(8, ref)]).is_identity
    assert not Compose([PhaseNoise(0.01), IQImbalance(0.0, 0.0)]).is_identity


def test_params_report_theta():
    op = Compose([PhaseNoise(0.02), IQImbalance(0.4, 3.0)])
    assert op.params() == {
        "PhaseNoise": {"sigma_w": 0.02},
        "IQImbalance": {"gain_db": 0.4, "phase_deg": 3.0},
    }


# I/Q <-> complex helpers

def test_iq_complex_round_trip_is_exact():
    frame = np.random.default_rng(3).standard_normal((FRAME_LEN, 2)).astype(np.float32)
    assert np.array_equal(to_iq(to_complex(frame)), frame)


# Phase noise

def test_phase_variance_grows_linearly_in_sample_index():
    sigma, n_trials = 0.02, 400
    op = PhaseNoise(sigma)
    ones = np.ones(FRAME_LEN, dtype=np.complex64)
    phases = np.stack([np.angle(op(ones, frame_rng("c", i))) for i in range(n_trials)])
    phases = np.unwrap(phases, axis=1)

    var = phases.var(axis=0)
    expected = sigma ** 2 * np.arange(FRAME_LEN)
    assert var[0] == 0.0                                    # phi[0] is pinned to zero
    # Slope of var vs n, fitted through the origin; ~5% tolerance at 400 trials.
    n = np.arange(FRAME_LEN)
    slope = float((n * var).sum() / (n * n).sum())
    assert slope == pytest.approx(sigma ** 2, rel=0.05)
    assert np.corrcoef(var, expected)[0, 1] > 0.99


def test_phase_noise_psd_slope_is_minus_20_db_per_decade():
    sigma, n_trials = 0.05, 200
    op = PhaseNoise(sigma)
    ones = np.ones(FRAME_LEN, dtype=np.complex64)
    psd = np.zeros(FRAME_LEN // 2 + 1)
    for i in range(n_trials):
        phi = np.unwrap(np.angle(op(ones, frame_rng("c", i))))
        psd += np.abs(np.fft.rfft(phi)) ** 2
    psd /= n_trials

    # Fit over a mid decade, away from DC and the Nyquist edge.
    freqs = np.fft.rfftfreq(FRAME_LEN)
    band = (freqs > 5 / FRAME_LEN) & (freqs < 0.05)
    slope = np.polyfit(np.log10(freqs[band]), 10 * np.log10(psd[band]), 1)[0]
    assert slope == pytest.approx(-20.0, abs=2.0)


def test_phase_noise_scales_with_sigma():
    ones = np.ones(FRAME_LEN, dtype=np.complex64)
    small = np.unwrap(np.angle(PhaseNoise(0.01)(ones, frame_rng("c", 7))))
    large = np.unwrap(np.angle(PhaseNoise(0.03)(ones, frame_rng("c", 7))))
    # Same key -> same standard normals, so the paths differ by exactly the sigma ratio.
    assert np.allclose(large, 3.0 * small, atol=1e-4)


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        PhaseNoise(-0.1)


# IQ imbalance

@pytest.mark.parametrize("gain_db,phase_deg", [(0.2, 1.0), (0.4, 3.0), (0.6, 5.0), (0.0, 2.0), (0.5, 0.0)])
def test_measured_irr_matches_the_ratio_implied_by_a_and_psi(gain_db, phase_deg):
    op = IQImbalance(gain_db, phase_deg)
    y = op(_tone(), frame_rng("c", 0))

    spectrum = np.abs(np.fft.fft(y))
    measured = 20.0 * np.log10(spectrum[_TONE_BIN] / spectrum[-_TONE_BIN])
    assert measured == pytest.approx(op.irr_db(), abs=0.1)


def test_irr_lands_in_the_literature_range():
    # 0.2-0.6 dB and 1-5 deg should give IRR ~30-40 dB.
    for gain_db, phase_deg in [(0.2, 1.0), (0.4, 3.0), (0.6, 5.0)]:
        assert 25.0 < IQImbalance(gain_db, phase_deg).irr_db() < 45.0


def test_sign_convention_i_untouched_q_carries_the_mismatch():
    op = IQImbalance(0.4, 3.0)
    x = _frame(1)
    y = op(x, frame_rng("c", 0))
    g, psi = 10.0 ** (0.4 / 20.0), np.deg2rad(3.0)
    assert np.allclose(y.real, x.real, atol=1e-6)
    assert np.allclose(y.imag, g * (x.imag * np.cos(psi) + x.real * np.sin(psi)), atol=1e-6)


def test_iq_imbalance_is_scale_invariant():
    op = IQImbalance(0.4, 3.0)
    x = _frame(2)
    a, b = op(x, frame_rng("c", 0)), op((7.5 * x).astype(np.complex64), frame_rng("c", 0))
    assert np.allclose(b, 7.5 * a, rtol=1e-5, atol=1e-6)


# Quantization

@pytest.mark.parametrize("n_bits", [2, 3, 4, 6, 8])
def test_distinct_output_levels_match_two_to_the_b(n_bits):
    # Uniform over the full range, so every bin is populated at 1024 samples.
    rng = np.random.default_rng(11)
    x = (rng.uniform(-1, 1, FRAME_LEN) + 1j * rng.uniform(-1, 1, FRAME_LEN)).astype(np.complex64)
    y = Quantize(n_bits, PeakReference())(x, frame_rng("c", 0))
    levels = np.unique(np.concatenate((y.real, y.imag)))
    assert levels.size == 2 ** n_bits


def test_quantization_error_is_bounded_by_half_a_step():
    # Peak reference, so nothing clips and the bound is purely the rounding error.
    n_bits = 6
    x = _frame(4)
    reference = PeakReference()
    y = Quantize(n_bits, reference)(x, frame_rng("c", 0))
    delta = 2.0 * reference(x) / 2 ** n_bits
    assert np.abs(y.real - x.real).max() <= delta / 2 + 1e-6
    assert np.abs(y.imag - x.imag).max() <= delta / 2 + 1e-6


def test_quantization_clips_beyond_full_scale():
    fs, n_bits = 1.0, 4
    x = np.array([5.0 + 5.0j, -5.0 - 5.0j], dtype=np.complex64)
    y = Quantize(n_bits, FixedReference(fs))(x, frame_rng("c", 0))
    delta = 2.0 * fs / 2 ** n_bits
    assert y.real[0] == pytest.approx(fs - delta / 2, abs=1e-6)
    assert y.real[1] == pytest.approx(-fs + delta / 2, abs=1e-6)


@pytest.mark.parametrize("bad", [0, -1, True, 3.5])
def test_invalid_bit_counts_rejected(bad):
    with pytest.raises(ValueError):
        Quantize(bad, PeakReference())


def test_quantize_requires_a_reference():
    with pytest.raises(ValueError, match="ReferenceLevel"):
        Quantize(8)


# Reference levels

def test_peak_reference_never_clips():
    x = _frame(5)
    fs = PeakReference()(x)
    assert np.abs(np.concatenate((x.real, x.imag))).max() <= fs + 1e-9


def test_percentile_reference_is_robust_to_an_outlier():
    x = _frame(6)
    spiked = x.copy()
    spiked[0] = 50.0 + 50.0j
    ref = PercentileReference(99.0)
    assert ref(spiked) == pytest.approx(ref(x), rel=0.15)
    # The peak reference, by contrast, is dragged all the way to the outlier.
    assert PeakReference()(spiked) > 10 * PeakReference()(x)


def test_fixed_reference_ignores_the_frame():
    assert FixedReference(3.0)(_frame(7)) == 3.0
    assert FixedReference(3.0)(_frame(8) * 100) == 3.0


def test_silent_frame_does_not_divide_by_zero():
    zeros = np.zeros(FRAME_LEN, dtype=np.complex64)
    y = Quantize(8, PeakReference())(zeros, frame_rng("c", 0))
    assert np.all(np.isfinite(y.real)) and np.all(np.isfinite(y.imag))


# DC offset

def test_dc_offset_shifts_each_branch_by_the_fraction_of_full_scale():
    fs = 2.0
    x = _frame(9)
    y = DCOffset(0.01, -0.005, FixedReference(fs))(x, frame_rng("c", 0))
    assert np.allclose(y.real - x.real, 0.01 * fs, atol=1e-6)
    assert np.allclose(y.imag - x.imag, -0.005 * fs, atol=1e-6)


def test_dc_offset_requires_a_reference():
    with pytest.raises(ValueError, match="ReferenceLevel"):
        DCOffset(0.01, 0.0)


# Chain construction from config specs

def test_build_compose_resolves_names_and_nested_references():
    chain = build_compose([
        {"name": "phase_noise", "kwargs": {"sigma_w": 0.01}},
        {"name": "quantize", "kwargs": {"n_bits": 8, "reference": {"name": "percentile",
                                                                   "kwargs": {"percentile": 99.9}}}},
    ])
    assert [type(d).__name__ for d in chain.distortions] == ["PhaseNoise", "Quantize"]
    assert isinstance(chain.distortions[1].reference, PercentileReference)
    assert chain.params()["Quantize"]["reference"]["percentile"] == 99.9


def test_build_compose_of_nothing_is_identity():
    assert build_compose([]).is_identity
    assert build_compose(None).is_identity


@pytest.mark.parametrize("spec", [{"name": "nope"}, {"kwargs": {}}, "phase_noise"])
def test_unknown_or_malformed_specs_fail_loudly(spec):
    with pytest.raises(ValueError):
        build_compose([spec])


def test_compose_order_is_load_bearing():
    """Swapping two elements must change the output, or an ordering bug could hide silently."""
    ref = FixedReference(2.0)
    x = _frame(10)
    forward = Compose([DCOffset(0.05, 0.0, ref), Quantize(4, ref)])(x, frame_rng("c", 0))
    reverse = Compose([Quantize(4, ref), DCOffset(0.05, 0.0, ref)])(x, frame_rng("c", 0))
    assert not np.allclose(forward, reverse)


def test_compose_applies_elements_in_list_order():
    """List order IS application order: Compose([A, B]) == B(A(x)), reading along the chain."""
    ref = FixedReference(2.0)
    a, b = DCOffset(0.05, -0.02, ref), Quantize(5, ref)
    x = _frame(11)
    manual = b(a(x, frame_rng("c", 0)), frame_rng("c", 0))
    assert np.array_equal(Compose([a, b])(x, frame_rng("c", 0)), manual)


def test_declared_chain_order_matches_the_implementation():
    """[P, G, B, Q] must apply as Q(B(G(P(x)))) -- i.e. the thesis' Q_b . B_d . G_{a,psi} . P_sigma.
    One shared rng, consumed in list order, so this is exact rather than approximate."""
    ref = FixedReference(2.0)
    p, g = PhaseNoise(0.01), IQImbalance(0.4, 3.0)
    b_op, q = DCOffset(0.01, -0.005, ref), Quantize(6, ref)
    x = _frame(12)

    rng = frame_rng("c", 0)
    manual = q(b_op(g(p(x, rng), rng), rng), rng)
    composed = Compose([p, g, b_op, q])(x, frame_rng("c", 0))
    assert np.array_equal(composed, manual)
    # And the reverse-notation reading is genuinely a different signal.
    assert not np.allclose(composed, Compose([q, b_op, g, p])(x, frame_rng("c", 0)))
