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
    ReferenceLevel,
    build_compose,
    frame_rng,
    sigma_w_from_phase_noise,
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


def _grid_step(y):
    """Spacing of the quantizer's reconstruction grid, read back off the output levels."""
    levels = np.unique(np.concatenate((y.real, y.imag)).astype(np.float64))
    diffs = np.diff(levels)
    return float(np.median(diffs[diffs > 0]))


def _all_operators():
    ref = PeakReference()
    return [
        PhaseNoise(0.01),
        IQImbalance(0.4, 3.0),
        DCOffset(0.01, -0.005),
        Quantize(8),
        Compose([PhaseNoise(0.01), IQImbalance(0.4, 3.0), Quantize(8)], reference=ref),
    ]


# Shape / dtype / finiteness

@pytest.mark.parametrize("op", _all_operators(), ids=lambda o: type(o).__name__)
def test_preserves_shape_dtype_and_finiteness(op):
    x = _frame()
    # FS is handed in from outside, as Compose does; scale-invariant operators ignore it.
    y = op(x, frame_rng("c", 0), PeakReference()(x))
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
    assert not Compose([PhaseNoise(0.0), Quantize(8)], reference=ref).is_identity
    assert not Compose([PhaseNoise(0.01), IQImbalance(0.0, 0.0)]).is_identity


def test_params_report_theta():
    op = Compose([PhaseNoise(0.02), IQImbalance(0.4, 3.0)])
    assert op.params() == {
        "PhaseNoise": {"sigma_w": 0.02},
        "IQImbalance": {"gain_db": 0.4, "phase_deg": 3.0},
    }


def test_params_report_the_reference_and_where_it_is_measured():
    """The AGC model and its measurement point are part of theta, recorded per chain."""
    op = Compose([PhaseNoise(0.02), IQImbalance(0.4, 3.0), DCOffset(0.01, 0.0), Quantize(8)],
                 reference=PeakReference())
    assert op.params()["_reference"] == {
        "kind": "peak", "headroom_db": 0.0, "measured_after": "IQImbalance",
    }
    # No consumer -> nothing to measure, and the metadata says so rather than inventing a point.
    assert Compose([PhaseNoise(0.02)], reference=PeakReference()).measurement_point() is None
    # First operator is the consumer -> measured on the chain's input.
    assert Compose([Quantize(8)], reference=PeakReference()).measurement_point() == "input"


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


# Phase noise from a datasheet figure

# The R820T line in configs/conditions.yaml, and the hand calculation it was checked against.
_DATASHEET = {"l_dbc_hz": -98.0, "offset_hz": 1.0e4, "sample_rate_hz": 1.024e6}
_HAND_SIGMA_W = 7.8e-4


def test_sigma_w_matches_the_hand_calculation():
    """-98 dBc/Hz at 10 kHz, sampled at 1.024 MS/s -> 7.8e-4 rad/sample."""
    assert sigma_w_from_phase_noise(**_DATASHEET) == pytest.approx(_HAND_SIGMA_W, rel=0.01)


def test_sigma_w_scales_as_the_physics_says():
    sigma = sigma_w_from_phase_noise(**_DATASHEET)
    # sigma_w^2 = c / f_s: the SAME oscillator sampled twice as fast walks 1/sqrt(2) as far per
    # sample. This is the whole point of deriving sigma_w instead of hardcoding it.
    faster = sigma_w_from_phase_noise(**{**_DATASHEET, "sample_rate_hz": 2 * 1.024e6})
    assert faster == pytest.approx(sigma / np.sqrt(2.0))
    # c ~ f_offset^2 * 10^(L/10): 10 dB better at the same offset is sqrt(10) less phase noise,
    # and a figure quoted a decade further out describes a 10x worse oscillator.
    assert sigma_w_from_phase_noise(**{**_DATASHEET, "l_dbc_hz": -108.0}) == pytest.approx(
        sigma / np.sqrt(10.0))
    assert sigma_w_from_phase_noise(**{**_DATASHEET, "offset_hz": 1.0e5}) == pytest.approx(
        sigma * 10.0)


@pytest.mark.parametrize("bad", [{"offset_hz": 0.0}, {"offset_hz": -1.0}, {"sample_rate_hz": 0.0}])
def test_sigma_w_rejects_unphysical_rates(bad):
    with pytest.raises(ValueError):
        sigma_w_from_phase_noise(**{**_DATASHEET, **bad})


def test_phase_noise_accepts_the_datasheet_spelling_and_records_it():
    op = PhaseNoise(phase_noise_dbc_hz=-98.0, offset_hz=1.0e4, sample_rate_hz=1.024e6)
    assert op.sigma_w == pytest.approx(_HAND_SIGMA_W, rel=0.01)
    # theta keeps the passport values, not just what they reduce to.
    assert op.params() == {"sigma_w": op.sigma_w, "phase_noise_dbc_hz": -98.0,
                           "offset_hz": 1.0e4, "sample_rate_hz": 1.024e6}
    # ...and the two spellings are the same operator once converted.
    assert np.array_equal(op(_frame(), frame_rng("c", 0)),
                          PhaseNoise(op.sigma_w)(_frame(), frame_rng("c", 0)))


def test_phase_noise_spellings_are_exclusive_and_complete():
    with pytest.raises(ValueError, match="not both"):
        PhaseNoise(sigma_w=0.01, phase_noise_dbc_hz=-98.0, offset_hz=1.0e4, sample_rate_hz=1.024e6)
    with pytest.raises(ValueError, match="BOTH"):
        PhaseNoise(phase_noise_dbc_hz=-98.0, sample_rate_hz=1.024e6)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        PhaseNoise(phase_noise_dbc_hz=-98.0, offset_hz=1.0e4)     # no f_s -> refuse to guess
    with pytest.raises(ValueError):
        PhaseNoise()
    # A directly stated sigma_w is already per-sample, so a stray f_s changes nothing.
    assert PhaseNoise(0.01, sample_rate_hz=1.024e6).params() == {"sigma_w": 0.01}


def test_phase_noise_reads_a_yaml_string_number():
    """YAML 1.1 hands back '1.0e4' as a STRING; the operator must not silently mis-scale."""
    op = PhaseNoise(phase_noise_dbc_hz="-98.0", offset_hz="1.0e4", sample_rate_hz="1.024e6")
    assert op.sigma_w == pytest.approx(_HAND_SIGMA_W, rel=0.01)
    with pytest.raises(ValueError, match="offset_hz"):
        PhaseNoise(phase_noise_dbc_hz=-98.0, offset_hz="ten kHz", sample_rate_hz=1.024e6)


def test_build_compose_injects_the_top_level_sample_rate():
    spec = [{"name": "phase_noise", "kwargs": {"phase_noise_dbc_hz": -98.0, "offset_hz": 1.0e4}}]
    chain = build_compose(spec, sample_rate_hz=1.024e6)
    assert chain.params()["PhaseNoise"]["sigma_w"] == pytest.approx(_HAND_SIGMA_W, rel=0.01)
    # Without a declared f_s the datasheet figure cannot be converted -- and is not guessed at.
    with pytest.raises(ValueError, match="sample_rate_hz"):
        build_compose(spec)


def test_build_compose_rejects_a_per_operator_sample_rate():
    """f_s belongs to the dataset: one rate for every condition in the file, or none."""
    with pytest.raises(ValueError, match="TOP level"):
        build_compose([{"name": "phase_noise",
                        "kwargs": {"phase_noise_dbc_hz": -98.0, "offset_hz": 1.0e4,
                                   "sample_rate_hz": 1.024e6}}])


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
    y = Quantize(n_bits)(x, frame_rng("c", 0), PeakReference()(x))
    levels = np.unique(np.concatenate((y.real, y.imag)))
    assert levels.size == 2 ** n_bits


def test_quantization_error_is_bounded_by_half_a_step():
    # Peak reference, so nothing clips and the bound is purely the rounding error.
    n_bits = 6
    x = _frame(4)
    fs = PeakReference()(x)
    y = Quantize(n_bits)(x, frame_rng("c", 0), fs)
    delta = 2.0 * fs / 2 ** n_bits
    assert np.abs(y.real - x.real).max() <= delta / 2 + 1e-6
    assert np.abs(y.imag - x.imag).max() <= delta / 2 + 1e-6


def test_quantization_clips_beyond_full_scale():
    fs, n_bits = 1.0, 4
    x = np.array([5.0 + 5.0j, -5.0 - 5.0j], dtype=np.complex64)
    y = Quantize(n_bits)(x, frame_rng("c", 0), fs)
    delta = 2.0 * fs / 2 ** n_bits
    assert y.real[0] == pytest.approx(fs - delta / 2, abs=1e-6)
    assert y.real[1] == pytest.approx(-fs + delta / 2, abs=1e-6)


@pytest.mark.parametrize("bad", [0, -1, True, 3.5])
def test_invalid_bit_counts_rejected(bad):
    with pytest.raises(ValueError):
        Quantize(bad)


def test_quantize_without_a_full_scale_level_fails_loudly():
    """No silent self-measurement: an unsupplied FS is an error, not a fallback."""
    with pytest.raises(ValueError, match="full-scale"):
        Quantize(8)(_frame(), frame_rng("c", 0))


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
    y = Compose([Quantize(8)], reference=PeakReference())(zeros, frame_rng("c", 0))
    assert np.all(np.isfinite(y.real)) and np.all(np.isfinite(y.imag))


# DC offset

def test_dc_offset_shifts_each_branch_by_the_fraction_of_full_scale():
    fs = 2.0
    x = _frame(9)
    y = DCOffset(0.01, -0.005)(x, frame_rng("c", 0), fs)
    assert np.allclose(y.real - x.real, 0.01 * fs, atol=1e-6)
    assert np.allclose(y.imag - x.imag, -0.005 * fs, atol=1e-6)


def test_dc_offset_without_a_full_scale_level_fails_loudly():
    with pytest.raises(ValueError, match="full-scale"):
        DCOffset(0.01, 0.0)(_frame(), frame_rng("c", 0))


# One full-scale level per frame, measured at the declared point

class _CountingReference(ReferenceLevel):
    """Wraps a reference and counts how often the chain asks it for a level."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def __call__(self, x):
        self.calls += 1
        return self.inner(x)

    def params(self): return self.inner.params()


def test_full_scale_is_measured_once_per_frame():
    ref = _CountingReference(PeakReference())
    chain = Compose([PhaseNoise(0.01), IQImbalance(0.4, 3.0), DCOffset(0.01, 0.0), Quantize(8)],
                    reference=ref)
    chain(_frame(), frame_rng("c", 0))
    assert ref.calls == 1


def test_full_scale_is_measured_after_the_rf_chain_and_before_the_dc_offset():
    """The declared point of 1.2.1: the AGC sees G_{a,psi}'s output. It does not see the DC
    induced after it, and it cannot depend on its own quantizer."""
    ref = PeakReference()
    x = _frame(13)
    p, g = PhaseNoise(0.01), IQImbalance(0.4, 3.0)
    b, q = DCOffset(0.05, 0.0), Quantize(6)

    # Only PhaseNoise draws, and it draws first, so one shared rng reproduces the chain exactly.
    rng = frame_rng("c", 0)
    after_g = g(p(x, rng), rng)
    fs = ref(after_g)
    manual = q(b(after_g, rng, fs), rng, fs)
    assert np.array_equal(Compose([p, g, b, q], reference=ref)(x, frame_rng("c", 0)), manual)
    # And the point is load-bearing: B_d genuinely moves the level a later measurement would see.
    assert ref(b(after_g, rng, fs)) != pytest.approx(fs)


def test_dc_offset_does_not_move_the_quantizer_full_scale():
    """The defect this design removes: adding B_d must not change Q_b's step size."""
    x = _frame(0)
    ref = PeakReference()
    with_dc = Compose([DCOffset(0.05, 0.0), Quantize(6)], reference=ref)(x, frame_rng("c", 0))
    without = Compose([Quantize(6)], reference=ref)(x, frame_rng("c", 0))
    # Same step size: the quantization grid is identical, only the shifted content differs.
    assert _grid_step(with_dc) == pytest.approx(_grid_step(without), rel=1e-6)


def test_fs_after_overrides_the_declared_measurement_point():
    """The point is a chain parameter, not a hard-coded rule -- and moving it is visible.
    A 6 dB gain error doubles the Q branch, so measuring before or after G_{a,psi} differs."""
    x = _frame(0)
    ref = PeakReference()
    ops = [IQImbalance(6.0, 0.0), Quantize(6)]
    default = Compose(ops, reference=ref)(x, frame_rng("c", 0))              # after G_{a,psi}
    early = Compose(ops, reference=ref, fs_after=-1)(x, frame_rng("c", 0))   # at the input
    assert Compose(ops, reference=ref, fs_after=-1).measurement_point() == "input"
    assert _grid_step(early) != pytest.approx(_grid_step(default), rel=1e-6)


@pytest.mark.parametrize("bad", [-2, 2, 5])
def test_fs_after_outside_the_chain_is_rejected(bad):
    with pytest.raises(ValueError, match="fs_after"):
        Compose([DCOffset(0.05, 0.0), Quantize(6)], reference=PeakReference(), fs_after=bad)


def test_fs_after_may_not_strand_a_consumer_before_the_measurement_point():
    """Measuring after B_d would leave B_d itself with no level -- a declaration error,
    caught when the chain is built rather than once per frame."""
    with pytest.raises(ValueError, match="before the measurement point"):
        Compose([DCOffset(0.05, 0.0), Quantize(6)], reference=PeakReference(), fs_after=0)


def test_fs_after_without_a_reference_is_rejected():
    with pytest.raises(ValueError, match="needs a reference"):
        Compose([PhaseNoise(0.01)], fs_after=-1)


def test_an_enclosing_chain_wins_over_a_nested_reference():
    """A nested chain never re-measures: the level travels down, it is not recomputed."""
    x = _frame(14)
    inner_ref = _CountingReference(PeakReference())
    inner = Compose([Quantize(6)], reference=inner_ref)
    outer = Compose([DCOffset(0.05, 0.0), inner], reference=FixedReference(2.0))

    y = outer(x, frame_rng("c", 0))
    assert inner_ref.calls == 0
    expected = Compose([DCOffset(0.05, 0.0), Quantize(6)], reference=FixedReference(2.0))
    assert np.array_equal(y, expected(x, frame_rng("c", 0)))


# Chain construction from config specs

def test_build_compose_resolves_names_and_the_chain_level_reference():
    chain = build_compose({
        "reference": {"name": "percentile", "kwargs": {"percentile": 99.9}},
        "operators": [
            {"name": "phase_noise", "kwargs": {"sigma_w": 0.01}},
            {"name": "quantize", "kwargs": {"n_bits": 8}},
        ],
    })
    assert [type(d).__name__ for d in chain.distortions] == ["PhaseNoise", "Quantize"]
    assert isinstance(chain.reference, PercentileReference)
    assert chain.params()["_reference"]["percentile"] == 99.9
    assert chain.params()["_reference"]["measured_after"] == "PhaseNoise"


def test_build_compose_rejects_a_per_operator_reference():
    """The old spelling must fail with a message that says where 'reference' moved to."""
    with pytest.raises(ValueError, match="condition level"):
        build_compose([{"name": "quantize", "kwargs": {"n_bits": 8,
                                                       "reference": {"name": "peak"}}}])


def test_build_compose_requires_a_reference_when_the_chain_has_a_consumer():
    """Fail at build time, not hours into a generation run."""
    with pytest.raises(ValueError, match="full-scale level"):
        build_compose([{"name": "quantize", "kwargs": {"n_bits": 8}}])
    with pytest.raises(ValueError, match="full-scale level"):
        build_compose([{"name": "dc_offset", "kwargs": {"offset_i": 0.01, "offset_q": 0.0}}])
    # A degenerate consumer asks for nothing, so a bare list stays legal.
    assert build_compose([{"name": "quantize", "kwargs": {"n_bits": None}}]).is_identity


def test_build_compose_reads_fs_after_from_the_condition():
    spec = {
        "reference": {"name": "peak"},
        "operators": [
            {"name": "iq_imbalance", "kwargs": {"gain_db": 6.0, "phase_deg": 0.0}},
            {"name": "quantize", "kwargs": {"n_bits": 6}},
        ],
    }
    assert build_compose(spec).measurement_point() == "IQImbalance"    # the declared default
    assert build_compose({**spec, "fs_after": -1}).measurement_point() == "input"


def test_build_compose_of_nothing_is_identity():
    assert build_compose([]).is_identity
    assert build_compose(None).is_identity
    assert build_compose({"operators": []}).is_identity


@pytest.mark.parametrize("spec", [{"name": "nope"}, {"kwargs": {}}, "phase_noise"])
def test_unknown_or_malformed_specs_fail_loudly(spec):
    with pytest.raises(ValueError):
        build_compose([spec])


@pytest.mark.parametrize("spec", [{"name": "phase_noise"}, {"operators": [], "refrence": {}}])
def test_unknown_condition_level_keys_fail_loudly(spec):
    with pytest.raises(ValueError, match="unknown condition keys"):
        build_compose(spec)


def test_compose_order_is_load_bearing():
    """Swapping two elements must change the output, or an ordering bug could hide silently."""
    ref = FixedReference(2.0)
    x = _frame(10)
    forward = Compose([DCOffset(0.05, 0.0), Quantize(4)], reference=ref)(x, frame_rng("c", 0))
    reverse = Compose([Quantize(4), DCOffset(0.05, 0.0)], reference=ref)(x, frame_rng("c", 0))
    assert not np.allclose(forward, reverse)


def test_compose_applies_elements_in_list_order():
    """List order IS application order: Compose([A, B]) == B(A(x)), reading along the chain."""
    fs = 2.0
    a, b = DCOffset(0.05, -0.02), Quantize(5)
    x = _frame(11)
    manual = b(a(x, frame_rng("c", 0), fs), frame_rng("c", 0), fs)
    composed = Compose([a, b], reference=FixedReference(fs))(x, frame_rng("c", 0))
    assert np.array_equal(composed, manual)


def test_declared_chain_order_matches_the_implementation():
    """[P, G, B, Q] must apply as Q(B(G(P(x)))) -- i.e. the thesis' Q_b . B_d . G_{a,psi} . P_sigma.
    One shared rng, consumed in list order, so this is exact rather than approximate.
    A fixed reference, so the ordering is tested on its own, not through the FS measurement."""
    ref, fs = FixedReference(2.0), 2.0
    p, g = PhaseNoise(0.01), IQImbalance(0.4, 3.0)
    b_op, q = DCOffset(0.01, -0.005), Quantize(6)
    x = _frame(12)

    rng = frame_rng("c", 0)
    manual = q(b_op(g(p(x, rng), rng), rng, fs), rng, fs)
    composed = Compose([p, g, b_op, q], reference=ref)(x, frame_rng("c", 0))
    assert np.array_equal(composed, manual)
    # And the reverse-notation reading is genuinely a different signal.
    assert not np.allclose(composed, Compose([q, b_op, g, p], reference=ref)(x, frame_rng("c", 0)))
