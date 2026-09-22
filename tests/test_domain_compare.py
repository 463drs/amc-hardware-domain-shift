"""Tests for the figure-free numbers in scripts.analysis.domain_compare, on generated signals only."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from scripts.analysis import domain_compare as dc

FS = dc.FS_RX
N = dc.FRAME_LEN
SPS = 8


def _qpsk(n_frames: int, cfo_hz: float = 0.0, phase: float = 0.0, seed: int = 0) -> np.ndarray:
    """Rectangular-pulse QPSK at 128 kBd with a known CFO and carrier phase."""
    rng = np.random.default_rng(seed)
    sym = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, (n_frames, N // SPS))))
    t = np.arange(N)
    return np.repeat(sym, SPS, axis=1) * np.exp(1j * (2 * np.pi * cfo_hz * t / FS + phase))


def _white(n_frames: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_frames, N)) + 1j * rng.standard_normal((n_frames, N))) / np.sqrt(2)


def _band(lo_khz: float, hi_khz: float) -> np.ndarray:
    return (dc.FREQ_KHZ >= lo_khz) & (dc.FREQ_KHZ <= hi_khz)


@pytest.mark.parametrize("cfo", [-2200.0, 0.0, 750.0, 9000.0])
def test_fourth_power_line_recovers_cfo(cfo):
    est = dc.cfo_per_frame(_qpsk(16, cfo), order=4)
    assert np.abs(est - cfo).max() < 20


def test_carrier_line_order_one():
    t = np.arange(N)
    tone = np.tile(np.exp(2j * np.pi * -3100.0 * t / FS), (4, 1))
    assert np.allclose(dc.cfo_per_frame(tone, order=1), -3100.0, atol=5)


def test_centroid_reads_a_shifted_band():
    t = np.arange(N)
    x = _white(64) * 0.01 + np.exp(2j * np.pi * 32_000 * t / FS)
    est = dc.cfo_per_frame(x, order=0, band=_band(20, 44))
    assert np.median(est) == pytest.approx(32_000, abs=300)


def test_derotate_puts_qpsk_back_on_fixed_points():
    x = _qpsk(8, cfo_hz=1500.0, phase=1.1)
    y = dc.derotate_frames(x, order=4, band=_band(-80, 80))
    # After CFO and phase removal every sample sits on one of four points (mod pi/2).
    ang = np.angle(y ** 4)
    assert np.abs(np.angle(np.exp(1j * (ang - ang[:, :1])))).max() < 0.1


def test_psd_stats_zero_for_identical_domains():
    x = _qpsk(32)
    s = dc.psd_stats(x, x, _band(-70, 70))
    assert s["psd_rms_db"] == pytest.approx(0, abs=1e-9)
    assert s["psd_max_db"] == pytest.approx(0, abs=1e-9)
    assert s["psd_tilt_db"] == pytest.approx(0, abs=1e-9)


def test_tilt_is_change_across_span():
    f = np.linspace(-100, 100, 201)
    tilt, fit = dc._tilt(f, 0.03 * f + 5)
    assert tilt == pytest.approx(6.0)
    assert np.allclose(fit, 0.03 * f + 5)


def test_papr_constant_envelope_is_zero():
    assert np.allclose(dc.frame_papr_db(_qpsk(4)), 0, atol=1e-9)


def test_white_noise_floor_is_flat():
    x = _white(512)
    stats, _ = dc.noise_stats(x, x, _band(-70, 70), np.zeros(N, bool))
    assert abs(stats["noise_tilt_real_db"]) < 0.2
    assert abs(stats["noise_edge_droop_real_db"]) < 0.2


def test_sample_stats_sees_gain_imbalance_and_dc():
    x = _white(64)
    x = 2.0 * x.real + 1j * x.imag + 0.5
    s = dc.sample_stats(x, "t")
    assert s["iq_gain_ratio_t_db"] == pytest.approx(20 * np.log10(2), abs=0.1)
    assert s["dc_i_t"] > 0.2 and abs(s["dc_q_t"]) < 0.02


def test_cfo_stats_flags_spread_estimates():
    rng = np.random.default_rng(0)
    s = dc.cfo_stats(rng.normal(-2200, 50, 400), rng.normal(0, 8000, 400), order=4)
    assert s["cfo_reliable_real"] and not s["cfo_reliable_synth"]
    assert dc._cfo_txt(s, "synth").startswith("(")


def test_capture_line_finds_what_single_frames_miss():
    rng = np.random.default_rng(3)
    x = _qpsk(256, cfo_hz=-2300.0, phase=0.3) + 1.2 * _white(256, seed=4)   # ~-1.6 dB per sample
    per_frame = dc.cfo_per_frame(x, order=4)
    assert np.mean(np.abs(per_frame + 2300) <= 300) < 0.5
    cfo, order, prom = dc.capture_line(x)
    assert order == 4 and prom >= dc.LINE_MIN_DB and abs(cfo + 2300) < 20
    assert rng is not None


def test_capture_line_on_noise_stays_below_threshold():
    assert dc.capture_line(_white(256, seed=5))[2] < dc.LINE_MIN_DB


def test_adc_metrics_full_scale_sine_is_eight_bits():
    t = np.arange(N)
    x = 127.5 * np.exp(2j * np.pi * 1000 * t / FS)        # one slow cycle: <= 0.8 LSB/sample, no code skipped
    xq = (np.floor(x.real + 128) - 127.5) + 1j * (np.floor(x.imag + 128) - 127.5)
    x = np.tile(xq, (4, 1))
    m = dc.adc_metrics(x, x + (127.5 + 127.5j), _band(0.5, 1.5), None, "t")
    assert m["enob_t_bits"] == pytest.approx(8.0, abs=0.05)
    assert m["codes_t"] == 256 and m["peak_lsb_t"] == 127.5
    assert m["headroom_t_db"] == pytest.approx(0.0, abs=0.05)


def test_adc_metrics_small_signal_uses_few_codes():
    rng = np.random.default_rng(6)
    x = np.round(rng.standard_normal((8, N)) * 0.7 - 0.5) + 0.5
    x = x + 1j * (np.round(rng.standard_normal((8, N)) * 0.7 - 0.5) + 0.5)
    m = dc.adc_metrics(x, x + (127.5 + 127.5j), _band(-50, 50), None, "t")
    assert m["codes_t"] <= 8 and m["enob_t_bits"] < 1.5 and m["headroom_t_db"] > 30


def test_neighbour_fallback_uses_nearest_in_time():
    from scripts.analysis.build_real_cfodc import nearest_line_cfo
    pool = [{"file": f"f{i}", "mtime": float(t), "line_cfo_hz": v}
            for i, (t, v) in enumerate([(0, -2300), (10, -2290), (20, -2310), (30, -2295),
                                        (1000, -2480), (1010, -2470)])]
    cfo, files, gap = nearest_line_cfo({"file": "x", "mtime": 12.0}, pool)
    assert cfo == pytest.approx(-2297.5) and set(files) == {"f0", "f1", "f2", "f3"} and gap == 18
    assert nearest_line_cfo({"file": "x", "mtime": 1005.0}, pool)[0] < -2350


def test_derotation_modes():
    from scripts.analysis.build_real_cfodc import derotation
    x = _qpsk(16, cfo_hz=-2300.0)
    rec = {"cfo_hz": -2300.0, "line_cfo_hz": -2300.0, "line_ok": True, "line_order": 4, "lock_frac": 1.0}
    cfo, mode = derotation(rec, x)
    assert mode == "per-frame" and np.abs(cfo + 2300).max() < 20
    cfo, mode = derotation(rec | {"lock_frac": 0.2}, x)
    assert mode == "per-capture" and np.all(cfo == -2300.0)
    cfo, mode = derotation(rec | {"line_ok": False, "cfo_hz": -2450.0}, x)
    assert mode == "per-capture" and np.all(cfo == -2450.0)


def test_spread_covers_the_whole_range():
    rows = np.arange(100, 2100)
    got = dc._spread(rows, 5)
    assert got[0] == 100 and got[-1] == 2099 and len(got) == 5


def test_write_csv_round_trip(tmp_path):
    rows = [{"class": "QPSK", "snr_bin": 16, "psd_rms_db": 1.23456789, "cfo_reliable_real": True}]
    path = dc.write_csv(rows, tmp_path / "t.csv")
    with path.open(encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert back == [{"class": "QPSK", "snr_bin": "16", "psd_rms_db": "1.23457",
                     "cfo_reliable_real": "True"}]
