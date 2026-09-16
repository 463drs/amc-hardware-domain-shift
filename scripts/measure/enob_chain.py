"""
Effective quantization of the RTL-SDR Blog V4 output stream (u8 interleaved I/Q).

(A) noise_enob   - ENOB from the noise capture, referenced to DIGITAL full scale
                   (complex exponential |x| = 127.5 LSB). No tone needed.
                   This is ENOB_SNR of the whole chain at the given gain, not IEEE 1241
                   SINAD-ENOB, and not quantization alone.
(B) sine_dnl     - code-density (histogram) test on the tone capture -> DNL -> b_eff.
                   Insensitive to additive thermal noise; this is the quantizer itself.
(C) loading      - reference level of real frames in codes -> parameter for the
                   calibrated Quantize (full scale is a property of the chain).

Preconditions: identical fixed tuner gain in all captures, RTL AGC and tuner AGC off,
tone offset from DC, no clipping (clip == 0).
NOT EXECUTED. Run validate_synthetic() first.
"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FS = 1_024_000
N = 1 << 18
SKIP_S = 1.0          # discard tuner / USB start-up transient
BAND = 0.40           # integrate |f| < BAND*FS; trust only if droop_db is ~0
DC_GUARD = 8          # bins around DC always excluded (DC is reported separately)
SPUR_DB = 20.0        # excision threshold above the local median baseline
MED_WIN = 257         # sliding-median window, bins
MAX_BLOCKS = 30
B_NOM = 8
EDGE = 0.05           # histogram: drop codes within 5 % of the sine extremes
DNL_DETREND = 15      # codes; smooth DNL component = TX distortion / PDF model error
LN2 = np.log(2.0)


def load_codes(fname, skip_s=SKIP_S):
    raw = np.fromfile(fname, dtype=np.uint8)
    raw = raw[: raw.size - raw.size % 2]
    raw = raw[2 * int(skip_s * FS):]
    if raw.size < 2 * N:
        raise ValueError(f"{fname}: fewer than N samples after skipping {skip_s} s")
    return raw.reshape(-1, 2)                     # [:, 0] = I, [:, 1] = Q


def to_lsb(codes):
    c = codes.astype(np.float64) - 127.5
    return c[:, 0] + 1j * c[:, 1]


# ---------------------------------------------------------------- (A)
def noise_enob(codes):
    x = to_lsb(codes)
    w = np.hanning(N)
    f = np.fft.fftfreq(N)
    k = np.arange(N)
    near_dc = np.minimum(k, N - k) <= DC_GUARD
    inband = np.abs(f) < BAND
    nblk = min(x.size // N, MAX_BLOCKS)

    pn, excised, avg = [], [], np.zeros(N)
    for b in range(nblk):
        xb = x[b * N:(b + 1) * N]
        xb = xb - xb.mean()
        P = np.abs(np.fft.fft(xb * w)) ** 2 / (N * np.sum(w ** 2))   # sum(P) = mean|x|^2
        base = median_filter(P, size=MED_WIN, mode="wrap") / LN2       # chi2_2: mean = median/ln2
        bad = (P > base * LN2 * 10 ** (SPUR_DB / 10)) | near_dc
        P = np.where(bad, base, P)
        avg += P / nblk
        pn.append(P[inband].sum() / inband.mean())   # white-noise extrapolation to full band
        excised.append(bad[inband].mean())

    pn = np.asarray(pn)
    enob = (10 * np.log10(127.5 ** 2 / pn) - 1.76) / 6.02
    centre = avg[np.abs(f) < 0.1].mean()
    edge = avg[(np.abs(f) > BAND - 0.05) & inband].mean()
    I, Q = codes[:, 0], codes[:, 1]
    return dict(
        enob_med=np.median(enob), enob_min=enob.min(), enob_max=enob.max(),
        sigma_lsb=np.sqrt(np.median(pn) / 2),          # per channel, incl. quantization
        dc_lsb=(I.mean() - 127.5, Q.mean() - 127.5),
        n_codes=(np.unique(I).size, np.unique(Q).size),
        droop_db=10 * np.log10(edge / centre),
        excised=float(np.median(excised)),
        n_blocks=nblk,
    )


# ---------------------------------------------------------------- (B)
def sine_dnl(ch, sigma_n_lsb):
    """ch: one channel (uint8) of the tone capture.
    sigma_n_lsb: per-channel noise from noise_enob() at the same gain."""
    M = ch.size
    H = np.bincount(ch, minlength=256).astype(np.float64)
    v = ch.astype(np.float64) - 127.5
    off = v.mean()
    A = np.sqrt(2.0 * max(v.var() - sigma_n_lsb ** 2, 0.0))
    k = np.arange(256)
    lo = (k - 128.0 - off) / A                     # code k spans [k-128, k-127) LSB
    hi = (k - 127.0 - off) / A
    Pk = (np.arcsin(np.clip(hi, -1, 1)) - np.arcsin(np.clip(lo, -1, 1))) / np.pi
    use = (np.maximum(np.abs(lo), np.abs(hi)) < 1 - EDGE) & (k > 0) & (k < 255)

    dnl = H[use] / (M * Pk[use]) - 1.0
    dnl_hf = dnl - uniform_filter1d(dnl, DNL_DETREND, mode="nearest")
    wdt = 1.0 + dnl_hf
    # uniform input: sigma_q^2 = E[w^3] / (12 E[w]); ideal = 1/12 with E[w] = 1
    b_eff = B_NOM - 0.5 * np.log2(np.mean(wdt ** 3) / np.mean(wdt) ** 3)
    return dict(
        A_lsb=A, off_lsb=off,
        clip=float(np.mean((ch == 0) | (ch == 255))),
        n_used=int(use.sum()),
        dnl_hf_max=float(np.abs(dnl_hf).max()),
        dnl_trend_max=float(np.abs(dnl - dnl_hf).max()),
        b_eff=b_eff,
    )


# ---------------------------------------------------------------- (C)
def loading(codes, frame=1024, q=99.9):
    """Per-frame reference level in LSB of real (modulated) captures.
    PLACEHOLDER statistic max(|I|,|Q|): replace with exactly the statistic used by
    ReferenceLevel(percentile) in src/distortions.py, otherwise the numbers are not comparable."""
    m = np.abs(codes.astype(np.float64) - 127.5).max(axis=1)
    nf = m.size // frame
    ref = np.percentile(m[: nf * frame].reshape(nf, frame), q, axis=1)
    return dict(ref_med=np.median(ref), ref_p5=np.percentile(ref, 5),
                ref_p95=np.percentile(ref, 95), fs_lsb=127.5, n_frames=nf)


# ---------------------------------------------------------------- self-check
def _quantize(v_lsb):
    return np.clip(np.floor(v_lsb + 128.0), 0, 255).astype(np.uint8)


def validate_synthetic(seed=0, sigma=1.5, A=110.0):
    rng = np.random.default_rng(seed)
    n = MAX_BLOCKS * N
    noise = np.stack([_quantize(rng.normal(0, sigma, n)),
                      _quantize(rng.normal(0, sigma, n))], axis=1)
    ph = 2 * np.pi * 0.1234567 * np.arange(n)
    tone_i = _quantize(A * np.cos(ph) + rng.normal(0, sigma, n))

    rn = noise_enob(noise)
    expected = (10 * np.log10(127.5 ** 2 / (2 * (sigma ** 2 + 1 / 12))) - 1.76) / 6.02
    rt = sine_dnl(tone_i, rn["sigma_lsb"])
    print(f"[synthetic] ENOB {rn['enob_med']:.3f} (expected {expected:.3f}), "
          f"sigma {rn['sigma_lsb']:.3f} LSB (expected {np.sqrt(sigma**2 + 1/12):.3f})")
    print(f"[synthetic] b_eff {rt['b_eff']:.3f} (expected {B_NOM}), A {rt['A_lsb']:.2f} "
          f"(expected {A}), |DNL_hf|max {rt['dnl_hf_max']:.4f} (statistical only)")


if __name__ == "__main__":
    validate_synthetic()

    rn = noise_enob(load_codes("noise_cap.bin"))
    print(rn)
    if rn["sigma_lsb"] < 0.5:
        print("INVALID: noise below ~0.5 LSB; the uniform-error model does not hold, "
              "ENOB reflects rounding of a quasi-constant input.")
    if abs(rn["droop_db"]) > 0.5:
        print(f"BAND too wide: {rn['droop_db']:.2f} dB droop at the band edge.")

    tone = load_codes("tone_cap.bin")
    for name, ch in (("I", tone[:, 0]), ("Q", tone[:, 1])):
        rt = sine_dnl(ch, rn["sigma_lsb"])
        if rt["clip"] > 0:
            print(f"{name}: clipping {rt['clip']:.2e}; reduce TX level")
        print(name, rt)

    # print(loading(load_codes("real_frames.bin")))
