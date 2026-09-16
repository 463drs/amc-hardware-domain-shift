"""
Fast calibration of theta for the `calibrated` condition from conductive captures
(HackRF CW -> 30 dB attenuator -> RTL-SDR Blog V4, fixed -g, same frequency region
as the future real-domain recordings).

  tone_pos.bin : RX tuned to F - DELTA -> tone at +DELTA
  tone_neg.bin : RX tuned to F + DELTA -> tone at -DELTA
  mod.bin      : optional, modulated signal at the planned real-domain level

IQ-imbalance model used here (asymmetric, I is the reference):
    I' = I,   Q' = a * (Q cos(psi) + I sin(psi))
    IRR = (1 + a^2 - 2 a cos psi) / (1 + a^2 + 2 a cos psi)
Map (a, psi) to the convention of IQImbalance in src/distortions.py; verify the
mapping by checking that IQImbalance applied to a clean tone reproduces IRR.

Phase noise: Wiener model, sigma_w = per-sample increment std (same as the generator).
Result is an UPPER BOUND (TX + RX).

NOT EXECUTED. Run validate_synthetic() first:  python calibrate_theta.py --validate
"""
import sys
import numpy as np
from scipy.signal import welch

FS = 1_024_000
DELTA = 100_000
SKIP_S = 1.0
DC_GUARD = 8
PN_BAND = (1_000.0, 4_000.0)          # fit band, expected 1/f^2 (Wiener) region
FLOOR_BAND = (60_000.0, 90_000.0)     # white phase floor from additive noise
FULL_SCALE = 127.5


def load(fname):
    raw = np.fromfile(fname, dtype=np.uint8)
    raw = raw[: raw.size - raw.size % 2][2 * int(SKIP_S * FS):]
    c = raw.reshape(-1, 2)
    clip = float(np.mean((c == 0) | (c == 255)))
    v = c.astype(np.float64) - 127.5
    return v[:, 0], v[:, 1], clip


# ------------------------------------------------------------ IQ imbalance
def iq_imbalance(I, Q, nblk=1 << 14):
    """y = K1 x + K2 x*.  For a tone, c(+f) * c(-f) / |c(+f)|^2 = K2 / conj(K1) = mu,
    independent of tone phase and window scalloping. Asymmetric model:
    mu = (1 - z) / (1 + z),  z = a * exp(-j psi).
    Only the component at the mirror bin is measured, so any TX image that lands
    there is included (tone_cap.bin); on tone_ret.bin the TX image sits elsewhere."""
    y = (I - I.mean()) + 1j * (Q - Q.mean())
    nb = y.size // nblk
    Y = np.fft.fft(y[: nb * nblk].reshape(nb, nblk) * np.hanning(nblk), axis=1)
    k = np.arange(nblk)
    S = np.abs(Y) ** 2
    S[:, np.minimum(k, nblk - k) <= DC_GUARD] = 0.0
    p = np.argmax(S, axis=1)                       # per block, follows drift
    r = np.arange(nb)
    cp, cm = Y[r, p], Y[r, (-p) % nblk]
    mu = cm * cp / np.abs(cp) ** 2
    mu = np.median(mu.real) + 1j * np.median(mu.imag)
    z = (1 - mu) / (1 + mu)
    return float(np.abs(z)), float(-np.angle(z))


def irr_model(a, psi):
    return (1 + a * a - 2 * a * np.cos(psi)) / (1 + a * a + 2 * a * np.cos(psi))


def irr_spectral(I, Q, n=1 << 18):
    y = (I[:n] - I[:n].mean()) + 1j * (Q[:n] - Q[:n].mean())
    S = np.abs(np.fft.fft(y * np.hanning(n))) ** 2
    k = np.arange(n)
    S[np.minimum(k, n - k) <= DC_GUARD] = 0.0
    p = int(np.argmax(S))
    band = lambda c: S[(c + np.arange(-30, 31)) % n].sum()
    f_carrier = (p if p < n // 2 else p - n) * FS / n
    return band((-p) % n) / band(p), f_carrier


# ------------------------------------------------------------ phase noise
def phase_noise(I, Q, a, psi):
    I = I - I.mean()
    Q = Q - Q.mean()
    Qc = (Q / a - I * np.sin(psi)) / np.cos(psi)          # undo imbalance: no 2*DELTA ripple
    phi = np.unwrap(np.angle(I + 1j * Qc))
    f, S = welch(phi, fs=FS, nperseg=1 << 16, detrend="linear",
                 return_onesided=False, scaling="density")  # two-sided, rad^2/Hz
    af = np.abs(f)
    fit = (af >= PN_BAND[0]) & (af <= PN_BAND[1])
    floor = np.median(S[(af >= FLOOR_BAND[0]) & (af <= FLOOR_BAND[1])])
    # Wiener: S_phi(f) = sigma_w^2 / (FS * 4 sin^2(pi f / FS))
    g = FS * 4 * np.sin(np.pi * af[fit] / FS) ** 2
    raw = S[fit] * g
    sub = (S[fit] - floor) * g
    lo = af[fit] < np.sqrt(PN_BAND[0] * PN_BAND[1])
    return dict(
        sigma_w_raw=float(np.sqrt(np.median(raw))),                 # floor not removed
        sigma_w=float(np.sqrt(max(np.median(sub), 0.0))),           # floor removed
        floor_frac=float(np.median(floor * g) / np.median(raw)),    # > ~0.3: fit band too high
        wiener_dev_db=float(10 * np.log10(np.median(sub[lo]) / np.median(sub[~lo]))),  # ~0 expected
    )


# ------------------------------------------------------------ quantization loading
def loading(I, Q, frame=1024, q=99.9):
    """PLACEHOLDER statistic max(|I|,|Q|): replace with the exact statistic of
    ReferenceLevel(percentile) in src/distortions.py."""
    m = np.maximum(np.abs(I), np.abs(Q))
    nf = m.size // frame
    ref = np.percentile(m[: nf * frame].reshape(nf, frame), q, axis=1)
    med = float(np.median(ref))
    return dict(ref_med_lsb=med, ref_p5=float(np.percentile(ref, 5)),
                ref_p95=float(np.percentile(ref, 95)), headroom=FULL_SCALE / med)


# ------------------------------------------------------------ driver
def analyse_tone(fname):
    I, Q, clip = load(fname)
    a, psi = iq_imbalance(I, Q)
    irr_s, fc = irr_spectral(I, Q)
    pn = phase_noise(I, Q, a, psi)
    peak = float(np.max(np.hypot(I, Q)))
    return dict(file=fname, clip=clip, peak_lsb=peak, f_carrier=fc,
                a=a, psi_deg=np.degrees(psi),
                irr_model_db=10 * np.log10(irr_model(a, psi)),
                irr_spec_db=10 * np.log10(irr_s), **pn)


def _q(v):
    return np.clip(np.floor(v + 128.0), 0, 255) - 127.5


def validate_synthetic(a=1.05, psi=0.03, sw=7.7e-4, A=110.0, sigma=1.5, seed=0):
    rng = np.random.default_rng(seed)
    n = 1 << 24
    th = 2 * np.pi * DELTA / FS * np.arange(n) + np.cumsum(rng.normal(0, sw, n))
    I = _q(A * np.cos(th) + rng.normal(0, sigma, n))
    Q = _q(a * A * np.sin(th + psi) + rng.normal(0, sigma, n))
    ah, ph = iq_imbalance(I, Q)
    irr_s, fc = irr_spectral(I, Q)
    pn = phase_noise(I, Q, ah, ph)
    print(f"a      {ah:.5f}  (true {a})")
    print(f"psi    {ph:.5f}  (true {psi})")
    print(f"IRR    model {10*np.log10(irr_model(ah, ph)):.2f} dB, "
          f"spectral {10*np.log10(irr_s):.2f} dB, true {10*np.log10(irr_model(a, psi)):.2f} dB")
    print(f"f_c    {fc:.0f} Hz (true {DELTA})")
    print(f"sigma_w {pn['sigma_w']:.3e} (raw {pn['sigma_w_raw']:.3e}, true {sw:.3e}), "
          f"floor_frac {pn['floor_frac']:.2f}, wiener_dev {pn['wiener_dev_db']:.2f} dB")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_synthetic()
        sys.exit()

    # tone_cap.bin: RX at 433.0 MHz, tone +298 kHz, TX image coincides with RX image
    # tone_ret.bin: RX at 433.2 MHz, tone +98 kHz, TX image at -498 kHz -> RX image alone
    res = [analyse_tone("tone_cap.bin"), analyse_tone("tone_ret.bin")]
    for r in res:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
        if r["clip"] > 0:
            print("  CLIPPING: lower hackrf -x and recapture")
        if r["floor_frac"] > 0.3:
            print("  phase floor dominates the fit band: sigma_w is biased; use sigma_w_raw as upper bound")
        if abs(r["wiener_dev_db"]) > 1.0:
            print("  slope deviates from 1/f^2 in the fit band: Wiener model questionable here")

    rx = res[1]
    print("\ntheta_hat:")
    print(f"  a       = {rx['a']:.5f}   (tone_ret only)")
    print(f"  psi_deg = {rx['psi_deg']:.4f}  (tone_ret only)")
    print(f"  IRR_RX  = {rx['irr_model_db']:.2f} dB;  tone_cap (RX+TX) = {res[0]['irr_model_db']:.2f} dB")
    print(f"  sigma_w = {np.median([r['sigma_w'] for r in res]):.4e}  (UPPER BOUND, TX+RX)")

    try:
        I, Q, clip = load("mod.bin")
        print("\nloading:", loading(I, Q), "clip", clip)
    except FileNotFoundError:
        print("\nmod.bin absent: quantization in `calibrated` = `quantization` condition (state it).")
