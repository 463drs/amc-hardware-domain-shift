import numpy as np

fs = 2_048_000
n = fs * 4                       # 4 s loop
f_tone = 300_000.25              # f_tone * 4 s = 1 200 001 cycles: seamless, period = whole file
rng = np.random.default_rng(0)

t = np.arange(n) / fs
x = 100 * np.exp(2j * np.pi * f_tone * t)
tpdf = lambda: rng.uniform(-0.5, 0.5, n) + rng.uniform(-0.5, 0.5, n)   # ±1 LSB

iq = np.empty(2 * n, dtype=np.int8)
iq[0::2] = np.round(x.real + tpdf())
iq[1::2] = np.round(x.imag + tpdf())
iq.tofile("tone_300k.bin")
print("ready:", iq.nbytes / 1e6, "MB")