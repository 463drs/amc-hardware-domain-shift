"""Receiver-chain distortions R_theta = Q_b . B_d . G_{a,psi} . P_sigma (injected part of
y = R_theta(R_0(s))). Compose order reads along the chain; the notation reads right to left."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np

FRAME_DTYPE = np.complex64

# Numerical guard for a silent frame, so a per-frame reference never divides by zero.
_FS_EPS: float = 1e-12


# I/Q <-> complex helpers

def to_complex(frame: np.ndarray) -> np.ndarray:
    """(T, 2) float32 [I, Q] -> (T,) complex64. Inverse of to_iq."""
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.shape[1] != 2:
        raise ValueError(f"expected a (T, 2) frame, got {frame.shape}")
    return (frame[:, 0].astype(np.float32) + 1j * frame[:, 1].astype(np.float32)).astype(FRAME_DTYPE)


def to_iq(x: np.ndarray) -> np.ndarray:
    """(T,) complex64 -> (T, 2) float32 [I, Q]. Inverse of to_complex."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"expected a (T,) complex frame, got {x.shape}")
    return np.stack((x.real, x.imag), axis=1).astype(np.float32)


# Reference level (AGC model)

class ReferenceLevel(ABC):
    """Full-scale amplitude that makes b bits and offset d meaningful, computed per frame.
    TODO(open): which strategy the six conditions use is NOT decided; set it in the config."""

    # RadioML's 29 dB inter-frame amplitude spread is a SYNTHESIS artefact (SNR is a separate
    # parameter there); per-frame referencing leaves only PAPR, a genuine property of the mod.

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def params(self) -> dict: ...


def _branch_magnitudes(x: np.ndarray) -> np.ndarray:
    """|I| and |Q| pooled -- the two identical ADCs see branches, not |x|."""
    return np.concatenate((np.abs(x.real), np.abs(x.imag)))


class PeakReference(ReferenceLevel):
    """FS = frame peak branch magnitude x headroom, so nothing clips at 0 dB headroom.
    Outlier-sensitive: one stray sample sets FS and starves the rest of that frame."""

    def __init__(self, headroom_db: float = 0.0):
        self.headroom_db = float(headroom_db)

    def __call__(self, x):
        peak = float(_branch_magnitudes(x).max(initial=0.0))
        return max(peak * 10.0 ** (self.headroom_db / 20.0), _FS_EPS)

    def params(self): return {"kind": "peak", "headroom_db": self.headroom_db}


class PercentileReference(ReferenceLevel):
    """FS = high percentile of branch magnitudes x headroom: outlier-robust, and closer to a
    real AGC, which targets a loading factor. Cost: a small deliberate clipped fraction."""

    def __init__(self, percentile: float = 99.9, headroom_db: float = 0.0):
        if not (0.0 < percentile <= 100.0): raise ValueError("percentile must be in (0, 100]")
        self.percentile = float(percentile)
        self.headroom_db = float(headroom_db)

    def __call__(self, x):
        fs = float(np.percentile(_branch_magnitudes(x), self.percentile))
        return max(fs * 10.0 ** (self.headroom_db / 20.0), _FS_EPS)

    def params(self):
        return {"kind": "percentile", "percentile": self.percentile, "headroom_db": self.headroom_db}


class FixedReference(ReferenceLevel):
    """FS = one constant for the whole dataset. Rejected here -- not for clipping, which
    recalibration fixes, but because it bakes a RadioML synthesis artefact in as physics."""

    def __init__(self, full_scale: float):
        if full_scale <= 0: raise ValueError("full_scale must be > 0")
        self.full_scale = float(full_scale)

    def __call__(self, x): return self.full_scale

    def params(self): return {"kind": "fixed", "full_scale": self.full_scale}


REFERENCE_LEVELS = {
    "peak": PeakReference,
    "percentile": PercentileReference,
    "fixed": FixedReference,
}


# Distortion operators

class Distortion(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

    @property
    @abstractmethod
    def is_identity(self) -> bool: ...

    @abstractmethod
    def params(self) -> dict: ...

class Compose(Distortion):
    def __init__(self, distortions): self.distortions = list(distortions)

    def __call__(self, x, rng):
        for distortion in self.distortions:
            x = distortion(x, rng)
        return x

    @property
    def is_identity(self): return all(o.is_identity for o in self.distortions)

    def params(self): return {type(d).__name__: d.params() for d in self.distortions}

class PhaseNoise(Distortion):
    def __init__(self, sigma_w: float):
        if sigma_w < 0: raise ValueError("sigma_w must be >= 0")
        self.sigma_w = sigma_w

    @property
    def is_identity(self): return self.sigma_w == 0.0

    def params(self): return {"sigma_w": self.sigma_w}

    def __call__(self, x, rng):
        if self.is_identity:
            return x
        w = rng.normal(0.0, self.sigma_w, size=x.shape[0])
        phi = np.cumsum(w)
        phi -= phi[0]
        return (x * np.exp(1j * phi)).astype(np.complex64)


class IQImbalance(Distortion):
    """Quadrature mismatch: I is the reference branch, Q carries all of it, with
    I' = I and Q' = g*(Q*cos(psi) + I*sin(psi)), g = 10^(gain_db/20), psi > 0 tilting Q toward I."""

    def __init__(self, gain_db: float, phase_deg: float):
        self.gain_db = float(gain_db)
        self.phase_deg = float(phase_deg)

    @property
    def is_identity(self): return self.gain_db == 0.0 and self.phase_deg == 0.0

    def params(self): return {"gain_db": self.gain_db, "phase_deg": self.phase_deg}

    def coefficients(self) -> tuple[complex, complex]:
        """(alpha, beta) of the equivalent y = alpha*x + beta*conj(x)."""
        g = 10.0 ** (self.gain_db / 20.0)
        psi = np.deg2rad(self.phase_deg)
        return (1.0 + g * np.exp(1j * psi)) / 2.0, (1.0 - g * np.exp(-1j * psi)) / 2.0

    def irr_db(self) -> float:
        """Image rejection ratio implied by (gain_db, phase_deg); the tests assert against this."""
        alpha, beta = self.coefficients()
        return float("inf") if beta == 0 else 20.0 * np.log10(abs(alpha) / abs(beta))

    def __call__(self, x, rng):
        if self.is_identity:
            return x
        alpha, beta = self.coefficients()
        return (alpha * x + beta * np.conj(x)).astype(np.complex64)


class DCOffset(Distortion):
    """Additive per-branch DC as a SIGNED FRACTION of full scale (0.01 == 1% of FS).
    The thesis' scalar d is the offset_i == offset_q case."""

    def __init__(self, offset_i: float, offset_q: float, reference: ReferenceLevel | None = None):
        self.offset_i = float(offset_i)
        self.offset_q = float(offset_q)
        if not self.is_identity and reference is None:
            raise ValueError("DCOffset needs a ReferenceLevel: the offset is a fraction of full scale")
        self.reference = reference

    @property
    def is_identity(self): return self.offset_i == 0.0 and self.offset_q == 0.0

    def params(self):
        return {
            "offset_i": self.offset_i,
            "offset_q": self.offset_q,
            "reference": None if self.reference is None else self.reference.params(),
        }

    def __call__(self, x, rng):
        reference = self.reference
        if self.is_identity or reference is None:
            return x
        fs = reference(x)
        return (x + fs * (self.offset_i + 1j * self.offset_q)).astype(np.complex64)


class Quantize(Distortion):
    """Uniform mid-rise b-bit ADC per branch over [-FS, +FS], clipping to the outermost level.
    n_bits=None bypasses the ADC -- the degenerate value, since no finite b is an identity."""

    def __init__(self, n_bits: int | None, reference: ReferenceLevel | None = None):
        if n_bits is not None:
            if isinstance(n_bits, bool) or not isinstance(n_bits, int):
                raise ValueError(f"n_bits must be an int or None, got {n_bits!r}")
            if n_bits < 1:
                raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        self.n_bits = n_bits
        if not self.is_identity and reference is None:
            raise ValueError("Quantize needs a ReferenceLevel: b bits are defined against full scale")
        self.reference = reference

    @property
    def is_identity(self): return self.n_bits is None

    def params(self):
        return {
            "n_bits": self.n_bits,
            "reference": None if self.reference is None else self.reference.params(),
        }

    def __call__(self, x, rng):
        n_bits, reference = self.n_bits, self.reference
        if n_bits is None or reference is None:
            return x
        fs = reference(x)
        n_levels = 2 ** n_bits
        delta = 2.0 * fs / n_levels

        def q(v):
            # Bin index, clipped; reconstruct at the bin centre (mid-rise, no code word at 0).
            k = np.clip(np.floor((v + fs) / delta), 0, n_levels - 1)
            return (k + 0.5) * delta - fs

        return (q(x.real) + 1j * q(x.imag)).astype(np.complex64)


DISTORTIONS = {
    "phase_noise": PhaseNoise,
    "iq_imbalance": IQImbalance,
    "dc_offset": DCOffset,
    "quantize": Quantize,
}


# Building a chain from configuration

def build_reference(spec: Any) -> ReferenceLevel:
    """Build a ReferenceLevel from a {name, kwargs} mapping, or pass an instance through."""
    if isinstance(spec, ReferenceLevel):
        return spec
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(f"reference must be a mapping with a 'name' key, got {spec!r}")
    name = spec["name"]
    if name not in REFERENCE_LEVELS:
        raise ValueError(f"Unknown reference {name!r}. Available: {sorted(REFERENCE_LEVELS)}.")
    return REFERENCE_LEVELS[name](**spec.get("kwargs", {}))


def build_distortion(spec: Any) -> Distortion:
    """Build one operator from a {name, kwargs} mapping, resolving a nested reference."""
    if isinstance(spec, Distortion):
        return spec
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(f"distortion must be a mapping with a 'name' key, got {spec!r}")
    name = spec["name"]
    if name not in DISTORTIONS:
        raise ValueError(f"Unknown distortion {name!r}. Available: {sorted(DISTORTIONS)}.")
    kwargs = dict(spec.get("kwargs", {}))
    if kwargs.get("reference") is not None:
        kwargs["reference"] = build_reference(kwargs["reference"])
    return DISTORTIONS[name](**kwargs)


def build_compose(specs: Iterable[Any] | None) -> Compose:
    """Build the chain from a list of specs, read along the signal chain."""
    return Compose([build_distortion(s) for s in (specs or [])])


# Per-frame keyed RNG

# Recorded verbatim in every generated file's metadata.
RNG_SCHEME = "np.random.default_rng([blake2b64(condition), frame_index])"


def condition_key(condition: str) -> int:
    """Stable 64-bit key for a condition name; Python's hash() is salted per process."""
    return int.from_bytes(hashlib.blake2b(condition.encode("utf-8"), digest_size=8).digest(), "big")


def frame_rng(condition: str, frame_index: int) -> np.random.Generator:
    """Generator keyed by (condition, GLOBAL frame index), never by iteration order.
    Frame i draws the same numbers whether generated first, last, alone or in another batch."""
    if frame_index < 0: raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    return np.random.default_rng([condition_key(condition), int(frame_index)])


def apply_to_frame(compose: Compose, frame: np.ndarray, condition: str, frame_index: int) -> np.ndarray:
    """Apply a chain to one stored (T, 2) float32 frame, preserving shape and dtype.
    An identity chain skips the complex round-trip, so baseline output is bit-identical."""
    if compose.is_identity:
        return frame
    out = compose(to_complex(frame), frame_rng(condition, frame_index))
    return to_iq(out)
