"""Receiver-chain distortions R_theta = Q_b . B_d . G_{a,psi} . P_sigma (injected part of
y = R_theta(R_0(s))). Compose order reads along the chain; the notation reads right to left."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np

FRAME_DTYPE = np.complex64

ConfigFloat = float | str

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
    """Full-scale amplitude that makes b bits and offset d meaningful, computed ONCE per frame
    by Compose and handed to every consumer -- it is a property of the chain, not of an operator.
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
    def __call__(self, x: np.ndarray, rng: np.random.Generator,
                 fs: float | None = None) -> np.ndarray: ...

    @property
    @abstractmethod
    def is_identity(self) -> bool: ...

    @abstractmethod
    def params(self) -> dict: ...

    @property
    def requires_full_scale(self) -> bool:
        """Does this operator need the frame's full-scale level? Scale-invariant ones do not,
        and it is exactly the invariant ones that may precede the measurement point."""
        return False

    accepts_sample_rate: bool = False


class Compose(Distortion):
    """Applies operators in list order, along the signal chain.

    Full scale is measured ONCE per frame, at a single declared point, and handed to every
    consumer."""

    def __init__(self, distortions: Iterable[Distortion], reference: ReferenceLevel | None = None,
                 fs_after: int | None = None):
        self.distortions = list(distortions)
        self.reference = reference
        if fs_after is not None:
            fs_after = int(fs_after)
            if reference is None:
                raise ValueError("fs_after moves the measurement point, so it needs a reference")
            if not (-1 <= fs_after < len(self.distortions)):
                raise ValueError(
                    f"fs_after must be in [-1, {len(self.distortions) - 1}], got {fs_after}")
            stranded = [type(d).__name__
                        for d in self.distortions[:fs_after + 1] if d.requires_full_scale]
            if stranded:
                raise ValueError(
                    f"fs_after={fs_after} puts {', '.join(stranded)} before the measurement "
                    "point, where there is no full-scale level yet")
        self.fs_after = fs_after

    def _measure_at(self) -> int:
        """Index of the operator BEFORE which full scale is measured; == len means never.
        Default: the first consumer, i.e. immediately after the last scale-invariant operator."""
        if self.fs_after is not None:
            return self.fs_after + 1
        for i, distortion in enumerate(self.distortions):
            if distortion.requires_full_scale:
                return i
        return len(self.distortions)

    def measurement_point(self) -> str | None:
        """The declared point, named for the metadata: the operator FS is measured after."""
        index = self._measure_at()
        if self.reference is None or index >= len(self.distortions):
            return None
        return "input" if index == 0 else type(self.distortions[index - 1]).__name__

    def __call__(self, x, rng, fs=None):
        # An enclosing Compose that already measured wins: a nested chain never re-measures.
        reference = self.reference if fs is None else None
        measure_at = self._measure_at() if reference is not None else None
        for i, distortion in enumerate(self.distortions):
            if reference is not None and i == measure_at:
                fs = reference(x)
            x = distortion(x, rng, fs)
        return x

    @property
    def is_identity(self): return all(o.is_identity for o in self.distortions)

    @property
    def requires_full_scale(self):
        # A chain with its own reference is self-sufficient and asks nothing of its caller.
        return self.reference is None and any(d.requires_full_scale for d in self.distortions)

    def params(self):
        theta = {type(d).__name__: d.params() for d in self.distortions}
        if self.reference is not None:
            # Underscored: this is the chain's AGC model, not one of the operators.
            theta["_reference"] = {**self.reference.params(),
                                   "measured_after": self.measurement_point()}
        return theta


def sigma_w_from_phase_noise(l_dbc_hz: float, offset_hz: float, sample_rate_hz: float) -> float:
    """Wiener increment std (rad/sample) from a datasheet single-sideband phase-noise figure.

    White frequency noise makes L(f) fall as 1/f^2: L(f) = c / (4 pi^2 f^2), and the random walk
    that produces it has per-sample variance sigma_w^2 = c / f_s. So ONE datasheet point -- L at
    a stated offset -- plus the sampling rate fixes sigma_w. Keep the pair in the config and
    sigma_w follows f_s automatically; -98 dBc/Hz at 10 kHz, 1.024 MS/s -> 7.8e-4 rad/sample.

    The same conversion applies to a bench measurement of L, which is how theta-hat will be read
    off the real receiver -- so the config and the measurement stay in the same units."""
    if offset_hz <= 0:
        raise ValueError(f"offset_hz must be > 0, got {offset_hz}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    c = 4.0 * np.pi ** 2 * offset_hz ** 2 * 10.0 ** (l_dbc_hz / 10.0)
    return float(np.sqrt(c / sample_rate_hz))


def _as_float(value: Any, field: str) -> float:
    """Coerce a config number at the boundary. YAML 1.1 reads an unsigned exponent (1.0e4) as a
    STRING, so a physical constant written the obvious way would otherwise reach the arithmetic
    as text; failing here names the field instead of blowing up three frames down."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number, got {value!r}") from None


class PhaseNoise(Distortion):
    """Wiener phase walk phi_n = phi_{n-1} + w_n, w ~ N(0, sigma_w^2), pinned at phi_0 = 0.

    sigma_w is per SAMPLE, so it only means something against a sampling rate. State it either
    directly, or as the oscillator's datasheet figure (phase_noise_dbc_hz at offset_hz), which
    f_s -- declared once for the whole conditions file -- converts. Prefer the datasheet pair:
    theta then records the measured quantities rather than a derivative of them, and changing
    f_s moves sigma_w with it instead of leaving a stale hardcoded number behind."""

    accepts_sample_rate = True

    def __init__(self, sigma_w: ConfigFloat | None = None,
                 phase_noise_dbc_hz: ConfigFloat | None = None,
                 offset_hz: ConfigFloat | None = None,
                 sample_rate_hz: ConfigFloat | None = None):
        self.phase_noise_dbc_hz: float | None = None
        self.offset_hz: float | None = None
        self.sample_rate_hz: float | None = None

        if phase_noise_dbc_hz is not None or offset_hz is not None:
            if sigma_w is not None:
                raise ValueError("give sigma_w OR the datasheet pair (phase_noise_dbc_hz, "
                                 "offset_hz), not both -- they would disagree silently")
            if phase_noise_dbc_hz is None or offset_hz is None:
                raise ValueError("the datasheet spelling needs BOTH phase_noise_dbc_hz and "
                                 "offset_hz: a level without its offset fixes nothing")
            if sample_rate_hz is None:
                raise ValueError("a phase-noise figure is per Hz at an offset, so converting it "
                                 "needs f_s: declare 'sample_rate_hz' at the top level of the "
                                 "conditions file")
            self.phase_noise_dbc_hz = _as_float(phase_noise_dbc_hz, "phase_noise_dbc_hz")
            self.offset_hz = _as_float(offset_hz, "offset_hz")
            self.sample_rate_hz = _as_float(sample_rate_hz, "sample_rate_hz")
            sigma_w = sigma_w_from_phase_noise(
                self.phase_noise_dbc_hz, self.offset_hz, self.sample_rate_hz)
        elif sigma_w is None:
            raise ValueError("PhaseNoise needs sigma_w, or the datasheet pair "
                             "(phase_noise_dbc_hz, offset_hz)")

        self.sigma_w = _as_float(sigma_w, "sigma_w")
        if self.sigma_w < 0: raise ValueError("sigma_w must be >= 0")

    @property
    def is_identity(self): return self.sigma_w == 0.0

    def params(self):
        if self.phase_noise_dbc_hz is None:
            return {"sigma_w": self.sigma_w}
        # The passport values travel alongside the derived one, so theta stays traceable.
        return {"sigma_w": self.sigma_w, "phase_noise_dbc_hz": self.phase_noise_dbc_hz,
                "offset_hz": self.offset_hz, "sample_rate_hz": self.sample_rate_hz}

    def __call__(self, x, rng, fs=None):
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

    def __call__(self, x, rng, fs=None):
        if self.is_identity:
            return x
        alpha, beta = self.coefficients()
        return (alpha * x + beta * np.conj(x)).astype(np.complex64)


class DCOffset(Distortion):
    """Additive per-branch DC as a SIGNED FRACTION of full scale (0.01 == 1% of FS).
    The thesis' scalar d is the offset_i == offset_q case."""

    def __init__(self, offset_i: float, offset_q: float):
        self.offset_i = float(offset_i)
        self.offset_q = float(offset_q)

    @property
    def is_identity(self): return self.offset_i == 0.0 and self.offset_q == 0.0

    @property
    def requires_full_scale(self): return not self.is_identity

    def params(self):
        return {"offset_i": self.offset_i, "offset_q": self.offset_q}

    def __call__(self, x, rng, fs=None):
        if self.is_identity:
            return x
        if fs is None:
            # No silent fallback: measuring FS here is the defect this signature removes.
            raise ValueError("DCOffset requires a full-scale level; it is measured once per "
                             "frame by Compose, at the chain's declared measurement point")
        return (x + fs * (self.offset_i + 1j * self.offset_q)).astype(np.complex64)


class Quantize(Distortion):
    """Uniform mid-rise b-bit ADC per branch over [-FS, +FS], clipping to the outermost level.
    n_bits=None bypasses the ADC -- the degenerate value, since no finite b is an identity."""

    def __init__(self, n_bits: int | None):
        if n_bits is not None:
            if isinstance(n_bits, bool) or not isinstance(n_bits, int):
                raise ValueError(f"n_bits must be an int or None, got {n_bits!r}")
            if n_bits < 1:
                raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        self.n_bits = n_bits

    @property
    def is_identity(self): return self.n_bits is None

    @property
    def requires_full_scale(self): return not self.is_identity

    def params(self):
        return {"n_bits": self.n_bits}

    def __call__(self, x, rng, fs=None):
        n_bits = self.n_bits
        if n_bits is None:
            return x
        if fs is None:
            # No silent fallback: the step must not depend on what preceded the quantizer.
            raise ValueError("Quantize requires a full-scale level; it is measured once per "
                             "frame by Compose, at the chain's declared measurement point")
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


def build_distortion(spec: Any, sample_rate_hz: float | None = None) -> Distortion:
    """Build one operator from a {name, kwargs} mapping. `sample_rate_hz` is the dataset's
    nominal f_s, injected into the operators whose config may be written in datasheet units."""
    if isinstance(spec, Distortion):
        return spec
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(f"distortion must be a mapping with a 'name' key, got {spec!r}")
    name = spec["name"]
    if name not in DISTORTIONS:
        raise ValueError(f"Unknown distortion {name!r}. Available: {sorted(DISTORTIONS)}.")
    kwargs = dict(spec.get("kwargs", {}))
    if "reference" in kwargs:
        raise ValueError(
            f"{name!r} carries a 'reference' kwarg, but full scale is a property of the CHAIN, "
            "measured once per frame: move 'reference' up to the condition level, alongside "
            "'operators'."
        )
    if "sample_rate_hz" in kwargs:
        raise ValueError(
            f"{name!r} carries a 'sample_rate_hz' kwarg, but f_s is a property of the DATASET, "
            "not of an operator: declare 'sample_rate_hz' once at the TOP level of the "
            "conditions file, so every condition is converted against the same rate."
        )
    cls = DISTORTIONS[name]
    if sample_rate_hz is not None and cls.accepts_sample_rate:
        kwargs["sample_rate_hz"] = sample_rate_hz
    return cls(**kwargs)


_CONDITION_KEYS = frozenset({"operators", "reference", "fs_after"})


def build_compose(spec: Iterable[Any] | dict | None,
                  sample_rate_hz: float | None = None) -> Compose:
    """Build the chain, read along the signal chain. `spec` is either a bare list of operator
    specs, or a mapping {operators, reference, fs_after} when the chain needs a full-scale
    level -- which is declared once for the whole chain, not per operator.

    `sample_rate_hz` is the conditions file's top-level f_s, handed down to the operators that
    take datasheet units; without it, such an operator refuses to guess."""
    if spec is None:
        return Compose([])
    reference, fs_after = None, None
    if isinstance(spec, dict):
        unknown = set(spec) - _CONDITION_KEYS
        if unknown:
            raise ValueError(f"unknown condition keys {sorted(unknown)}; "
                             f"expected any of {sorted(_CONDITION_KEYS)}.")
        specs = spec.get("operators") or []
        if spec.get("reference") is not None:
            reference = build_reference(spec["reference"])
        fs_after = spec.get("fs_after")
    else:
        specs = spec
    if isinstance(specs, (str, bytes)):
        raise ValueError(f"operators must be a list of specs, got {specs!r}")

    compose = Compose([build_distortion(s, sample_rate_hz) for s in specs],
                      reference=reference, fs_after=fs_after)
    if compose.requires_full_scale:
        # Loud at build time rather than hours into a generation run.
        consumers = sorted({type(d).__name__ for d in compose.distortions if d.requires_full_scale})
        raise ValueError(f"{', '.join(consumers)} need a full-scale level, so this condition must "
                         "declare a 'reference' (peak / percentile / fixed) at the condition level.")
    return compose


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
