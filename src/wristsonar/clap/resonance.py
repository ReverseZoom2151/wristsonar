"""Resonator fitting, and what the resonance does and does not pin down.

A clap is not a generic broadband transient. Two palms meeting enclose a
cavity, the cavity collapses, air is forced out through the gap between the
hands, and the mass of air in that gap oscillating against the compliance of
the trapped volume is a Helmholtz resonator. Fu et al. (2025) verified this
directly with high-speed imaging of ten volunteers, engineered replicas and
finite element modelling: a Helmholtz model predicts the clap frequency, and
the short duration comes from soft-tissue damping rather than from radiation.

That physics says the right feature extractor is a resonator fit, not a
spectrogram. A second-order all-pole model has exactly two free parameters
once gain is removed, a pole angle and a pole radius, and those map directly
onto the two things the physics offers: which resonance, and how heavily
damped. Jylha and Erkut (DAFx 2008) took the same route, fitting a
second-order all-pole model by Steiglitz-McBride and classifying from the
coefficients, and this module follows them.

The degeneracy, stated up front
-------------------------------
The Helmholtz frequency is

    f = (c / 2 pi) * sqrt(S / (V * L_eff))

with ``c`` the speed of sound, ``S`` the neck (vent) cross-sectional area,
``V`` the enclosed volume and ``L_eff`` the effective neck length including
end correction. A measured ``f`` therefore identifies the single quantity

    S / (V * L_eff)   [1/m^2]

and nothing finer. A large cavity vented through a large gap aliases exactly
onto a small cavity vented through a small gap. Reporting a volume requires
assuming the vent, and any volume this module returns is conditional on the
vent numbers the caller passed in. :func:`helmholtz_geometry_ratio` returns
the quantity that is actually identified; :func:`cavity_volume_m3` returns
the conditional reading and says so.

References
----------
Repp (1987), JASA 81(4):1100-1109. Peltola, Erkut, Cook and Valimaki (2007),
IEEE TASLP 15(3):1021-1029, for per-clap-type resonator parameters with
centre frequencies from about 500 Hz to about 1562 Hz. Jylha and Erkut
(DAFx 2008). Fu, Kiyama, Liu, Zhang and Jung (2025), PRR 7, 013259.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, residuez

from wristsonar.types import SPEED_OF_SOUND

__all__ = [
    "ResonanceFit",
    "ResonanceFitError",
    "cavity_volume_m3",
    "fit_resonance",
    "helmholtz_frequency_hz",
    "helmholtz_geometry_ratio",
    "steiglitz_mcbride_fit",
]

_EPS = 1e-20
_LN_1000 = math.log(1000.0)


class ResonanceFitError(RuntimeError):
    """Raised when no usable resonance can be recovered from a window."""


@dataclass(frozen=True, slots=True)
class ResonanceFit:
    """The dominant resonance of one impulsive event.

    All three of ``centre_frequency_hz``, ``bandwidth_hz`` and
    ``decay_rate_s`` come from a single complex pole pair, so they are not
    independent measurements. Bandwidth and decay rate are the same number in
    two units: ``bandwidth = decay_rate / pi``.
    """

    centre_frequency_hz: float
    bandwidth_hz: float
    """Half-power full width of the resonance."""

    decay_rate_s: float
    """Exponential decay constant alpha in nepers per second."""

    pole_radius: float
    energy: float
    """Mean square of the fitted window. An amplitude proxy, not a
    configuration cue: Fu et al. (2025) found cavity gauge pressure scales
    quadratically with clap speed, so loudness tracks how hard the hands were
    swung and is deliberately kept out of the classifier."""

    residual: float
    """Relative error of the modelled impulse response, in [0, inf)."""

    order: int
    sample_rate: int

    @property
    def q_factor(self) -> float:
        return self.centre_frequency_hz / max(self.bandwidth_hz, _EPS)

    @property
    def t60_s(self) -> float:
        """Time for the resonance to fall 60 dB."""
        return _LN_1000 / max(self.decay_rate_s, _EPS)


def _stabilise(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reflect any pole on or outside the unit circle back inside.

    Least squares on a short, noisy window occasionally returns an unstable
    denominator. Reflecting preserves the magnitude response up to a gain and
    keeps the decay rate interpretable as a decay rather than a growth.
    """
    roots = np.roots(a)
    if roots.size == 0:
        return a
    magnitudes = np.abs(roots)
    if np.all(magnitudes < 1.0):
        return a
    outside = magnitudes >= 1.0
    roots[outside] = 1.0 / np.conjugate(roots[outside])
    fixed = np.real(np.poly(roots))
    return np.asarray(fixed / fixed[0], dtype=np.float64)


def steiglitz_mcbride_fit(
    response: NDArray[np.float64],
    order: int = 2,
    *,
    numerator_order: int = 1,
    iterations: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit ``B(z) / A(z)`` to an impulse response by Steiglitz-McBride.

    The input is treated as a unit impulse at sample zero, which is what a
    struck or vented cavity receives. Each iteration prefilters both the
    assumed input and the observation by ``1 / A`` from the previous
    iteration and re-solves the linear least squares problem, which is the
    standard fix for the bias that plain equation-error identification
    carries. Jylha and Erkut used the same method on the same problem.

    ``numerator_order`` defaults to one rather than zero. The denominator is
    where all the physics lives, and one zero costs nothing in that respect,
    but it lets the model represent a response whose first retained sample is
    not the excitation instant. Windowing never lands exactly on the
    excitation, and with a strictly all-pole numerator that misalignment
    turns into a phase the model cannot express and pays for by biasing the
    pole angle. Set ``numerator_order=0`` to reproduce the strict all-pole
    formulation.

    Returns the numerator (length ``numerator_order + 1``) and the
    denominator (length ``order + 1``, leading coefficient one).
    """
    if order < 2:
        raise ValueError("order must be at least 2 to represent a resonance")
    if numerator_order < 0:
        raise ValueError("numerator_order cannot be negative")
    y = np.asarray(response, dtype=np.float64)
    lag = max(order, numerator_order)
    if y.size < 4 * order + numerator_order:
        raise ResonanceFitError(
            f"window of {y.size} samples is too short for an order {order} fit"
        )
    scale = float(np.max(np.abs(y)))
    if scale <= 0.0:
        raise ResonanceFitError("window is silent")
    y = y / scale

    n = y.size
    impulse = np.zeros(n, dtype=np.float64)
    impulse[0] = 1.0
    a = np.concatenate(([1.0], np.zeros(order, dtype=np.float64)))
    n_numerator = numerator_order + 1
    solution = np.zeros(n_numerator + order, dtype=np.float64)

    for _ in range(max(1, iterations)):
        input_f = np.asarray(lfilter([1.0], a, impulse), dtype=np.float64)
        output_f = np.asarray(lfilter([1.0], a, y), dtype=np.float64)
        rows = n - lag
        design = np.empty((rows, n_numerator + order), dtype=np.float64)
        for k in range(n_numerator):
            design[:, k] = input_f[lag - k : n - k]
        for k in range(1, order + 1):
            design[:, n_numerator + k - 1] = -output_f[lag - k : n - k]
        target = output_f[lag:]
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        a = _stabilise(np.concatenate(([1.0], solution[n_numerator:])))

    b = np.asarray(solution[:n_numerator] * scale, dtype=np.float64)
    return b, a


def fit_resonance(
    window: NDArray[np.float64],
    sample_rate: int,
    *,
    order: int = 2,
    numerator_order: int = 1,
    f_min: float = 200.0,
    f_max: float = 4000.0,
    iterations: int = 5,
    taper_fraction: float = 0.15,
    start_at_peak: bool = True,
) -> ResonanceFit:
    """Fit a resonator to one impulsive window and report the dominant mode.

    Parameters
    ----------
    window:
        Samples covering the event. If ``start_at_peak`` the fit begins at the
        sample of maximum absolute amplitude, since the all-pole model assumes
        the excitation lands at sample zero and any pre-excitation samples
        would be modelled as part of the response.
    order:
        Denominator order. Two is the Jylha and Erkut choice and is what the
        physics supports. Higher orders fit better and generalise worse, and
        the extra poles are not separately meaningful.
    f_min, f_max:
        Band in which a resonance is accepted. Below ``f_min`` is handling
        rumble; above ``f_max`` is the broadband contact click, which carries
        surface texture rather than cavity size.
    taper_fraction:
        Fraction of the window given a raised-cosine fade at the end, to stop
        truncation from being read as an abrupt extra decay.

    Raises
    ------
    ResonanceFitError:
        If no complex pole pair falls inside ``[f_min, f_max]``.
    """
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a mono window, got shape {x.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if start_at_peak and x.size > 0:
        x = x[int(np.argmax(np.abs(x))) :]
    x = x - float(np.mean(x))
    if x.size < 4 * order:
        raise ResonanceFitError(
            f"window of {x.size} samples is too short for an order {order} fit"
        )

    energy = float(np.mean(np.square(x)))

    if taper_fraction > 0.0:
        fade = max(2, round(taper_fraction * x.size))
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade)))
        x = x.copy()
        x[-fade:] *= ramp

    b, a = steiglitz_mcbride_fit(
        x, order=order, numerator_order=numerator_order, iterations=iterations
    )

    unit_impulse = np.zeros(x.size, dtype=np.float64)
    unit_impulse[0] = 1.0
    modelled = np.asarray(lfilter(b, a, unit_impulse), dtype=np.float64)
    denominator = float(np.sum(np.square(x))) + _EPS
    residual = float(np.sum(np.square(x - modelled)) / denominator)

    residues, poles, _ = residuez(b, a)
    best_index = -1
    best_score = -np.inf
    for index, pole in enumerate(poles):
        if np.imag(pole) <= 0.0:
            continue
        frequency = float(np.angle(pole)) * sample_rate / (2.0 * np.pi)
        if not (f_min <= frequency <= f_max):
            continue
        radius = float(np.abs(pole))
        # Height this pole pair contributes at its own resonance, which is the
        # honest way to say "dominant" when order > 2 gives several pairs.
        score = float(np.abs(residues[index])) / max(1.0 - radius, _EPS)
        if score > best_score:
            best_score = score
            best_index = index

    if best_index < 0:
        raise ResonanceFitError(
            f"no complex pole pair between {f_min:g} and {f_max:g} Hz"
        )

    pole = poles[best_index]
    radius = float(np.abs(pole))
    centre = float(np.angle(pole)) * sample_rate / (2.0 * np.pi)
    decay = -math.log(max(radius, _EPS)) * sample_rate
    return ResonanceFit(
        centre_frequency_hz=centre,
        bandwidth_hz=decay / math.pi,
        decay_rate_s=decay,
        pole_radius=radius,
        energy=energy,
        residual=residual,
        order=order,
        sample_rate=sample_rate,
    )


def helmholtz_geometry_ratio(
    frequency_hz: float, *, speed_of_sound: float = SPEED_OF_SOUND
) -> float:
    """The quantity a measured resonance actually identifies, in 1/m^2.

    Inverting ``f = (c / 2 pi) sqrt(S / (V L_eff))`` gives

        S / (V * L_eff) = (2 pi f / c)^2

    This is one number, and it is all the resonance provides. Any statement
    about ``V`` alone requires ``S`` and ``L_eff`` from somewhere else, which
    for a pair of hands means an assumption about how open the gap between
    them was. Two hands cupped loosely with a wide gap and two hands cupped
    tightly with a narrow gap can sit at the same frequency.
    """
    if frequency_hz <= 0.0:
        raise ValueError("frequency must be positive")
    return float((2.0 * math.pi * frequency_hz / speed_of_sound) ** 2)


def helmholtz_frequency_hz(
    volume_m3: float,
    neck_area_m2: float,
    neck_length_m: float,
    *,
    neck_radius_m: float | None = None,
    end_correction: float = 0.85,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> float:
    """Forward Helmholtz model, for generating and for checking round trips.

    ``L_eff = neck_length_m + end_correction * a``, where ``a`` is the neck
    radius, taken as ``sqrt(neck_area_m2 / pi)`` unless given. The 0.85
    coefficient is the standard flanged single-ended correction; a clap vent
    is neither cleanly flanged nor cleanly unflanged, so treat this as a
    parameter with roughly 20 percent slack rather than a constant.
    """
    if volume_m3 <= 0.0 or neck_area_m2 <= 0.0:
        raise ValueError("volume and neck area must be positive")
    if neck_length_m < 0.0:
        raise ValueError("neck length cannot be negative")
    radius = (
        math.sqrt(neck_area_m2 / math.pi) if neck_radius_m is None else neck_radius_m
    )
    effective_length = neck_length_m + end_correction * radius
    if effective_length <= 0.0:
        raise ValueError("effective neck length must be positive")
    return float(
        speed_of_sound
        / (2.0 * math.pi)
        * math.sqrt(neck_area_m2 / (volume_m3 * effective_length))
    )


def cavity_volume_m3(
    frequency_hz: float,
    *,
    neck_area_m2: float = 1.0e-4,
    neck_length_m: float = 0.015,
    neck_radius_m: float | None = None,
    end_correction: float = 0.85,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> float:
    """Cavity volume implied by a resonance, conditional on the vent.

    .. math:: V = \\frac{c^2 S}{4 \\pi^2 f^2 L_{eff}}

    Assumptions, all of which matter:

    1. Lumped Helmholtz behaviour. Valid only while the cavity is small
       against a wavelength. At 1 kHz the wavelength is 34 cm and a cupped
       pair of hands is a few cm, so this holds comfortably.
    2. The vent is a single effective neck of area ``neck_area_m2`` and
       physical length ``neck_length_m``, with end correction as in
       :func:`helmholtz_frequency_hz`. Real hands leak through a ragged slot
       between the thumbs and between the fingers, so this is a lumped stand
       in for a distributed aperture.
    3. Rigid, non-absorbing walls. Palm tissue is neither, which is precisely
       why claps damp fast (Fu et al. 2025). Compliant walls lower the
       measured frequency relative to a rigid cavity of the same volume, so
       this estimate is biased high in volume.
    4. Adiabatic compression of the enclosed air, standard for audio rates.

    The defaults, a 1 cm^2 vent 1.5 cm deep, put a 500 Hz very cupped clap at
    roughly 60 mL, which is the right order for a pair of cupped palms. They
    are plausible, not measured. Because the resonance identifies only
    ``S / (V * L_eff)`` (see :func:`helmholtz_geometry_ratio`), the returned
    volume scales linearly with whatever ``neck_area_m2`` you assume and
    inversely with ``L_eff``. Getting the vent wrong by a factor of two gets
    the volume wrong by a factor of two, and nothing in the recording can tell
    you that it happened. Use this for ordering claps by cavity size, which it
    does reliably, and not as a measurement of anyone's hands.
    """
    ratio = helmholtz_geometry_ratio(frequency_hz, speed_of_sound=speed_of_sound)
    radius = (
        math.sqrt(neck_area_m2 / math.pi) if neck_radius_m is None else neck_radius_m
    )
    effective_length = neck_length_m + end_correction * radius
    if neck_area_m2 <= 0.0 or effective_length <= 0.0:
        raise ValueError("neck area and effective length must be positive")
    return float(neck_area_m2 / (ratio * effective_length))
