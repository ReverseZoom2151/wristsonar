"""Synthesis of the linear FMCW sweep and of the reference used to match it.

Two things in here are choices rather than transcriptions of a formula, so
they are argued for rather than asserted.

The window. An unwindowed sweep is gated on and off at full amplitude, and
that discontinuity splatters energy right across the spectrum, including well
below the 18 kHz floor that makes the sweep inaudible in the first place. A
device that clicks eighty times a second is not a device anyone wears, so the
taper is not primarily a signal-processing nicety here.

It does also buy sidelobe suppression, and the size of that gain is worth
being precise about because the intuition oversells it. Measured on the
default configuration by `tests/signal/test_chirp.py`: far sidelobes of the
correlation drop from about -22 dB unwindowed to -24 dB at alpha 0.25 and
-40 dB at alpha 0.5, but the *first* sidelobe stays near -13 dB whatever the
taper. That last part surprises people. The textbook picture, where an
amplitude taper in time becomes a spectral taper because an LFM sweep maps
time onto frequency, needs a large time-bandwidth product to hold. This sweep
has BT = 3000 * 0.0125 = 37.5, which is small, so its spectrum is dominated by
Fresnel ripple rather than being the clean rectangle the picture assumes. The
practical consequence is downstream: peak detection has to tolerate a -13 dB
sidelobe no matter what is done here.

Tukey rather than Hann, and alpha 0.25 rather than more. Hann tapers the whole
sweep and broadens the correlation mainlobe from the theoretical c/2B of
5.7 cm to 10.7 cm, an 87 percent loss of the one quantity this project is
already short of. Tukey at alpha 0.25 tapers an eighth of the sweep at each
end and costs 12 percent (6.4 cm). Given that the first sidelobe does not
improve either way, paying 87 percent for far sidelobes nobody is
sidelobe-limited by would be a bad trade.

The reference. Matched filtering is correlation against the transmitted
waveform, and the transmitted waveform is real. But the quantity that carries
range is the envelope of the correlation, not the correlation itself, whose
carrier oscillates at 19.5 kHz and crosses zero every 26 microseconds. Rather
than correlating real against real and taking a Hilbert transform afterwards,
this module exposes the analytic reference directly: correlating a real frame
against the analytic reference yields, exactly, the analytic signal of the
real correlation. The proof is short and the consequence is that the envelope
comes out of the same FFT that does the correlation. `matched_filter` relies
on this identity and `tests/signal/test_matched_filter.py` pins it.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from wristsonar.types import ChirpConfig

__all__ = [
    "DEFAULT_TUKEY_ALPHA",
    "analytic_reference",
    "chirp_window",
    "linear_chirp",
    "sample_chirp_at",
    "windowed_chirp",
]

DEFAULT_TUKEY_ALPHA = 0.25
"""Fraction of the sweep spent tapering, split between the two ends."""


def _tukey_at(u: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Tukey window evaluated at normalised sweep position u in [0, 1].

    Written as a closed form in continuous u rather than pulled from
    `scipy.signal.windows.tukey` because the simulator has to evaluate the
    transmitted waveform, window included, at fractional sample delays. A
    window that only exists on a sample grid would force an interpolation
    right where sub-bin accuracy is being tested, which is exactly the kind of
    circularity that makes a test agree with a bug.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"tukey alpha must lie in [0, 1], got {alpha}")
    w = np.ones_like(u)
    if alpha == 0.0:
        return w
    half = alpha / 2.0
    rise = u < half
    fall = u > 1.0 - half
    w[rise] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (u[rise] - half)))
    w[fall] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (u[fall] - 1.0 + half)))
    return w


def chirp_window(
    config: ChirpConfig, alpha: float = DEFAULT_TUKEY_ALPHA
) -> NDArray[np.float64]:
    """The amplitude taper applied to one sweep, sampled on the frame grid.

    Uses the periodic convention u = n/N rather than the symmetric n/(N-1),
    so that the sampled window and the continuous window used by the
    simulator are the same function and not two functions that nearly agree.
    """
    n = config.n_samples
    u = np.arange(n, dtype=np.float64) / float(n)
    return _tukey_at(u, alpha)


def _phase(config: ChirpConfig, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Instantaneous phase of a linear up-sweep, in radians.

    f(t) = f_start + (B/T) t, so phase is its integral. The quadratic term is
    where all the range information lives: it is what makes the
    autocorrelation compress to 1/B rather than 1/f.
    """
    rate = config.bandwidth / config.duration_s
    return 2.0 * np.pi * (config.f_start * t + 0.5 * rate * t * t)


def sample_chirp_at(
    config: ChirpConfig,
    t: NDArray[np.float64],
    alpha: float = DEFAULT_TUKEY_ALPHA,
) -> NDArray[np.float64]:
    """The windowed transmitted waveform evaluated at arbitrary times.

    Zero outside [0, duration). Exposed because the forward model needs to
    place an echo at a fractional sample delay, and evaluating the closed form
    at shifted times is exact where resampling a sampled chirp is not.
    """
    t = np.asarray(t, dtype=np.float64)
    inside = (t >= 0.0) & (t < config.duration_s)
    u = np.zeros_like(t)
    np.divide(t, config.duration_s, out=u, where=inside)
    out = np.sin(_phase(config, np.where(inside, t, 0.0))) * _tukey_at(u, alpha)
    return np.asarray(np.where(inside, out, 0.0), dtype=np.float64)


def linear_chirp(config: ChirpConfig) -> NDArray[np.float64]:
    """The bare unwindowed sweep, unit amplitude.

    Kept public mainly so the cost of not windowing can be measured rather
    than argued about.
    """
    t = np.arange(config.n_samples, dtype=np.float64) / config.sample_rate
    return np.asarray(np.sin(_phase(config, t)), dtype=np.float64)


def windowed_chirp(
    config: ChirpConfig, alpha: float = DEFAULT_TUKEY_ALPHA
) -> NDArray[np.float64]:
    """What the speaker is actually asked to emit."""
    return linear_chirp(config) * chirp_window(config, alpha)


def analytic_reference(
    config: ChirpConfig, alpha: float = DEFAULT_TUKEY_ALPHA
) -> NDArray[np.complex128]:
    """The matched-filter reference, as an analytic (one-sided) signal.

    `scipy.signal.hilbert` returns x + i H(x), so the real part is exactly
    `windowed_chirp` and no information is added. What it buys is that the
    correlation of a real frame against this reference is already the analytic
    correlation, whose magnitude is the range envelope.
    """
    return np.asarray(
        scipy.signal.hilbert(windowed_chirp(config, alpha)), dtype=np.complex128
    )
