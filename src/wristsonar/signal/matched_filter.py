"""Matched filtering: from a received frame to range-resolved echo energy.

Four details decide whether this module is correct, and all four are the kind
that produce plausible wrong answers rather than exceptions.

Zero padding. Correlation by FFT is circular. Without padding, energy at a lag
near the end of the frame wraps around and appears at a near lag, which in
this application means a reflector at 2 m appearing as a fingertip at 5 cm.
The transform length is therefore at least len(frame) + len(reference) - 1,
rounded up to a length the FFT likes.

The lag mapping. Lag k samples is a round-trip flight time of k/fs seconds
over an acoustic path of 2r metres, so r = c*k/(2*fs) and one bin is
bin_metres = c/(2*fs). At 48 kHz that is 3.57 mm of reflector range. Note
what this is not: it is not the range resolution, which is c/2B and about
57 mm at the default 3 kHz sweep. Bin spacing is how finely the answer is
written down, resolution is how finely two answers can be told apart, and
conflating the two is how this literature ends up quoting millimetres.

The envelope. The raw correlation oscillates at the 19.5 kHz carrier and
crosses zero every 26 microseconds, so its samples near the peak say more
about carrier phase than about range. The envelope is the quantity that
carries range. It is obtained here for free: correlating a real frame against
the analytic reference gives the analytic signal of the real correlation, so
its magnitude is the Hilbert envelope without a second transform. Sketch of
why: with r_a = r + iH(r), corr(x, r_a) = C - i*corr(x, H(r)), and
corr(x, H(r)) = -H(C), so the result is C + iH(C). The identity is exact for
an ideal analytic reference and holds to about 1e-5 relative for one built by
Hilbert transforming a finite record, which is what is done here. Doing it
this way round also gets the operation order right: the envelope is taken on
the full correlation and then truncated to the frame, where a Hilbert
transform applied after truncation would ring off the cut.

Sub-bin interpolation. One bin is 3.57 mm and the mainlobe is tens of bins
wide, so quantising to the nearest bin throws away most of the available
precision for no reason. Parabolic interpolation on the log magnitude of the
three samples around the peak is the standard estimator: a mainlobe is
approximately Gaussian near its apex, a Gaussian is exactly a parabola in the
log domain, and so the fit is not merely a smoothing convenience but the right
model. `tests/signal/test_interpolation.py` measures that it beats bin
quantisation rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft
import scipy.signal
from numpy.typing import NDArray

from wristsonar.types import SPEED_OF_SOUND

__all__ = [
    "RangePeak",
    "bin_metres_for",
    "correlate_fft",
    "envelope",
    "matched_filter",
    "parabolic_offset",
    "peak_ranges",
    "range_axis",
    "strongest_peak",
]

_LOG_FLOOR = 1e-300
"""Keeps log of an exactly zero bin finite. Well below any real amplitude."""


def bin_metres_for(sample_rate: int) -> float:
    """Reflector range represented by one correlation lag, in metres.

    The factor of two is the round trip. Carrying it here rather than at the
    call sites is what lets `EchoProfile.crop` be written in metres of range
    without every caller re-deriving the same halving.
    """
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    return SPEED_OF_SOUND / (2.0 * sample_rate)


def range_axis(n_bins: int, sample_rate: int) -> NDArray[np.float64]:
    """Reflector range of each bin, in metres."""
    if n_bins < 0:
        raise ValueError("n_bins cannot be negative")
    return np.arange(n_bins, dtype=np.float64) * bin_metres_for(sample_rate)


def correlate_fft(
    frame: NDArray[np.float64], reference: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Cross-correlate a frame against a reference at all non-negative lags.

    Returns c[k] = sum_n frame[n + k] * conj(reference[n]) for k in
    [0, len(frame)), which is the lag convention in which a target at delay d
    produces a peak at index d. Negative lags are physically meaningless here
    (an echo cannot precede its chirp) so they are not returned, but they are
    still padded for rather than folded on top of the positive lags.

    The result is divided by the energy of the real part of the reference, so
    an isolated reflector of amplitude a produces a peak of magnitude a. That
    calibration is not cosmetic: it turns the amplitude-linearity test into a
    statement about an absolute number instead of a ratio of two outputs, and
    a ratio would still pass under a shared scale error.
    """
    if frame.ndim != 1 or reference.ndim != 1:
        raise ValueError("correlate_fft operates on one dimensional signals")
    if frame.size == 0 or reference.size == 0:
        raise ValueError("cannot correlate an empty signal")
    n_full = frame.size + reference.size - 1
    n_fft = int(scipy.fft.next_fast_len(n_full))
    spectrum = scipy.fft.fft(frame.astype(np.float64), n_fft) * np.conjugate(
        scipy.fft.fft(reference.astype(np.complex128), n_fft)
    )
    corr = scipy.fft.ifft(spectrum)[: frame.size]
    energy = float(np.sum(np.real(reference) ** 2))
    if energy <= 0.0:
        raise ValueError("reference has no energy")
    return np.asarray(corr / energy, dtype=np.complex128)


def envelope(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hilbert envelope of a real signal.

    Provided for the equivalence test against the analytic-reference path
    rather than because the pipeline needs it. If the two ever disagree, one
    of them has picked up a sign convention from a library update.
    """
    return np.asarray(np.abs(scipy.signal.hilbert(signal)), dtype=np.float64)


def matched_filter(
    frame: NDArray[np.float64],
    reference: NDArray[np.complex128],
    *,
    n_bins: int | None = None,
) -> NDArray[np.float64]:
    """Range-resolved echo envelope for one frame.

    `n_bins` truncates the output. The natural choice is the frame length,
    beyond which the correlation is running off the end of the recorded data
    and its amplitude decays for reasons that have nothing to do with the
    scene.
    """
    profile = np.abs(correlate_fft(frame, reference))
    if n_bins is not None:
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        profile = profile[:n_bins]
        if profile.size < n_bins:
            profile = np.pad(profile, (0, n_bins - profile.size))
    return np.asarray(profile, dtype=np.float64)


def parabolic_offset(samples: NDArray[np.float64], index: int) -> float:
    """Sub-bin offset of a peak, in bins, from a three-point log-parabolic fit.

    Returns a value in roughly (-0.5, 0.5). Zero at the array edges, where
    there is no bracketing triple: refusing to extrapolate is the honest
    behaviour, since a peak at bin 0 usually means the true peak is outside
    the window entirely.
    """
    if index <= 0 or index >= samples.size - 1:
        return 0.0
    y0, y1, y2 = (
        float(np.log(max(float(samples[index - 1]), _LOG_FLOOR))),
        float(np.log(max(float(samples[index]), _LOG_FLOOR))),
        float(np.log(max(float(samples[index + 1]), _LOG_FLOOR))),
    )
    curvature = y0 - 2.0 * y1 + y2
    if curvature >= 0.0:
        # Not a concave apex, so the sample is not really a peak. Better to
        # report the bin centre than a wild extrapolation from a flat top.
        return 0.0
    offset = 0.5 * (y0 - y2) / curvature
    return float(np.clip(offset, -0.5, 0.5))


@dataclass(frozen=True, slots=True)
class RangePeak:
    """One detected reflector, with the evidence for believing in it."""

    bin_index: int
    """Nearest bin, before interpolation."""

    range_m: float
    """Sub-bin interpolated reflector range, in metres."""

    amplitude: float
    """Interpolated envelope magnitude, calibrated to reflector amplitude."""

    peak_to_noise: float
    """Amplitude over the median of the profile.

    Carried on every peak rather than checked once and discarded, because the
    low-SNR failure mode of a matched filter is not silence, it is a confident
    peak in the wrong bin. A caller that never looks at this number will get
    exactly that, and a caller that does can tell the two apart.
    """


def _interpolated(
    profile: NDArray[np.float64], index: int, bin_metres: float, floor: float
) -> RangePeak:
    offset = parabolic_offset(profile, index)
    amplitude = float(profile[index])
    if 0 < index < profile.size - 1 and offset != 0.0:
        y0 = np.log(max(float(profile[index - 1]), _LOG_FLOOR))
        y2 = np.log(max(float(profile[index + 1]), _LOG_FLOOR))
        amplitude = float(
            np.exp(np.log(max(amplitude, _LOG_FLOOR)) - 0.25 * (y0 - y2) * offset)
        )
    return RangePeak(
        bin_index=index,
        range_m=(index + offset) * bin_metres,
        amplitude=amplitude,
        peak_to_noise=amplitude / floor if floor > 0.0 else float("inf"),
    )


def _noise_floor(profile: NDArray[np.float64]) -> float:
    """Median of the profile, as a robust stand-in for the noise level.

    Median rather than mean because a handful of real reflectors would drag a
    mean upward and quietly raise the detection bar in exactly the frames that
    contain something.
    """
    return float(np.median(profile))


def peak_ranges(
    profile: NDArray[np.float64],
    bin_metres: float,
    *,
    min_peak_to_noise: float = 4.0,
    min_prominence_ratio: float = 0.3,
    max_peaks: int | None = None,
) -> list[RangePeak]:
    """Detected reflectors, strongest first.

    `min_prominence_ratio` is measured against the strongest peak in the
    frame, and it is what decides whether two nearby reflectors are reported
    as one or two. That makes it the knob on which the project's central
    physical claim rests, so it is a named argument with a documented default
    rather than a threshold buried in the body. The default of 0.3, about
    -10.5 dB, sits deliberately above the first correlation sidelobe, which
    `tests/signal/test_chirp.py` measures at roughly -13 dB and which no
    amount of windowing removes at this sweep's time-bandwidth product. Set it
    lower and the sidelobes of one reflector become two extra reflectors.
    """
    if bin_metres <= 0:
        raise ValueError("bin_metres must be positive")
    if profile.size < 3:
        return []
    floor = _noise_floor(profile)
    peak_max = float(np.max(profile))
    if peak_max <= 0.0:
        return []
    height = min_peak_to_noise * floor if floor > 0.0 else 0.0
    # Guard the ends with zeros before searching. Prominence is measured down
    # to whichever base is higher, and at the array edge that search runs out
    # of array rather than out of signal. Concretely: a reflector 3 cm away
    # sits eight bins in, its mainlobe is eighteen bins wide, so its left
    # skirt is cut off at bin zero at 70 percent of peak height and its
    # apparent prominence collapses to 0.29. It would disappear entirely, and
    # a hand-tracking system losing reflectors specifically when they come
    # close to the watch is not a subtle problem. Zero is the physically
    # correct extension: causality forbids an echo before lag zero.
    padded = np.concatenate(([0.0], profile, [0.0]))
    found, _ = scipy.signal.find_peaks(
        padded,
        height=height if height > 0.0 else None,
        prominence=min_prominence_ratio * peak_max,
    )
    indices = [int(i) - 1 for i in found if 0 <= int(i) - 1 < profile.size]
    peaks = [_interpolated(profile, i, bin_metres, floor) for i in indices]
    peaks.sort(key=lambda p: p.amplitude, reverse=True)
    if max_peaks is not None:
        peaks = peaks[:max_peaks]
    return peaks


def strongest_peak(
    profile: NDArray[np.float64],
    bin_metres: float,
    *,
    min_peak_to_noise: float = 4.0,
) -> RangePeak | None:
    """The single most likely reflector, or None if nothing clears the floor.

    Returning None is the point. A matched filter always has a global maximum,
    and reporting it unconditionally is how a noise-only frame becomes a
    confident hand position.
    """
    if bin_metres <= 0:
        raise ValueError("bin_metres must be positive")
    if profile.size == 0:
        return None
    floor = _noise_floor(profile)
    index = int(np.argmax(profile))
    peak = _interpolated(profile, index, bin_metres, floor)
    if peak.amplitude <= 0.0:
        # A silent profile has a global maximum too. It is bin zero, and
        # reporting it would put a phantom reflector against the skin.
        return None
    if peak.peak_to_noise < min_peak_to_noise:
        return None
    return peak
