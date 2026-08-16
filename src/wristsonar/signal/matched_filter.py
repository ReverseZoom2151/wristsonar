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
model. `tests/signal/test_matched_filter.py` measures that it beats bin
quantisation rather than assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft
import scipy.signal
from numpy.typing import NDArray

from wristsonar.types import SPEED_OF_SOUND, EchoProfile

__all__ = [
    "RangePeak",
    "bin_metres_for",
    "correlate_fft",
    "envelope",
    "matched_filter",
    "noise_floor",
    "parabolic_offset",
    "peak_ranges",
    "peak_ranges_of",
    "range_axis",
    "strongest_peak",
    "strongest_peak_of",
]

_LOG_FLOOR = 1e-300
"""Keeps log of an exactly zero bin finite. Well below any real amplitude."""

NOISE_QUANTILE = 0.10
"""Fraction of bins within a block assumed to be free of reflector energy.

See :func:`noise_floor`. Ten percent leaves the estimate valid until a scene
fills nine tenths of the window, which is well past anything the crop this
package recommends produces.
"""

NOISE_BLOCK_BINS = 96
"""Bins over which the noise level is treated as constant.

See :func:`noise_floor`. Long enough that a block still contains quiet bins
next to any plausible reflector, short enough that the frame-truncation taper
described in :func:`noise_floor` is nearly flat across one over the part of
the profile a hand lives in: across the first 96 of the default 600 bins it
falls by 8 percent. It steepens badly in the last block, which is exactly why
the blocks are combined by median rather than by minimum.
"""

_QUANTILE_TO_MEDIAN = math.sqrt(2.0 * math.log(2.0)) / math.sqrt(
    -2.0 * math.log(1.0 - NOISE_QUANTILE)
)
"""Rescales the low quantile of a Rayleigh envelope onto its median.

The envelope of circular Gaussian noise is Rayleigh, whose quantile function
is sigma * sqrt(-2 ln(1 - q)), so the q quantile and the median differ by a
factor that depends on q alone and not on the noise level. Multiplying by it
makes the estimator agree with the median it replaces on a noise-only
profile, which is what lets `min_peak_to_noise` keep the meaning it had.
"""


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
    """Reflector range of each bin, in metres, measured from range zero.

    Correct only for an uncropped profile. A cropped one's bin zero is its
    near edge, and this function is not given it, so plot a cropped profile
    against `range_offset_m + range_axis(...)` or against
    :meth:`~wristsonar.types.EchoProfile.range_of_bin` rather than against
    this alone.
    """
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
    """Amplitude over :func:`noise_floor`.

    Carried on every peak rather than checked once and discarded, because the
    low-SNR failure mode of a matched filter is not silence, it is a confident
    peak in the wrong bin. A caller that never looks at this number will get
    exactly that, and a caller that does can tell the two apart.
    """


def _interpolated(
    profile: NDArray[np.float64],
    index: int,
    bin_metres: float,
    floor: float,
    range_offset_m: float,
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
        range_m=range_offset_m + (index + offset) * bin_metres,
        amplitude=amplitude,
        peak_to_noise=amplitude / floor if floor > 0.0 else float("inf"),
    )


def noise_floor(profile: NDArray[np.float64]) -> float:
    """Amplitude below which a bin is taken to hold no reflector.

    Mean is wrong here for the obvious reason: a handful of real reflectors
    drags it upward and raises the detection bar in exactly the frames that
    contain something. Median is the usual fix and it is what this function
    used to return, but median only works while most bins are noise. That
    holds on a full 600 bin profile, where a hand occupies a few tens of bins
    out of six hundred. It does not hold on the 2 cm to 30 cm crop this
    package recommends, which is 78 bins at the default sweep and about five
    range-resolution cells wide, so a hand and the skirts of its own mainlobes
    cover well over half of it. The median then sits on the signal, the
    threshold rises above the peak and the detector reports nothing on
    precisely the window it is aimed at. Measured on a three-reflector hand in
    that crop: peak over floor fell from 193 on the full profile to 1.7 on the
    crop, and both `peak_ranges` and `strongest_peak` went silent.

    So the estimate comes from a low quantile instead, which stays on noise
    for as long as a tenth of the window is free of reflector energy. That
    alone is not enough, because a low quantile has the opposite weakness. A
    full profile's noise level is not flat: at lag k only N - k samples of the
    reference still overlap the frame, so the correlation of noise decays as
    sqrt(1 - k/N) and the last bins are quiet for a reason that is arithmetic
    rather than acoustic. A quantile taken over the whole profile lands in
    that tail and reads far too low. Measured over 200 noise-only frames, a
    global low quantile turned 27 percent false positives at
    `min_peak_to_noise = 4` into 92 percent.

    The floor is therefore estimated per block of :data:`NOISE_BLOCK_BINS`
    bins and the block estimates are combined by median. Within one block the
    taper is nearly constant, so the low quantile measures the local noise;
    across blocks the median picks a representative level rather than the
    quietest end. On the same 200 frames this scores 10 percent, better than
    the median estimator it replaces as well as better than the global
    quantile. A profile shorter than one block, which is what a crop is, falls
    through to a single block and is a plain low quantile, which is correct
    there because a 78 bin window has no room for the taper to matter.

    The block quantile is rescaled by the Rayleigh factor in
    :data:`_QUANTILE_TO_MEDIAN` so that on noise it reproduces the median it
    replaces and thresholds calibrated against the median still mean what they
    meant. That rescaling assumes the low bins are noise, so in a window so
    crowded that even they carry signal it overshoots, and the result is
    capped at the profile maximum to keep `peak_to_noise` at or above one. A
    window that saturated has no floor to measure and the honest report is
    that nothing stands out of it.

    Magnitude is taken first, so a signed differential profile is measured by
    how far its bins move rather than by where its bins happen to sit.
    """
    magnitude = np.abs(profile)
    if magnitude.size == 0:
        return 0.0
    n_blocks = max(1, magnitude.size // NOISE_BLOCK_BINS)
    per_block = [
        float(np.quantile(block, NOISE_QUANTILE))
        for block in np.array_split(magnitude, n_blocks)
    ]
    estimate = float(np.median(per_block)) * _QUANTILE_TO_MEDIAN
    return min(estimate, float(np.max(magnitude)))


def peak_ranges(
    profile: NDArray[np.float64],
    bin_metres: float,
    *,
    min_peak_to_noise: float = 4.0,
    min_prominence_ratio: float = 0.3,
    max_peaks: int | None = None,
    range_offset_m: float = 0.0,
) -> list[RangePeak]:
    """Detected reflectors, strongest first.

    `range_offset_m` is the one-way distance of bin zero, and it must be
    supplied whenever `profile` is the `samples` of a cropped
    :class:`~wristsonar.types.EchoProfile`. Omitting it reads every range low
    by the crop's near edge. Prefer :func:`peak_ranges_of`, which takes the
    profile itself and cannot be called wrongly.

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
    if range_offset_m < 0.0:
        raise ValueError("range_offset_m cannot be negative")
    if profile.size < 3:
        return []
    floor = noise_floor(profile)
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
    peaks = [
        _interpolated(profile, i, bin_metres, floor, range_offset_m) for i in indices
    ]
    peaks.sort(key=lambda p: p.amplitude, reverse=True)
    if max_peaks is not None:
        peaks = peaks[:max_peaks]
    return peaks


def strongest_peak(
    profile: NDArray[np.float64],
    bin_metres: float,
    *,
    min_peak_to_noise: float = 4.0,
    range_offset_m: float = 0.0,
) -> RangePeak | None:
    """The single most likely reflector, or None if nothing clears the floor.

    Returning None is the point. A matched filter always has a global maximum,
    and reporting it unconditionally is how a noise-only frame becomes a
    confident hand position.

    `range_offset_m` carries the same warning as in :func:`peak_ranges`. Use
    :func:`strongest_peak_of` when a profile object is at hand.
    """
    if bin_metres <= 0:
        raise ValueError("bin_metres must be positive")
    if range_offset_m < 0.0:
        raise ValueError("range_offset_m cannot be negative")
    if profile.size == 0:
        return None
    floor = noise_floor(profile)
    index = int(np.argmax(profile))
    peak = _interpolated(profile, index, bin_metres, floor, range_offset_m)
    if peak.amplitude <= 0.0:
        # A silent profile has a global maximum too. It is bin zero, and
        # reporting it would put a phantom reflector against the skin.
        return None
    if peak.peak_to_noise < min_peak_to_noise:
        return None
    return peak


def peak_ranges_of(
    profile: EchoProfile,
    *,
    min_peak_to_noise: float = 4.0,
    min_prominence_ratio: float = 0.3,
    max_peaks: int | None = None,
) -> list[RangePeak]:
    """:func:`peak_ranges` applied to a profile, crop offset included.

    Exists because the two-argument form takes bare `samples` plus
    `bin_metres`, and a caller who has cropped has no way to be reminded that
    a third number is now needed. Every range this returns is measured from
    the transducer, whether or not the profile has been cropped.
    """
    return peak_ranges(
        profile.samples.astype(np.float64),
        profile.bin_metres,
        min_peak_to_noise=min_peak_to_noise,
        min_prominence_ratio=min_prominence_ratio,
        max_peaks=max_peaks,
        range_offset_m=profile.range_offset_m,
    )


def strongest_peak_of(
    profile: EchoProfile, *, min_peak_to_noise: float = 4.0
) -> RangePeak | None:
    """:func:`strongest_peak` applied to a profile, crop offset included."""
    return strongest_peak(
        profile.samples.astype(np.float64),
        profile.bin_metres,
        min_peak_to_noise=min_peak_to_noise,
        range_offset_m=profile.range_offset_m,
    )
