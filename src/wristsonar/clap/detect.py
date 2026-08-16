"""Onset detection for impulsive events.

The job is narrow: find the samples at which a short, sharply decaying
acoustic event begins, and hand back a window that starts at the excitation.
Everything downstream fits a resonator to that window, so a window that
starts late or that contains a sustained sound produces a fit that means
nothing.

Three design commitments follow from that.

Adaptive floor. The detector compares a short-time level against a trailing
median of recent levels rather than against a fixed threshold. A recording
made while a fan spins up, or while the wearer walks from a quiet room into a
loud one, moves its noise floor by tens of dB. A fixed threshold either
misses everything after the floor rises or fires continuously.

Decay test. A rising edge alone is not evidence of an impulse. A door buzzer,
a hand dryer and a spoken syllable all have rising edges. What separates a
clap is that its energy collapses within tens of milliseconds, because the
soft tissue of the palms damps the cavity hard (Fu et al. 2025). The detector
measures the drop in mean square level between an early and a late slice of
the candidate window and rejects anything that does not fall far enough.

Rise test. The level must also be well above where it was just before the
candidate. This suppresses re-detection of ripples inside the decay tail of
an event that has already been reported, which the refractory period alone
does not fully handle when two claps land close together.

References
----------
Fu, Kiyama, Liu, Zhang and Jung (2025), Physical Review Research 7, 013259,
on why claps are short: soft-tissue damping, not radiation loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import find_peaks

__all__ = ["ImpulseEvent", "detect_impulses"]

_EPS = 1e-20


@dataclass(frozen=True, slots=True)
class ImpulseEvent:
    """One detected impulsive event and the window cut around it.

    ``index`` is the sample of peak absolute amplitude, which is taken as the
    excitation instant. ``window`` begins a little before that, so that a
    caller who wants to see the attack can, and so that a one-sample error in
    peak picking does not truncate the response.
    """

    index: int
    """Sample index of the peak, in the original recording."""

    time_s: float
    sample_rate: int
    peak_amplitude: float
    onset_strength_db: float
    """Level of the onset frame above the trailing median floor."""

    decay_db: float
    """Drop in mean square level from the early slice to the late slice."""

    rise_db: float
    """Level of the early slice above the slice immediately preceding onset."""

    window: NDArray[np.float64]
    """Samples from ``index - pre`` to ``index + post``."""

    pre_samples: int
    """How many samples of ``window`` precede ``index``."""

    @property
    def response(self) -> NDArray[np.float64]:
        """The window from the excitation instant onward.

        This is what a resonator fit should see. Samples before the peak are
        pre-excitation and carry no information about the cavity.
        """
        return self.window[self.pre_samples :]


def _frame_levels(
    signal: NDArray[np.float64], frame: int, hop: int
) -> NDArray[np.float64]:
    """Root mean square level per frame, in dB."""
    frames = sliding_window_view(signal, frame)[::hop]
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return 20.0 * np.log10(rms + _EPS)


def _trailing_median(levels: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Median of the ``width`` most recent frames, inclusive of the current.

    Causal by construction. A centred filter would let a loud event lift its
    own floor before it arrives, which costs sensitivity to the first of a
    pair of close claps.
    """
    padded = np.pad(levels, (width - 1, 0), mode="edge")
    return np.median(sliding_window_view(padded, width), axis=1)


def _mean_square(signal: NDArray[np.float64], start: int, stop: int) -> float:
    start = max(0, start)
    stop = min(signal.size, stop)
    if stop <= start:
        return _EPS
    return float(np.mean(np.square(signal[start:stop])) + _EPS)


def detect_impulses(
    signal: NDArray[np.float64],
    sample_rate: int,
    *,
    frame_s: float = 0.004,
    hop_s: float = 0.002,
    floor_window_s: float = 0.5,
    threshold_db: float = 10.0,
    refractory_s: float = 0.12,
    pre_s: float = 0.002,
    window_s: float = 0.040,
    early_s: float = 0.008,
    late_s: float = 0.040,
    min_decay_db: float = 6.0,
    rise_reference_s: float = 0.012,
    min_rise_db: float = 6.0,
) -> list[ImpulseEvent]:
    """Find impulsive events in a mono recording.

    Parameters
    ----------
    signal:
        Mono samples. Any amplitude scale; the detector is scale invariant
        because every test is a ratio.
    sample_rate:
        Samples per second.
    frame_s, hop_s:
        Short-time analysis frame and hop. The hop bounds the timing accuracy
        before sample-level refinement, and the refinement searches a few hops
        either side of the frame peak.
    floor_window_s:
        Length of the trailing window whose median is the noise floor
        estimate. Long enough that a single clap cannot move it, short enough
        to follow a real change in the room.
    threshold_db:
        How far a frame must sit above the trailing floor to be a candidate.
    refractory_s:
        Minimum separation between reported events.
    pre_s, window_s:
        Window cut around the peak, before and after.
    early_s, late_s, min_decay_db:
        The decay test. Mean square level over ``[0, early_s)`` after the peak
        must exceed that over ``[early_s, late_s)`` by ``min_decay_db``. A
        sustained sound scores near 0 dB here and is rejected.
    rise_reference_s, min_rise_db:
        The rise test. The early slice must exceed the slice of the same
        length ending ``rise_reference_s`` before the peak.

    Returns
    -------
    Events in time order. Candidates whose window would run past the end of
    the recording are dropped, because a truncated window cannot be tested
    for decay and would be fitted as if it decayed instantly.
    """
    if signal.ndim != 1:
        raise ValueError(f"expected a mono signal, got shape {signal.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if late_s <= early_s:
        raise ValueError("late_s must exceed early_s")

    work = np.asarray(signal, dtype=np.float64)
    work = work - float(np.mean(work))

    frame = max(2, round(frame_s * sample_rate))
    hop = max(1, round(hop_s * sample_rate))
    if work.size < frame + hop:
        return []

    levels = _frame_levels(work, frame, hop)
    floor_width = max(3, round(floor_window_s / hop_s))
    floor = _trailing_median(levels, floor_width)
    excess = levels - floor

    distance = max(1, round(refractory_s / hop_s))
    peaks, _ = find_peaks(excess, height=threshold_db, distance=distance)

    pre = max(1, round(pre_s * sample_rate))
    post = max(2, round(window_s * sample_rate))
    early = max(1, round(early_s * sample_rate))
    late = max(early + 1, round(late_s * sample_rate))
    rise_ref = max(1, round(rise_reference_s * sample_rate))

    events: list[ImpulseEvent] = []
    last_index = -(10**9)
    for frame_index in peaks:
        centre = int(frame_index) * hop + frame // 2
        lo = max(0, centre - 2 * hop)
        hi = min(work.size, centre + 3 * hop)
        if hi <= lo:
            continue
        peak = lo + int(np.argmax(np.abs(work[lo:hi])))

        if peak - last_index < distance * hop:
            continue
        if peak - pre < 0 or peak + post > work.size:
            continue
        if peak + late > work.size:
            continue

        early_ms = _mean_square(work, peak, peak + early)
        late_ms = _mean_square(work, peak + early, peak + late)
        decay_db = 10.0 * np.log10(early_ms / late_ms)
        if decay_db < min_decay_db:
            continue

        before_ms = _mean_square(work, peak - rise_ref - early, peak - rise_ref)
        rise_db = 10.0 * np.log10(early_ms / before_ms)
        if rise_db < min_rise_db:
            continue

        events.append(
            ImpulseEvent(
                index=peak,
                time_s=peak / sample_rate,
                sample_rate=sample_rate,
                peak_amplitude=float(np.abs(work[peak])),
                onset_strength_db=float(excess[frame_index]),
                decay_db=float(decay_db),
                rise_db=float(rise_db),
                window=work[peak - pre : peak + post].copy(),
                pre_samples=pre,
            )
        )
        last_index = peak

    return events
