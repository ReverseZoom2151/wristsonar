"""From a continuous recording to the frames a model is allowed to see.

Three decisions live here.

Segmentation. Chirps go out back to back, so a frame is one chirp period and
frame k of the recording is the response to chirp k. This assumes the emitted
and recorded streams are already aligned, which on a real device they are not:
there is a hardware output-to-input latency of some tens of samples. That
alignment is a calibration problem belonging to the device layer, and pushing
it in here would give this layer a device dependency it is specifically
designed not to have. What this layer guarantees is that if the caller hands
over an aligned recording, lag zero is range zero.

The differential profile. Most of the energy coming back is not the hand. It
is the device body, the strap, and the wrist itself, all of which are
stationary with respect to the transducers and therefore identical frame to
frame. Subtracting consecutive profiles removes them. What survives is what
moved, which is the hand, and this is what most of this literature actually
feeds a model. The cost is real and worth stating: differencing is a
high-pass filter in time, so a hand held perfectly still becomes invisible.
These systems track motion and infer pose, they do not measure a static hand.

Normalisation. See `Normalisation`, which is blunt about what each mode
destroys.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from wristsonar.signal.chirp import DEFAULT_TUKEY_ALPHA, analytic_reference
from wristsonar.signal.matched_filter import bin_metres_for, matched_filter
from wristsonar.types import ChirpConfig, EchoProfile

__all__ = [
    "Normalisation",
    "differential_profiles",
    "normalise",
    "profile_from_frame",
    "profiles_from_recording",
    "segment_frames",
]


class Normalisation(StrEnum):
    """What to do about the fact that absolute amplitude is not trustworthy.

    Speaker and microphone response in the 18 to 21 kHz band varies by tens of
    dB between device models, and volume settings move it again within one
    device. A model trained on raw amplitude will learn the device. So
    normalising is usually right. But it is a deletion, not a cleanup, and the
    thing deleted is physical: echo amplitude carries reflector cross-section
    and spreading loss, which is to say roughly how big the reflector is and
    how far away. After normalisation only the shape of the profile survives,
    and any range information the model recovers comes from where the peaks
    are, never from how loud they are.
    """

    NONE = "none"
    """Leave amplitude alone. Correct within one session on one device."""

    PEAK = "peak"
    """Divide by the largest magnitude, so the strongest echo reads 1.

    Cheap and interpretable, but a single loud bin sets the scale for the
    whole frame, so one specular glint rescales everything else.
    """

    L2 = "l2"
    """Divide by the Euclidean norm of the frame.

    Every bin contributes to the scale, so it does not hinge on one sample.
    Preferred default when frames will be batched into a model.
    """


def normalise(
    samples: NDArray[np.float64], mode: Normalisation = Normalisation.NONE
) -> NDArray[np.float64]:
    """Apply a normalisation mode, leaving an all-zero frame untouched.

    An all-zero frame is left alone rather than turned into NaN, because it
    occurs legitimately in a differential profile of a perfectly static scene
    and a NaN there would propagate into a whole training batch.
    """
    if mode is Normalisation.NONE:
        return samples
    if mode is Normalisation.PEAK:
        scale = float(np.max(np.abs(samples)))
    else:
        scale = float(np.linalg.norm(samples))
    if scale <= 0.0 or not np.isfinite(scale):
        return samples
    return np.asarray(samples / scale, dtype=np.float64)


def segment_frames(
    recording: NDArray[np.float64],
    config: ChirpConfig,
    *,
    hop: int | None = None,
) -> NDArray[np.float64]:
    """Cut a recording into chirp-length frames, shape (n_frames, n_samples).

    Any trailing partial frame is dropped rather than zero-padded. A padded
    frame produces a matched filter output that decays towards the far bins
    for a reason that is not in the scene, and one such frame at the end of
    every recording is a systematic artefact that a model will happily learn.
    """
    if recording.ndim != 1:
        raise ValueError("a recording is one dimensional")
    n = config.n_samples
    step = n if hop is None else hop
    if step <= 0:
        raise ValueError("hop must be positive")
    if recording.size < n:
        return np.zeros((0, n), dtype=np.float64)
    n_frames = 1 + (recording.size - n) // step
    starts = np.arange(n_frames) * step
    frames = np.stack([recording[s : s + n] for s in starts])
    return np.asarray(frames, dtype=np.float64)


def profile_from_frame(
    frame: NDArray[np.float64],
    reference: NDArray[np.complex128],
    sample_rate: int,
    timestamp_s: float,
    *,
    n_bins: int | None = None,
    mode: Normalisation = Normalisation.NONE,
) -> EchoProfile:
    """Matched filter one frame and wrap it with its range calibration."""
    profile = normalise(matched_filter(frame, reference, n_bins=n_bins), mode)
    return EchoProfile(
        samples=profile.astype(np.float32),
        bin_metres=bin_metres_for(sample_rate),
        timestamp_s=timestamp_s,
    )


def profiles_from_recording(
    recording: NDArray[np.float64],
    config: ChirpConfig,
    *,
    hop: int | None = None,
    n_bins: int | None = None,
    mode: Normalisation = Normalisation.NONE,
    t0: float = 0.0,
    alpha: float = DEFAULT_TUKEY_ALPHA,
) -> list[EchoProfile]:
    """The full absolute-profile path: segment, matched filter, calibrate.

    Defaults to no normalisation because the usual next step is
    `differential_profiles`, and per-frame normalisation before differencing
    would rescale each frame independently and leave a residue of the static
    background proportional to how much the scale moved. Normalise after
    differencing, not before.
    """
    reference = analytic_reference(config, alpha)
    frames = segment_frames(recording, config, hop=hop)
    step = config.n_samples if hop is None else hop
    dt = step / config.sample_rate
    return [
        profile_from_frame(
            frame,
            reference,
            config.sample_rate,
            t0 + i * dt,
            n_bins=n_bins,
            mode=mode,
        )
        for i, frame in enumerate(frames)
    ]


def differential_profiles(
    profiles: Sequence[EchoProfile],
    *,
    mode: Normalisation = Normalisation.NONE,
) -> list[EchoProfile]:
    """Frame-to-frame differences, suppressing everything that did not move.

    Returns one fewer profile than it is given, timestamped with the later
    frame of each pair, since the difference describes motion that had
    happened by then rather than motion at the midpoint.

    The result is signed. Taking a magnitude here would be tempting and wrong:
    the sign distinguishes a reflector arriving in a bin from one leaving it,
    which is the only direction information the profile carries at all.
    """
    if len(profiles) < 2:
        return []
    bin_metres = profiles[0].bin_metres
    n_bins = profiles[0].n_bins
    for p in profiles:
        if p.bin_metres != bin_metres or p.n_bins != n_bins:
            raise ValueError(
                "cannot difference profiles with different range calibrations"
            )
        if p.differential:
            raise ValueError("cannot difference an already differential profile")
    out: list[EchoProfile] = []
    for prev, cur in itertools.pairwise(profiles):
        diff = cur.samples.astype(np.float64) - prev.samples.astype(np.float64)
        out.append(
            EchoProfile(
                samples=normalise(diff, mode).astype(np.float32),
                bin_metres=bin_metres,
                timestamp_s=cur.timestamp_s,
                differential=True,
            )
        )
    return out
