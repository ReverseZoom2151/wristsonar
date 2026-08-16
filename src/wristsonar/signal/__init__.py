"""The signal layer: chirp synthesis, matched filtering, echo profiles.

Everything here is a pure function of its arguments. No device, no dataset, no
model. That is deliberate: this is the layer where a reproducibility bug hides
for months, because a sign error in a delay or an off-by-one in a lag mapping
produces plausible-looking numbers rather than a crash. Pure functions can be
pinned against a forward model, so `simulate` is a first-class part of the
layer rather than test scaffolding.

The one convention worth stating loudly, because it is the thing that goes
wrong: a range in this layer always means reflector range, the one-way
distance from the transducer to the reflector. The acoustic path is twice
that. `bin_metres` therefore carries the factor of two already, which is what
makes `n_bins * bin_metres` equal `ChirpConfig.max_unambiguous_range`.
"""

from __future__ import annotations

from wristsonar.signal.chirp import (
    analytic_reference,
    chirp_window,
    linear_chirp,
    sample_chirp_at,
    windowed_chirp,
)
from wristsonar.signal.echo_profile import (
    Normalisation,
    differential_profiles,
    normalise,
    profile_from_frame,
    profiles_from_recording,
    segment_frames,
)
from wristsonar.signal.matched_filter import (
    RangePeak,
    bin_metres_for,
    correlate_fft,
    envelope,
    matched_filter,
    parabolic_offset,
    peak_ranges,
    range_axis,
    strongest_peak,
)
from wristsonar.signal.simulate import (
    Reflector,
    simulate_frame,
    simulate_recording,
)

__all__ = [
    "Normalisation",
    "RangePeak",
    "Reflector",
    "analytic_reference",
    "bin_metres_for",
    "chirp_window",
    "correlate_fft",
    "differential_profiles",
    "envelope",
    "linear_chirp",
    "matched_filter",
    "normalise",
    "parabolic_offset",
    "peak_ranges",
    "profile_from_frame",
    "profiles_from_recording",
    "range_axis",
    "sample_chirp_at",
    "segment_frames",
    "simulate_frame",
    "simulate_recording",
    "strongest_peak",
    "windowed_chirp",
]
