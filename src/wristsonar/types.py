"""Core data types: the chirp, the echo profile, and the hand.

These are the shapes every other module agrees on. They are deliberately
plain, because the interesting constraints live in the physics rather than in
the containers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "JOINT_NAMES",
    "N_JOINTS",
    "SPEED_OF_SOUND",
    "ChirpConfig",
    "EchoProfile",
    "HandPose",
]

SPEED_OF_SOUND = 343.0
"""Metres per second, dry air at 20 C.

Fixed rather than measured. Temperature moves this by roughly 0.6 m/s per
degree, which at 10 cm range is well under the resolution of anything here.
"""


@dataclass(frozen=True, slots=True)
class ChirpConfig:
    """An FMCW sweep, and the sampling that captures its echoes.

    Defaults follow the parameters this literature has converged on: an 18 to
    21 kHz sweep at 48 kHz sampling, in 12.5 ms frames giving about 80 frames
    per second. The band is chosen from both ends. Below 18 kHz it is audible
    to most people; above 21 to 22 kHz consumer speakers and microphones roll
    off hard and the anti-alias filter takes the rest.
    """

    f_start: float = 18_000.0
    f_end: float = 21_000.0
    duration_s: float = 0.0125
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.f_end <= self.f_start:
            raise ValueError("chirp must sweep upward: f_end must exceed f_start")
        if self.duration_s <= 0:
            raise ValueError("chirp duration must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if self.f_end > self.sample_rate / 2:
            raise ValueError(
                f"f_end {self.f_end} Hz exceeds Nyquist "
                f"{self.sample_rate / 2} Hz; the sweep would alias"
            )

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    @property
    def n_samples(self) -> int:
        return round(self.duration_s * self.sample_rate)

    @property
    def frame_rate(self) -> float:
        """Frames per second if chirps are emitted back to back."""
        return 1.0 / self.duration_s

    @property
    def range_resolution(self) -> float:
        """Smallest separation at which two reflectors resolve, in metres.

        c / 2B. At the default 3 kHz bandwidth this is about 5.7 cm, which is
        wider than a hand. Any millimetre-scale output is therefore a learned
        prior over hand shape and not a measurement of geometry, and the
        project says so everywhere rather than quietly benefiting from the
        confusion.
        """
        return SPEED_OF_SOUND / (2.0 * self.bandwidth)

    @property
    def max_unambiguous_range(self) -> float:
        """Furthest one-way reflector distance that fits inside one frame."""
        return SPEED_OF_SOUND * self.duration_s / 2.0

    @property
    def wavelength_at_centre(self) -> float:
        """Metres. Features much smaller than this diffract rather than reflect."""
        centre = 0.5 * (self.f_start + self.f_end)
        return SPEED_OF_SOUND / centre


@dataclass(frozen=True, slots=True)
class EchoProfile:
    """One frame of range-resolved echo energy.

    The array is indexed by range bin. `bin_metres` converts an index to a
    one-way reflector distance, which is what makes a profile interpretable
    without knowing which chirp produced it. See the field below: the whole
    package depends on that convention, and this sentence said "round-trip"
    for a while, which would double every range for anyone who read it and
    stopped there.

    A differential profile, the frame-to-frame difference, is what most of
    this literature actually feeds a model: it suppresses the static
    reflections from the device body and the wrist, leaving what moved.
    """

    samples: NDArray[np.float32]
    bin_metres: float
    """One-way reflector distance represented by one bin.

    One-way, not round-trip. The factor of two is absorbed when the matched
    filter converts correlation lag to range, so a bin index multiplied by
    this value is the distance to the reflector rather than the distance the
    sound travelled. This has to be stated because both conventions are common
    and mixing them silently halves or doubles every range in the system.
    For an uncropped profile of a full frame, `n_bins * bin_metres` therefore
    equals `ChirpConfig.max_unambiguous_range`, which is also one-way. Once
    cropped that identity no longer holds and `max_range` is the quantity to
    read, since it adds the offset back.
    """

    timestamp_s: float
    differential: bool = False
    range_offset_m: float = 0.0
    """One-way distance represented by bin zero.

    Zero for a profile straight out of the matched filter. Cropping sets it,
    because a cropped profile's bin zero is the crop's near edge rather than
    range zero. Without it, every range computed as index times bin_metres
    reads low by exactly the near edge, and a reflector at 10.0 cm cropped to
    a 2 cm near edge reports 8.2 cm. Use `range_of_bin` rather than doing the
    multiplication by hand.
    """

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(
                f"an echo profile is one dimensional, got shape {self.samples.shape}"
            )
        if self.bin_metres <= 0:
            raise ValueError("bin_metres must be positive")
        if self.range_offset_m < 0:
            raise ValueError("range_offset_m cannot be negative")

    @property
    def n_bins(self) -> int:
        return int(self.samples.shape[0])

    def range_of_bin(self, index: float) -> float:
        """One-way reflector distance for a bin index, cropping included.

        Accepts a float so a sub-bin interpolated peak converts correctly.
        """
        return self.range_offset_m + index * self.bin_metres

    @property
    def near_range(self) -> float:
        """One-way distance of the first bin."""
        return self.range_offset_m

    @property
    def max_range(self) -> float:
        """One-way distance just past the last bin."""
        return self.range_offset_m + self.n_bins * self.bin_metres

    def crop(self, near_m: float, far_m: float) -> EchoProfile:
        """Keep only the range window that can contain a hand.

        Cropping is not an optimisation. Bins outside the plausible window
        carry room reflections, and a model handed them will learn the room,
        which is the classic way these systems score well in one place and
        fail in another.

        The result records where its bin zero now sits. Passing the raw
        `samples` of a cropped profile to a helper that takes `bin_metres`
        alone loses that, so prefer `range_of_bin` for anything positional.
        """
        if far_m <= near_m:
            raise ValueError("far_m must exceed near_m")
        near_bins = max(0.0, near_m - self.range_offset_m)
        far_bins = max(0.0, far_m - self.range_offset_m)
        lo = max(0, int(near_bins / self.bin_metres))
        hi = min(self.n_bins, int(far_bins / self.bin_metres))
        if hi <= lo:
            raise ValueError(
                f"crop window {near_m}..{far_m} m selects no bins "
                f"at {self.bin_metres:.4f} m per bin"
            )
        return EchoProfile(
            samples=self.samples[lo:hi],
            bin_metres=self.bin_metres,
            timestamp_s=self.timestamp_s,
            differential=self.differential,
            range_offset_m=self.range_offset_m + lo * self.bin_metres,
        )


JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
"""Twenty-one landmarks: the wrist plus four per digit.

This ordering matches the MediaPipe hand convention, which is the most widely
used and makes comparison against camera-based systems possible without a
remapping table that nobody maintains.
"""

N_JOINTS = len(JOINT_NAMES)


@dataclass(frozen=True, slots=True)
class HandPose:
    """Joint positions in metres, relative to the wrist unless stated.

    Wrist-relative by default because the sensor is on the wrist and has no
    idea where the wrist is in the world. Claiming global position from a
    wrist-mounted sensor would be inventing information.
    """

    joints: NDArray[np.float32]
    """Shape (N_JOINTS, 3)."""

    wrist_relative: bool = True
    handedness: str = "right"

    def __post_init__(self) -> None:
        if self.joints.shape != (N_JOINTS, 3):
            raise ValueError(
                f"expected {(N_JOINTS, 3)} joint array, got {self.joints.shape}"
            )
        if self.handedness not in ("left", "right"):
            raise ValueError(f"handedness must be left or right, got {self.handedness}")

    def joint(self, name: str) -> NDArray[np.float32]:
        try:
            return self.joints[JOINT_NAMES.index(name)]
        except ValueError as exc:
            raise KeyError(f"unknown joint {name!r}") from exc
