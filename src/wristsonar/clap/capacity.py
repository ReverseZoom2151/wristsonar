"""How much a single impulse can tell you, in bits, and what it cannot.

This module exists so the project states its limit numerically instead of
asserting it in prose. The predecessor project, Dapper, claimed 3D arm, wrist
and finger positions from a passive clap. The claim is not merely hard, it is
outside the channel.

Why the swing is silent
-----------------------
Sound from a clap is generated at and after contact: the cavity between the
palms collapses, air is expelled through the gap, and a Helmholtz resonance
rings (Fu et al. 2025). The approach itself radiates essentially nothing. A
hand travelling at a few metres per second is around Mach 0.02, and dipole
radiation from unsteady motion scales as M^6. At M = 0.02 that is a factor of
about 6e-11 against the sonic case. There is no arm trajectory in the
recording because the arm did not make a sound.

Why hand configuration and arm configuration separate
-----------------------------------------------------
Given the contact state, the two are acoustically independent. The resonator
is formed between the palms and knows only its own geometry. The same cupped
clap performed with the arms overhead, at the chest or behind the back
produces the same direct-path spectrum. Room reflections differ, but those
describe the room, and a system that learns them has learned a room rather
than a body.

The budget
----------
A single impulse carries on the order of 5 to 10 bits in total, of which
roughly 1.5 to 2 concern hand configuration. A 7-DOF arm pose quantised at 5
degrees needs about 32 bits. :func:`pose_gap` reports the difference so a
caller can see it rather than take it on faith, and
:func:`assert_within_capacity` turns it into an error at the point where an
over-claim would otherwise be made.

Mutual information is the right currency because it is what a classifier's
confusion matrix already contains. Accuracy alone hides structure: a
classifier that is 60 percent correct while its errors are confined to one
confusable pair carries considerably more information than one that is 60
percent correct with errors spread everywhere. Bits count that difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.protocol import Measurement, Protocol

__all__ = [
    "ARM_DOF",
    "ARM_JOINT_RANGE_DEG",
    "HAND_BITS_RANGE",
    "IMPULSE_BITS_RANGE",
    "POSE_RESOLUTION_DEG",
    "ChannelCapacity",
    "OverclaimError",
    "PoseGap",
    "arm_pose_bits",
    "assert_within_capacity",
    "capacity_measurement",
    "channel_capacity",
    "mutual_information_bits",
    "pose_gap",
]

ARM_DOF = 7
"""Degrees of freedom of a human arm from shoulder to wrist.

Three at the shoulder, one at the elbow, one for forearm pronation and two at
the wrist. Seven is the standard count and is one more than needed to place a
hand, which is why arm pose is redundant and why a single scalar cue cannot
resolve it even in principle.
"""

POSE_RESOLUTION_DEG = 5.0
"""Angular resolution at which a pose claim is being made."""

ARM_JOINT_RANGE_DEG = 120.0
"""Typical usable range per joint. Generous joints and tight ones average out
near this; the bit count moves only logarithmically, so the conclusion does
not depend on the choice. Doubling every range adds 7 bits, which does not
close a 30 bit gap."""

IMPULSE_BITS_RANGE = (5.0, 10.0)
"""Total information a single clap impulse carries, order of magnitude.

Set by the small number of resolvable resonance frequencies, the coarse
bandwidth or damping estimate, an amplitude that is mostly clap speed, and
timing. Not a theorem, a budget.
"""

HAND_BITS_RANGE = (1.5, 2.0)
"""The part of :data:`IMPULSE_BITS_RANGE` that concerns hand configuration.

Consistent with a four-class contact taxonomy recovered imperfectly: two bits
would be a perfect four-way call, and real classifiers land under that.
"""


class OverclaimError(ValueError):
    """Raised when a claimed output exceeds what the measurement can support."""


@dataclass(frozen=True, slots=True)
class ChannelCapacity:
    """Empirical information a classifier delivers, from its confusion matrix."""

    mutual_information_bits: float
    input_entropy_bits: float
    """Entropy of the true labels in the evaluation set."""

    output_entropy_bits: float
    max_bits: float
    """log2 of the number of classes. The ceiling for this label set."""

    accuracy: float
    samples: int

    @property
    def efficiency(self) -> float:
        """Fraction of the label set's ceiling actually delivered."""
        if self.max_bits <= 0.0:
            return 0.0
        return self.mutual_information_bits / self.max_bits

    @property
    def residual_entropy_bits(self) -> float:
        """Bits of the true label still unknown after seeing the prediction."""
        return self.input_entropy_bits - self.mutual_information_bits

    def describe(self) -> str:
        return (
            f"{self.mutual_information_bits:.2f} bits of "
            f"{self.max_bits:.2f} possible "
            f"({100.0 * self.efficiency:.0f} percent), "
            f"accuracy {self.accuracy:.3f}, n={self.samples}"
        )


def mutual_information_bits(confusion: NDArray[np.int64]) -> float:
    """Empirical mutual information between true and predicted labels.

    ``confusion`` has true labels on rows and predictions on columns. The
    joint distribution is the normalised matrix, and

        I(T; P) = sum_ij p_ij log2( p_ij / (p_i. p_.j) )

    with zero terms skipped. This is a plug-in estimate and is biased upward
    on small samples, by roughly (n_rows - 1)(n_cols - 1) / (2 n ln 2) bits.
    For a 4 by 4 matrix with 100 samples that is about 0.065 bits, which is
    small but not nothing when the whole quantity is under 2 bits. Report the
    sample count alongside, which :class:`ChannelCapacity` does.

    A classifier that ignores its input scores near zero here regardless of
    how accurate it looks on an unbalanced set, which is exactly the property
    accuracy lacks.
    """
    counts = np.asarray(confusion, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"confusion matrix must be 2D, got shape {counts.shape}")
    if np.any(counts < 0.0):
        raise ValueError("confusion matrix counts cannot be negative")
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("confusion matrix is empty")

    joint = counts / total
    row = joint.sum(axis=1, keepdims=True)
    column = joint.sum(axis=0, keepdims=True)
    independent = row * column
    mask = (joint > 0.0) & (independent > 0.0)
    if not mask.any():
        return 0.0
    terms = joint[mask] * np.log2(joint[mask] / independent[mask])
    return max(0.0, float(terms.sum()))


def _entropy_bits(probabilities: NDArray[np.float64]) -> float:
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def channel_capacity(confusion: NDArray[np.int64]) -> ChannelCapacity:
    """Summarise what a classifier's confusion matrix actually delivers."""
    counts = np.asarray(confusion, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("confusion matrix must be square")
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("confusion matrix is empty")
    joint = counts / total
    return ChannelCapacity(
        mutual_information_bits=mutual_information_bits(confusion),
        input_entropy_bits=_entropy_bits(joint.sum(axis=1)),
        output_entropy_bits=_entropy_bits(joint.sum(axis=0)),
        max_bits=math.log2(counts.shape[0]),
        accuracy=float(np.trace(counts)) / total,
        samples=round(total),
    )


def arm_pose_bits(
    dof: int = ARM_DOF,
    *,
    resolution_deg: float = POSE_RESOLUTION_DEG,
    range_deg: float = ARM_JOINT_RANGE_DEG,
) -> float:
    """Bits needed to name a joint configuration at a stated resolution.

    ``dof * log2(range_deg / resolution_deg)``. With the defaults, seven
    joints over 120 degrees at 5 degree steps, this is about 32 bits. It
    assumes joints are independent, which overstates the requirement, since
    real arm poses lie on a lower-dimensional manifold of habitual postures.
    Even a generous factor of four for that correlation leaves 30 bits, and
    the gap being argued about is larger than any such correction.
    """
    if dof < 1:
        raise ValueError("dof must be at least 1")
    if resolution_deg <= 0.0 or range_deg <= 0.0:
        raise ValueError("resolution and range must be positive")
    if range_deg < resolution_deg:
        raise ValueError("range cannot be finer than the resolution")
    return float(dof * math.log2(range_deg / resolution_deg))


@dataclass(frozen=True, slots=True)
class PoseGap:
    """The distance between what an event carries and what a pose costs."""

    available_bits: float
    required_bits: float
    dof: int
    resolution_deg: float
    range_deg: float = ARM_JOINT_RANGE_DEG
    """Joint range the requirement was computed over.

    Stored rather than re-read from the module constant, because
    ``achievable_resolution_deg`` used the constant while ``required_bits``
    used whatever the caller passed, so a non-default range produced two
    numbers describing different arms.
    """

    @property
    def deficit_bits(self) -> float:
        return max(0.0, self.required_bits - self.available_bits)

    @property
    def is_feasible(self) -> bool:
        return self.available_bits >= self.required_bits

    @property
    def events_required(self) -> float:
        """Independent impulses that would be needed, if they were independent.

        They are not. Successive claps by the same person in the same posture
        repeat most of their information, so this is a floor on the count and
        not a recipe. It is here to show the scale of the shortfall: a number
        in the tens means the shortfall is structural rather than a matter of
        a better model.
        """
        if self.available_bits <= 0.0:
            return math.inf
        return self.required_bits / self.available_bits

    @property
    def achievable_resolution_deg(self) -> float:
        """Angular resolution the available bits could actually support.

        Solving ``dof * log2(range / r) = available`` for ``r``. Expect a
        number larger than the joint's range, meaning the event does not
        distinguish one arm pose from any other.
        """
        return float(self.range_deg / (2.0 ** (self.available_bits / self.dof)))

    def describe(self) -> str:
        return (
            f"{self.available_bits:.2f} bits available against "
            f"{self.required_bits:.2f} needed for a {self.dof}-DOF pose at "
            f"{self.resolution_deg:g} degrees: short by "
            f"{self.deficit_bits:.2f} bits, a factor of "
            f"{2.0**self.deficit_bits:.3g} in the number of "
            f"distinguishable poses"
        )


def pose_gap(
    available_bits: float,
    *,
    dof: int = ARM_DOF,
    resolution_deg: float = POSE_RESOLUTION_DEG,
    range_deg: float = ARM_JOINT_RANGE_DEG,
) -> PoseGap:
    """Compare measured bits against the cost of an arm pose."""
    if available_bits < 0.0:
        raise ValueError("available_bits cannot be negative")
    return PoseGap(
        available_bits=available_bits,
        required_bits=arm_pose_bits(
            dof, resolution_deg=resolution_deg, range_deg=range_deg
        ),
        dof=dof,
        resolution_deg=resolution_deg,
        range_deg=range_deg,
    )


def assert_within_capacity(
    claimed_bits: float, available_bits: float, *, what: str = "claim"
) -> None:
    """Refuse a claim that needs more information than was measured.

    Call this at the boundary where a number leaves this package for a
    product surface, a paper or a demo. It is deliberately blunt: there is no
    tolerance parameter, because every over-claim in this literature was made
    by someone who felt their case was within tolerance.
    """
    if claimed_bits < 0.0 or available_bits < 0.0:
        raise ValueError("bit counts cannot be negative")
    if claimed_bits > available_bits:
        raise OverclaimError(
            f"{what} needs {claimed_bits:.2f} bits but the measurement "
            f"supports {available_bits:.2f}; short by "
            f"{claimed_bits - available_bits:.2f} bits"
        )


def capacity_measurement(
    confusion: NDArray[np.int64], protocol: Protocol
) -> Measurement:
    """Mutual information as a protocol-stamped measurement.

    Bits are as protocol-dependent as accuracy. A within-session confusion
    matrix from one clapper yields an information figure that says how well
    that person's claps separate from each other on that day, and
    :class:`~wristsonar.protocol.Protocol` keeps that attached to the number.
    """
    return Measurement(
        value=mutual_information_bits(confusion),
        unit="bits",
        protocol=protocol,
        samples=int(np.sum(np.asarray(confusion))),
    )
