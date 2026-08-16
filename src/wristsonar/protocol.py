"""How a result was obtained, carried with the result itself.

In acoustic hand-pose sensing the reported millimetre figures are not sensing
resolution. Range resolution is c/2B. This project's default sweep is 18 to
21 kHz, so B is 3 kHz and the resolution is about 5.7 cm. Widening to the full
18 to 22 kHz that consumer hardware allows before the roll-off gets you to
4.3 cm and no further. Either way the figure is wider than a hand.

The published millimetre numbers therefore come from regressing a coarse echo
signature onto the low-dimensional manifold that human hands actually occupy,
because tendon coupling means the twenty-one landmarks tracked here have far
fewer than sixty-three effective degrees of freedom.

That single fact explains the field's behaviour: excellent within-user
accuracy, two to three times degradation on unseen users, near-total immunity
to ambient noise, and collapse on unseen gesture distributions. Performance is
bounded by how well the training distribution covers the poses you will meet,
not by signal to noise ratio.

So the split is not metadata about a number. It is most of the number's
meaning. A bare float is not a result here, and this module makes that
structural rather than a matter of discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

__all__ = [
    "Split",
    "Protocol",
    "Measurement",
    "GroundTruth",
    "WEAKEST_SPLIT",
]


class Split(Enum):
    """Evaluation splits, ordered from weakest claim to strongest.

    The ordinal is the strength of the claim, not an index. Comparing two
    measurements taken under different splits is comparing two different
    questions, and the report layer refuses to do it silently.
    """

    WITHIN_SESSION = 0
    """Train and test drawn from one continuous wearing session.

    The weakest possible claim. The device never moved between fitting the
    model and testing it, so this measures interpolation within a single
    mounting geometry. Published alone, it is close to meaningless.
    """

    CROSS_SESSION = 1
    """Test session is a different wearing of the device by the same person.

    The first honest number. Remounting changes the occlusion geometry that
    the signal depends on, so this is where a system first has to survive
    something real.
    """

    CROSS_USER = 2
    """Test person contributed no training data at all.

    The number that decides whether a system is useful to anyone who did not
    help build it. Expect two to three times the cross-session error.
    """

    CROSS_DEVICE = 3
    """Test recordings come from a device model absent from training.

    Speaker and microphone frequency response varies by tens of dB between
    models in the 18 to 22 kHz band, so this is a distinct failure axis from
    cross-user and is usually not reported at all.
    """

    @property
    def is_honest(self) -> bool:
        """Whether this split alone supports a claim about a real system."""
        return self is not Split.WITHIN_SESSION


WEAKEST_SPLIT = Split.WITHIN_SESSION


class GroundTruth(Enum):
    """What the errors are measured against.

    This matters more than it looks. Several systems in this literature
    supervise against skeletons fitted to monocular video, so their reported
    error is error against another estimator rather than against motion
    capture. A millimetre figure means something different in each case, and
    averaging across them is meaningless.
    """

    OPTICAL_MOCAP = "optical-mocap"
    """Marker-based optical capture. The only option that bounds its own error."""

    DEPTH_TRACKER = "depth-tracker"
    """A depth camera hand tracker, for example Leap Motion."""

    VIDEO_FITTED = "video-fitted"
    """A skeleton or mesh fitted to monocular video. Error against an estimator."""

    SYNTHETIC = "synthetic"
    """Simulated, so exact by construction and honest only about simulation."""


@dataclass(frozen=True, slots=True)
class Protocol:
    """The conditions a measurement was taken under.

    Every field here changes what a number means. None of them is optional
    decoration.
    """

    split: Split
    dataset: str
    dataset_version: str
    ground_truth: GroundTruth
    """How the reference pose was obtained."""

    subjects: int
    """People contributing data to this evaluation."""

    calibration_minutes: float = 0.0
    """Per-user data given to the model before testing.

    The field norm has settled at roughly two minutes, which works and is not
    zero-shot. Reporting a tuned number without this figure overstates the
    system by however much the tuning bought.
    """

    held_out_poses: bool = False
    """True when the test set contains gestures absent from training.

    Manifold interpolation is the mechanism, so the manifold's edge is where
    these systems fail. A benchmark that never leaves the training
    distribution never finds out.
    """

    seed: int | None = None
    """Seed for any randomised splitting, so a split can be reproduced."""

    notes: str = ""

    def __post_init__(self) -> None:
        if self.subjects < 1:
            raise ValueError("a protocol must involve at least one subject")
        if self.calibration_minutes < 0:
            raise ValueError("calibration_minutes cannot be negative")
        if not self.dataset:
            raise ValueError("a protocol must name its dataset")
        if not self.dataset_version:
            raise ValueError(
                "a protocol must pin a dataset version; "
                "an unversioned dataset cannot be reproduced"
            )

    @property
    def is_zero_shot(self) -> bool:
        return self.calibration_minutes == 0.0

    def describe(self) -> str:
        """One line, suitable for stamping onto any table of numbers."""
        cal = (
            "zero-shot"
            if self.is_zero_shot
            else f"{self.calibration_minutes:g} min calibration"
        )
        ood = " held-out-poses" if self.held_out_poses else ""
        return (
            f"split={self.split.name.lower().replace('_', '-')} "
            f"{cal} n={self.subjects} "
            f"gt={self.ground_truth.value} "
            f"data={self.dataset}@{self.dataset_version}{ood}"
        )

    def at_calibration(self, minutes: float) -> Protocol:
        """The same protocol with a different calibration budget.

        Calibration is an axis to sweep, not a setting to pick once. A system
        is honestly described by its curve from zero minutes upward, and that
        curve is what the field currently buries.
        """
        return replace(self, calibration_minutes=minutes)


@dataclass(frozen=True, slots=True)
class Measurement:
    """A number and the conditions that produced it.

    Constructing one requires a protocol, so there is no code path that
    produces a reportable figure without one.
    """

    value: float
    unit: str
    protocol: Protocol
    """Never optional. This is the entire point of the module."""

    samples: int = 0
    """Frames or poses the value was computed over, for weighting and for
    knowing when a figure rests on too little evidence to mean much."""

    def __str__(self) -> str:
        return f"{self.value:.2f} {self.unit}  [{self.protocol.describe()}]"

    @property
    def is_honest(self) -> bool:
        """Whether this measurement alone supports a claim about a system."""
        return self.protocol.split.is_honest

    def comparable_to(self, other: Measurement) -> bool:
        """Whether two measurements answer the same question.

        Same split, same ground truth, same calibration budget, same units.
        Dataset may differ, since cross-dataset comparison is legitimate as
        long as everything else is held.
        """
        return (
            self.unit == other.unit
            and self.protocol.split is other.protocol.split
            and self.protocol.ground_truth is other.protocol.ground_truth
            and self.protocol.calibration_minutes
            == other.protocol.calibration_minutes
        )
