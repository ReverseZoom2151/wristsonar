"""Per user calibration as an axis to sweep, not a setting to pick once.

The field has settled on roughly two minutes of per user data and then quotes
the tuned number, usually without the curve and sometimes without the two
minutes. That is the single most common way an acoustic hand pose result
overstates itself, because the gap between the zero minute and the two minute
figure is often larger than the gap between the paper and its baseline.

A system is honestly described by its curve from zero upward. The shape of
that curve is the finding:

A steep drop from zero to one minute means the model has learned a population
prior and is being re-centred onto this person's hand. It is a personalisation
system, and its zero shot number is the one a new user experiences.

A flat curve means either genuine generalisation or, far more often, that the
model was already saturated by whatever it memorised. Check it against the per
subject mean pose baseline before believing the flattering reading.

A curve still falling at twenty minutes means the deployed accuracy depends on
how patient the user is, which is a product fact and belongs in the abstract.

Where the calibration data comes from matters as much as how much of it there
is. Two honest choices are made here and neither is configurable into
dishonesty. Calibration samples are taken in timestamp order from the start of
the held in pool, because a real enrolment is the first two minutes a user
gives you rather than a sample spread evenly across everything they will ever
do. And the pool is asserted disjoint from the test set, because drawing
calibration frames from the frames you are about to score on is a leak that
produces an extremely convincing curve.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol as TypingProtocol

import numpy as np

from wristsonar.eval.metrics import Alignment, MetricSet, PoseArray, evaluate
from wristsonar.eval.splits import Dataset, IndexArray, LeakageError, SplitIndices
from wristsonar.protocol import Measurement, Protocol

__all__ = [
    "CALIBRATION_BUDGETS_MIN",
    "CalibrationCurve",
    "CalibrationPoint",
    "Fitter",
    "sweep_calibration",
]

CALIBRATION_BUDGETS_MIN: tuple[float, ...] = (0.0, 1.0, 2.0, 20.0)
"""Zero, one, two and twenty minutes.

Zero is what a new user gets. Two is the field norm. Twenty is there to show
whether the curve has flattened or whether the quoted number is simply the
point where the authors stopped collecting.
"""


class Fitter(TypingProtocol):
    """Train on the given indices and predict poses for the test indices.

    Deliberately a callable rather than a class so that a model of any shape,
    including one living in another process, can be swept without this module
    knowing anything about it. It must not read dataset.poses at the test
    indices; nothing here can enforce that, which is why the baselines exist
    as a check on results that look too good.
    """

    def __call__(
        self, dataset: Dataset, train: IndexArray, test: IndexArray
    ) -> PoseArray: ...


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """One rung of the curve."""

    minutes_requested: float
    minutes_achieved: float
    n_calibration_samples: int
    metrics: MetricSet
    truncated: bool = False
    """True when the held in pool could not supply the requested budget.

    Reported rather than silently clamped, because a twenty minute point built
    from four minutes of data is not a twenty minute point.
    """

    @property
    def mpjpe(self) -> Measurement:
        return self.metrics.mpjpe

    def describe(self) -> str:
        flag = "  (TRUNCATED)" if self.truncated else ""
        return (
            f"{self.minutes_achieved:5.1f} min  "
            f"n_cal={self.n_calibration_samples:5d}  "
            f"MPJPE {self.metrics.mpjpe.value:7.2f} mm  "
            f"fingertips {self.metrics.breakdown.fingertip_mm:7.2f} mm{flag}"
        )


@dataclass(frozen=True, slots=True)
class CalibrationCurve:
    """The whole sweep. Reported as a curve, never as its best point."""

    points: tuple[CalibrationPoint, ...]
    split_name: str

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a calibration curve needs at least one point")

    @property
    def zero_shot(self) -> CalibrationPoint | None:
        """What a user who has just unboxed the watch experiences."""
        for point in self.points:
            if point.minutes_requested == 0.0:
                return point
        return None

    @property
    def field_norm(self) -> CalibrationPoint | None:
        """The roughly two minute point the literature quotes."""
        for point in self.points:
            if point.minutes_requested == 2.0:
                return point
        return None

    @property
    def best(self) -> CalibrationPoint:
        return min(self.points, key=lambda p: p.metrics.mpjpe.value)

    @property
    def personalisation_gain_mm(self) -> float | None:
        """How much of the tuned result was bought with the user's own time.

        Zero shot MPJPE minus the best MPJPE. Large means the system is a
        personalisation system and its zero shot figure is the one a new user
        meets. None when the sweep did not include a zero minute point, which
        is itself worth saying out loud.
        """
        zero = self.zero_shot
        if zero is None:
            return None
        return zero.metrics.mpjpe.value - self.best.metrics.mpjpe.value

    @property
    def is_monotone(self) -> bool:
        """Whether more calibration never made things worse.

        Not required. A non monotone curve usually means the evaluation set is
        too small for the differences being read off it, which is exactly what
        guard.check_evaluation_size is for.
        """
        values = [p.metrics.mpjpe.value for p in self.points]
        return all(b <= a + 1e-9 for a, b in itertools.pairwise(values))

    def measurements(self) -> tuple[Measurement, ...]:
        return tuple(p.metrics.mpjpe for p in self.points)

    def render(self) -> str:
        lines = [f"calibration curve  [{self.split_name}]"]
        lines.extend("  " + p.describe() for p in self.points)
        gain = self.personalisation_gain_mm
        if gain is None:
            lines.append(
                "  no zero-minute point in this sweep, so the cost of "
                "personalisation is unmeasured"
            )
        else:
            lines.append(
                f"  personalisation buys {gain:.2f} mm; a new user starts at "
                f"the zero-minute row, not the best one"
            )
        return "\n".join(lines)

    def to_json(self) -> list[dict[str, object]]:
        return [
            {
                "minutes_requested": p.minutes_requested,
                "minutes_achieved": p.minutes_achieved,
                "n_calibration_samples": p.n_calibration_samples,
                "truncated": p.truncated,
                "mpjpe_mm": p.metrics.mpjpe.value,
                "pa_mpjpe_mm": p.metrics.pa_mpjpe.value,
                "fingertip_mm": p.metrics.breakdown.fingertip_mm,
                "protocol": p.metrics.mpjpe.protocol.describe(),
            }
            for p in self.points
        ]


def _default_pool(dataset: Dataset, indices: SplitIndices) -> IndexArray:
    """Held in data belonging to the test participants.

    Everything from the test participants that is neither in the test set nor
    already in training. On a leave one user out fold that is empty by
    construction, which is correct: a cross user protocol has no held in data
    for that person, so a non zero calibration budget requires the caller to
    supply a pool explicitly and thereby to admit that the protocol is no
    longer zero shot cross user.
    """
    test_people = set(dataset.participant_of(indices.test))
    used = set(indices.train.tolist()) | set(indices.test.tolist())
    return np.asarray(
        [
            i
            for i in range(dataset.n_samples)
            if dataset.meta[i].participant in test_people and i not in used
        ],
        dtype=np.intp,
    )


def sweep_calibration(
    dataset: Dataset,
    indices: SplitIndices,
    fitter: Fitter,
    protocol: Protocol,
    *,
    sample_seconds: float,
    budgets: Sequence[float] = CALIBRATION_BUDGETS_MIN,
    calibration_pool: IndexArray | None = None,
    alignment: Alignment = Alignment.ROOT,
) -> CalibrationCurve:
    """Evaluate at each calibration budget and return the whole curve.

    sample_seconds is how much wall clock time one sample represents, which is
    what turns a budget in minutes into a number of frames. At the default
    12.5 ms chirp frame this is 0.0125, but a dataset of pose keyframes rather
    than raw frames will have a much larger value and getting it wrong
    silently rescales the entire x axis.

    Each budget moves the first n samples of the held in pool, in timestamp
    order, into training. The protocol for each point is stamped with its own
    calibration_minutes through Protocol.at_calibration, so a Measurement
    lifted off any rung of this curve still says how much of the user's time
    it cost.
    """
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if not budgets:
        raise ValueError("a calibration sweep needs at least one budget")

    pool = (
        _default_pool(dataset, indices)
        if calibration_pool is None
        else np.asarray(calibration_pool, dtype=np.intp)
    )
    if pool.size:
        clash = np.intersect1d(pool, indices.test)
        if clash.size:
            raise LeakageError(
                f"the calibration pool shares {clash.size} samples with the "
                f"test set of {indices.name!r}. Calibrating on the frames you "
                f"are about to score produces a very convincing curve and a "
                f"meaningless one."
            )
        order = np.argsort(
            [dataset.meta[int(i)].timestamp_s for i in pool], kind="stable"
        )
        pool = pool[order]

    truth = dataset.poses[indices.test]
    points: list[CalibrationPoint] = []
    for minutes in budgets:
        if minutes < 0:
            raise ValueError("a calibration budget cannot be negative")
        wanted = round(minutes * 60.0 / sample_seconds)
        if wanted > 0 and pool.size == 0:
            raise ValueError(
                f"budget of {minutes:g} minutes requested but the held-in pool "
                f"for {indices.name!r} is empty. On a leave-one-user-out fold "
                f"that is expected: pass calibration_pool explicitly, and "
                f"accept that the result is no longer zero-shot cross-user."
            )
        taken = min(wanted, int(pool.size))
        extra = pool[:taken]
        achieved = taken * sample_seconds / 60.0
        train = (
            indices.train
            if taken == 0
            else np.sort(np.concatenate([indices.train, extra]))
        )
        pred = fitter(dataset, train, indices.test)
        point_protocol = protocol.at_calibration(achieved)
        points.append(
            CalibrationPoint(
                minutes_requested=float(minutes),
                minutes_achieved=achieved,
                n_calibration_samples=taken,
                metrics=evaluate(pred, truth, point_protocol, alignment=alignment),
                truncated=taken < wanted,
            )
        )
    return CalibrationCurve(points=tuple(points), split_name=indices.name)
