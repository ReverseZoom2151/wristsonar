"""Shortcut learning guards, on by default and awkward to switch off.

The cautionary case is not hypothetical. The first independent reproduction
study in knee acoustics (arXiv 2405.15085) recorded a single healthy subject
over five days and obtained 96 percent leave-one-session-out accuracy with no
pathology present anywhere in the data. The classifier was not detecting knee
pathology. It was detecting the session, and the protocol had no way to
notice, because leave-one-session-out sounds like a strong protocol right up
until you ask how many people are in it.

The analogous failures in acoustic hand pose sensing:

One participant dominating the test set. A held out set that is 80 percent one
person measures how well the system does on that person. On a leave one user
out fold this is true by construction, which is why the aggregate across folds
is the thing to quote and why this guard fires on a single fold on purpose.

A test label distribution that nearly matches one training subject. If the
poses in the test set sit on top of one training participant's pose cloud,
then identifying that participant is sufficient to score well, and the model
can do that from mounting geometry alone without ever reading a finger. This
is the direct analogue of the knee study: the shortcut is subject identity,
not the target variable.

An evaluation set too small for the precision being claimed. Frames at 80 per
second are not independent samples; the independent unit is roughly the
session. Twelve participants of two sessions each gives twenty four
independent units, and quoting two decimal places of a millimetre off twenty
four units is claiming a precision the data cannot support. The field quotes
4.81 and 10.71 and 14.88 routinely.

Switching a guard off requires a Waiver naming the guard, giving a reason of
at least forty characters, and naming a person. The waiver is then printed in
the report, in the results table, every time. This is intentional friction:
the cheapest path must be to fix the split.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from wristsonar.eval.splits import Dataset, IndexArray, SplitIndices

__all__ = [
    "MIN_WAIVER_REASON_CHARS",
    "GuardFinding",
    "GuardName",
    "GuardReport",
    "GuardViolation",
    "Waiver",
    "check_evaluation_size",
    "check_label_distribution_match",
    "check_participant_concentration",
    "run_guards",
]

MIN_WAIVER_REASON_CHARS = 40


class GuardName(Enum):
    PARTICIPANT_CONCENTRATION = "participant-concentration"
    SINGLE_SUBJECT_LABEL_MATCH = "single-subject-label-match"
    EVALUATION_SIZE = "evaluation-size"


@dataclass(frozen=True, slots=True)
class Waiver:
    """An explicit, attributed decision to ignore a guard.

    The reason length floor is not bureaucracy. A guard is switched off in the
    hour before a deadline, and the friction is aimed at exactly that hour.
    """

    guard: GuardName
    reason: str
    approved_by: str

    def __post_init__(self) -> None:
        if len(self.reason.strip()) < MIN_WAIVER_REASON_CHARS:
            raise ValueError(
                f"a waiver for {self.guard.value} needs a reason of at least "
                f"{MIN_WAIVER_REASON_CHARS} characters explaining why the "
                f"shortcut this guard detects is not present; got "
                f"{len(self.reason.strip())}"
            )
        if not self.approved_by.strip():
            raise ValueError("a waiver must name the person accepting the risk")

    def describe(self) -> str:
        return f"WAIVED {self.guard.value} by {self.approved_by}: {self.reason.strip()}"


@dataclass(frozen=True, slots=True)
class GuardFinding:
    """The outcome of one guard on one evaluation set."""

    guard: GuardName
    passed: bool
    message: str
    detail: dict[str, float | int | str] = field(default_factory=dict)
    waiver: Waiver | None = None

    @property
    def blocking(self) -> bool:
        return not self.passed and self.waiver is None

    def describe(self) -> str:
        if self.passed:
            return f"pass  {self.guard.value}: {self.message}"
        if self.waiver is not None:
            return (
                f"WAIVED {self.guard.value}: {self.message}\n"
                f"         {self.waiver.describe()}"
            )
        return f"FAIL  {self.guard.value}: {self.message}"


class GuardViolation(Exception):  # noqa: N818
    """Raised when a guard fires and has not been waived."""

    def __init__(self, findings: Sequence[GuardFinding]) -> None:
        self.findings = tuple(findings)
        body = "\n".join(f.describe() for f in self.findings if f.blocking)
        super().__init__(
            "shortcut-learning guards fired; the evaluation set cannot support "
            "the number about to be reported.\n" + body
        )


@dataclass(frozen=True, slots=True)
class GuardReport:
    """Every guard's finding for one evaluation set."""

    findings: tuple[GuardFinding, ...]

    @property
    def blocking(self) -> tuple[GuardFinding, ...]:
        return tuple(f for f in self.findings if f.blocking)

    @property
    def waived(self) -> tuple[GuardFinding, ...]:
        return tuple(f for f in self.findings if not f.passed and f.waiver is not None)

    @property
    def clean(self) -> bool:
        return not self.blocking and not self.waived

    def render(self) -> str:
        return "\n".join(f.describe() for f in self.findings)

    def to_json(self) -> list[dict[str, object]]:
        return [
            {
                "guard": f.guard.value,
                "passed": f.passed,
                "waived": f.waiver is not None,
                "message": f.message,
                "detail": dict(f.detail),
            }
            for f in self.findings
        ]


def _mean_pose_per_participant(
    dataset: Dataset, idx: IndexArray
) -> dict[str, NDArray[np.float64]]:
    out: dict[str, NDArray[np.float64]] = {}
    people = {dataset.meta[int(i)].participant for i in idx}
    for person in people:
        own = np.asarray(
            [i for i in idx if dataset.meta[int(i)].participant == person],
            dtype=np.intp,
        )
        out[person] = dataset.poses[own].mean(axis=0)
    return out


def check_participant_concentration(
    dataset: Dataset,
    test: IndexArray,
    *,
    max_share: float = 0.5,
    waiver: Waiver | None = None,
) -> GuardFinding:
    """Fire when one participant dominates the evaluation set.

    max_share defaults to one half, which is already generous: a test set that
    is half one person is a study of that person with a control group.

    For leave one user out this fires on every individual fold, and it is
    supposed to. The quotable number from LOUO is the aggregate over folds, so
    pass the concatenation of every fold's test indices, not one fold's.
    """
    if test.size == 0:
        raise ValueError("cannot assess concentration of an empty test set")
    people = [dataset.meta[int(i)].participant for i in test]
    counts: dict[str, int] = {}
    for person in people:
        counts[person] = counts.get(person, 0) + 1
    top_person, top_count = max(counts.items(), key=lambda kv: kv[1])
    share = top_count / len(people)
    detail: dict[str, float | int | str] = {
        "top_participant": top_person,
        "top_share": float(share),
        "n_participants": len(counts),
        "n_samples": int(test.size),
    }
    if share <= max_share:
        return GuardFinding(
            guard=GuardName.PARTICIPANT_CONCENTRATION,
            passed=True,
            message=(
                f"{len(counts)} participants in the evaluation set, largest "
                f"share {share:.2f} at or under the {max_share:.2f} limit"
            ),
            detail=detail,
        )
    if len(counts) == 1:
        message = (
            f"the evaluation set is entirely participant {top_person!r}. This "
            f"is a single-subject result. Leave-one-session-out on one subject "
            f"is the exact protocol that produced 96 percent accuracy on data "
            f"containing no pathology in arXiv 2405.15085. Aggregate across "
            f"held-out participants before quoting anything."
        )
    else:
        message = (
            f"participant {top_person!r} is {share:.0%} of the evaluation set "
            f"({top_count} of {len(people)} samples), above the "
            f"{max_share:.0%} limit. The reported number is mostly about one "
            f"person."
        )
    return GuardFinding(
        guard=GuardName.PARTICIPANT_CONCENTRATION,
        passed=False,
        message=message,
        detail=detail,
        waiver=waiver,
    )


def check_label_distribution_match(
    dataset: Dataset,
    train: IndexArray,
    test: IndexArray,
    *,
    ratio: float = 0.20,
    waiver: Waiver | None = None,
) -> GuardFinding:
    """Fire when the test poses sit on top of one training subject's cloud.

    Method: take each training participant's mean pose, take the test set's
    mean pose, and compare the distance to the nearest training participant
    against the typical distance between training participants. If the test
    set is much closer to one training subject than training subjects are to
    each other, then recognising that subject is sufficient to score well and
    the target variable is optional. That is the shortcut.

    The floor is one fifth. It has to be well under one because a genuinely
    held out subject drawn from the same population does land nearer to its
    nearest training subject than the typical pair of training subjects, and
    more so the more training subjects there are. With four evenly spread
    training subjects the most separated a new subject can possibly be is
    about a third, so a floor above that would fire on every honest split. One
    fifth catches the case this guard is for, which is a test subject sitting
    almost on top of a training subject.

    Participants appearing on both sides of the split are excluded from the
    comparison set. On a within-user protocol the test poses matching that
    person's own training poses is the protocol working, not a shortcut, and a
    guard that fires on every within-user split is a guard that gets ignored.

    A training set containing a single participant fails outright, without a
    numeric test. There is no population to generalise over, so whatever the
    protocol is called it is a single-subject study.
    """
    if train.size == 0 or test.size == 0:
        raise ValueError("label distribution check needs a non-empty split")
    test_people = set(dataset.participant_of(test))
    all_train_means = _mean_pose_per_participant(dataset, train)
    # Participants who are on both sides are excluded from the comparison set.
    # On a within-user protocol the test distribution matching that person's
    # own training data is the protocol working as intended, not a shortcut.
    # The question this guard asks is whether the test distribution collapses
    # onto some other single subject.
    train_means = {
        person: mean
        for person, mean in all_train_means.items()
        if person not in test_people
    }
    test_mean = dataset.poses[test].mean(axis=0)

    if len(train_means) < 2:
        present = sorted(all_train_means)
        return GuardFinding(
            guard=GuardName.SINGLE_SUBJECT_LABEL_MATCH,
            passed=False,
            message=(
                f"fewer than two training participants are distinct from the "
                f"test set (training participants: {present}). No population "
                f"exists to generalise over, so this is a single-subject study "
                f"whatever the split is labelled."
            ),
            detail={
                "n_train_participants": len(all_train_means),
                "n_comparison_participants": len(train_means),
            },
            waiver=waiver,
        )

    names = sorted(train_means)
    stack = np.stack([train_means[n] for n in names])
    to_test = np.asarray(
        [
            float(np.linalg.norm(stack[i] - test_mean, axis=1).mean())
            for i in range(len(names))
        ]
    )
    pairwise: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise.append(float(np.linalg.norm(stack[i] - stack[j], axis=1).mean()))
    typical = float(np.median(pairwise))
    nearest_idx = int(np.argmin(to_test))
    nearest = float(to_test[nearest_idx])
    detail: dict[str, float | int | str] = {
        "nearest_train_participant": names[nearest_idx],
        "distance_to_nearest_mm": nearest * 1000.0,
        "typical_inter_subject_mm": typical * 1000.0,
        "n_train_participants": len(names),
    }
    if typical <= 0.0:
        return GuardFinding(
            guard=GuardName.SINGLE_SUBJECT_LABEL_MATCH,
            passed=False,
            message=(
                "every training participant has an identical mean pose, so "
                "the participants are not distinguishable and the split "
                "cannot be assessed"
            ),
            detail=detail,
            waiver=waiver,
        )
    observed = nearest / typical
    detail["ratio"] = observed
    if observed > ratio:
        return GuardFinding(
            guard=GuardName.SINGLE_SUBJECT_LABEL_MATCH,
            passed=True,
            message=(
                f"test poses sit {observed:.2f} of the typical inter-subject "
                f"distance from the nearest training participant "
                f"({names[nearest_idx]!r}), above the {ratio:.2f} floor"
            ),
            detail=detail,
        )
    return GuardFinding(
        guard=GuardName.SINGLE_SUBJECT_LABEL_MATCH,
        passed=False,
        message=(
            f"the test label distribution is {observed:.2f} of the typical "
            f"inter-subject distance from training participant "
            f"{names[nearest_idx]!r}, below the {ratio:.2f} floor. Identifying "
            f"that one subject would be enough to score well here, so the "
            f"result is not evidence that finger pose is being recovered."
        ),
        detail=detail,
        waiver=waiver,
    )


def check_evaluation_size(
    errors_mm: NDArray[np.floating],
    cluster_ids: Sequence[str],
    *,
    reported_decimals: int = 2,
    min_clusters: int = 5,
    waiver: Waiver | None = None,
) -> GuardFinding:
    """Fire when the evaluation set cannot support the precision being printed.

    Two failure modes, one guard.

    Too few independent units. Frames at 80 per second are near duplicates, so
    the independent unit is the session, not the frame. min_clusters defaults
    to five, which is low and still catches the single-subject-five-sessions
    shape of arXiv 2405.15085.

    Too many digits. The standard error is computed across clusters, not
    across frames, because computing it across frames divides by a sample size
    that does not exist and produces confidence intervals a hundred times too
    narrow. reported_decimals defaults to two because that is what the field
    prints: 4.81 mm, 10.71 mm, 14.88 mm. Most honest datasets in this field
    do not support two decimals, and saying so is the point.
    """
    errors = np.asarray(errors_mm, dtype=np.float64)
    if errors.ndim == 2:
        errors = errors.mean(axis=1)
    if errors.ndim != 1:
        raise ValueError("errors_mm must be per sample, or per sample per joint")
    if errors.size != len(cluster_ids):
        raise ValueError(
            f"errors_mm has {errors.size} rows but {len(cluster_ids)} cluster "
            f"ids were supplied"
        )
    if errors.size == 0:
        raise ValueError("cannot assess an empty evaluation set")

    groups: dict[str, list[float]] = {}
    for value, cid in zip(errors.tolist(), cluster_ids, strict=True):
        groups.setdefault(cid, []).append(value)
    names = sorted(groups)
    cluster_means = np.asarray([float(np.mean(groups[n])) for n in names])
    k = cluster_means.size

    detail: dict[str, float | int | str] = {
        "n_samples": int(errors.size),
        "n_clusters": k,
        "reported_decimals": reported_decimals,
    }

    if k < min_clusters:
        return GuardFinding(
            guard=GuardName.EVALUATION_SIZE,
            passed=False,
            message=(
                f"{k} independent evaluation units (participant and session "
                f"pairs), under the floor of {min_clusters}. The "
                f"{int(errors.size)} frames are not independent samples; "
                f"consecutive frames at frame rate are near duplicates."
            ),
            detail=detail,
            waiver=waiver,
        )

    sem = float(np.std(cluster_means, ddof=1) / math.sqrt(k)) if k > 1 else float("inf")
    detail["cluster_sem_mm"] = sem
    justified = reported_decimals if sem <= 0.0 else math.floor(-math.log10(2.0 * sem))
    detail["justified_decimals"] = justified

    if reported_decimals <= justified:
        return GuardFinding(
            guard=GuardName.EVALUATION_SIZE,
            passed=True,
            message=(
                f"{k} independent units, cluster standard error {sem:.3f} mm, "
                f"which supports {justified} decimal places and "
                f"{reported_decimals} are being printed"
            ),
            detail=detail,
        )
    return GuardFinding(
        guard=GuardName.EVALUATION_SIZE,
        passed=False,
        message=(
            f"{reported_decimals} decimal places are being printed but the "
            f"cluster standard error of {sem:.3f} mm across {k} independent "
            f"units supports only {justified}. The extra digits are noise "
            f"presented as resolution."
        ),
        detail=detail,
        waiver=waiver,
    )


def run_guards(
    dataset: Dataset,
    indices: SplitIndices,
    errors_mm: NDArray[np.floating],
    *,
    waivers: Sequence[Waiver] = (),
    max_participant_share: float = 0.5,
    label_match_ratio: float = 0.20,
    reported_decimals: int = 2,
    min_clusters: int = 5,
    raise_on_violation: bool = True,
) -> GuardReport:
    """Run every guard, and raise unless each one passes or is waived.

    Called by default from the reporting path. raise_on_violation exists so
    that a caller can collect findings and render them, which the report does,
    but the report then refuses to produce a headline while any guard is
    blocking, so turning the exception off does not buy a clean number.

    errors_mm is the per sample error over the test set, which is
    metrics.joint_errors output or its per sample mean. Clusters are formed as
    participant and session pairs, because that is the independent unit.
    """
    by_name = {w.guard: w for w in waivers}
    clusters = [
        f"{dataset.meta[int(i)].participant}/{dataset.meta[int(i)].session}"
        for i in indices.test
    ]
    findings = (
        check_participant_concentration(
            dataset,
            indices.test,
            max_share=max_participant_share,
            waiver=by_name.get(GuardName.PARTICIPANT_CONCENTRATION),
        ),
        check_label_distribution_match(
            dataset,
            indices.train,
            indices.test,
            ratio=label_match_ratio,
            waiver=by_name.get(GuardName.SINGLE_SUBJECT_LABEL_MATCH),
        ),
        check_evaluation_size(
            errors_mm,
            clusters,
            reported_decimals=reported_decimals,
            min_clusters=min_clusters,
            waiver=by_name.get(GuardName.EVALUATION_SIZE),
        ),
    )
    report = GuardReport(findings=findings)
    if raise_on_violation and report.blocking:
        raise GuardViolation(report.blocking)
    return report
