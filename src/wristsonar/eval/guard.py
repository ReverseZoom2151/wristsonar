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

Every guard is told which split it is judging, because the same observation is
a shortcut under one protocol and the protocol working under another. A test
set made of one participant is a fatal concentration on a cross user split and
is the definition of a cross session split. A guard that fires on every honest
within user split is a guard that gets ignored, and an ignored guard protects
nothing, so the split is an argument rather than something each guard has to
guess from the data.

Switching a guard off requires a Waiver naming the guard, giving a reason with
real content, naming a person, and setting an expiry date. The waiver is then
printed in the report, in the results table, every time, and it stops
suppressing the guard on its expiry date. The expiry is the part a deadline
cannot defeat: a length floor is satisfied by typing for ten seconds, whereas
an expiry means the guard comes back after the deadline has passed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from wristsonar.eval.splits import Dataset, IndexArray, SplitIndices
from wristsonar.protocol import Protocol, Split

__all__ = [
    "MAX_WAIVER_DAYS",
    "MIN_WAIVER_REASON_CHARS",
    "MIN_WAIVER_REASON_WORDS",
    "POPULATION_MIN_CLUSTERS",
    "WITHIN_USER_MIN_CLUSTERS",
    "GuardBinding",
    "GuardFinding",
    "GuardName",
    "GuardReport",
    "GuardViolation",
    "Waiver",
    "check_evaluation_size",
    "check_label_distribution_match",
    "check_participant_concentration",
    "evaluation_clusters",
    "run_guards",
    "split_fingerprint",
]

MIN_WAIVER_REASON_CHARS = 40
MIN_WAIVER_REASON_WORDS = 8
"""Distinct words, not characters.

Forty characters of anything is one held-down key. Requiring distinct words
does not make a reason true, but it does make an empty reason cost about as
much to write as a real one, which removes the only advantage the empty one
had.
"""

MAX_WAIVER_DAYS = 90
"""How long a waiver may suppress a guard before it has to be argued again."""

POPULATION_MIN_CLUSTERS = 5
"""Independent participant and session units required of a population split."""

WITHIN_USER_MIN_CLUSTERS = 2
"""Units required of a within user split, where the unit is a time block.

A within user protocol holds out one session per fold by construction, so it
has exactly one participant-and-session unit and the population floor of five
can never be met. The floor is not lowered to make the number look better: the
clusters it is measured over are contiguous time blocks, which are not
independent, and check_evaluation_size says so in the finding rather than
letting the resulting precision estimate pass for a real one.
"""


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
    expires_on: date
    """The day this waiver stops working.

    A guard switched off in the hour before a deadline stays off forever
    unless something turns it back on, and nothing ever does. An expiry is the
    something. It cannot be satisfied by typing faster, and the cost of
    renewing it falls on the same person who chose to accept the risk.
    """

    def __post_init__(self) -> None:
        reason = self.reason.strip()
        if len(reason) < MIN_WAIVER_REASON_CHARS:
            raise ValueError(
                f"a waiver for {self.guard.value} needs a reason of at least "
                f"{MIN_WAIVER_REASON_CHARS} characters explaining why the "
                f"shortcut this guard detects is not present; got "
                f"{len(reason)}"
            )
        distinct = {word.strip(".,;:()").casefold() for word in reason.split()}
        distinct.discard("")
        if len(distinct) < MIN_WAIVER_REASON_WORDS:
            raise ValueError(
                f"a waiver for {self.guard.value} needs a reason of at least "
                f"{MIN_WAIVER_REASON_WORDS} distinct words saying why the "
                f"shortcut is not present in this evaluation set; got "
                f"{len(distinct)}. A run of one character clears a length "
                f"floor and explains nothing."
            )
        if not self.approved_by.strip():
            raise ValueError("a waiver must name the person accepting the risk")
        today = date.today()
        if self.expires_on <= today:
            raise ValueError(
                f"a waiver for {self.guard.value} expires on {self.expires_on}, "
                f"which is not in the future. A waiver that is already expired "
                f"suppresses nothing, and back-dating one is how a guard gets "
                f"switched off permanently."
            )
        if self.expires_on > today + timedelta(days=MAX_WAIVER_DAYS):
            raise ValueError(
                f"a waiver may run for at most {MAX_WAIVER_DAYS} days; "
                f"{self.expires_on} is further out than that. A guard that is "
                f"off for a year is a guard that has been deleted."
            )

    def is_expired(self, today: date | None = None) -> bool:
        return self.expires_on <= (today or date.today())

    def describe(self) -> str:
        state = " (EXPIRED)" if self.is_expired() else ""
        return (
            f"WAIVED{state} {self.guard.value} by {self.approved_by} "
            f"until {self.expires_on.isoformat()}: {self.reason.strip()}"
        )


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
        """Failing, and not suppressed by a waiver that is still in date."""
        if self.passed:
            return False
        return self.waiver is None or self.waiver.is_expired()

    def describe(self) -> str:
        if self.passed:
            return f"pass  {self.guard.value}: {self.message}"
        if self.waiver is not None and not self.waiver.is_expired():
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


def split_fingerprint(dataset: Dataset, indices: SplitIndices) -> str:
    """Content identity of one partition of one dataset.

    Hashes the sample count, both index lists and every sample's participant,
    session and device key. Two different splits, or the same split over
    different metadata, cannot produce the same digest, which is what lets a
    report refuse a GuardReport that was computed somewhere easier.
    """
    digest = hashlib.sha256()
    digest.update(str(dataset.n_samples).encode("utf-8"))
    digest.update(np.sort(indices.train).tobytes())
    digest.update(np.sort(indices.test).tobytes())
    for row in dataset.meta:
        digest.update(
            f"{row.participant_key}|{row.session_key}|{row.device_key}\n".encode()
        )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GuardBinding:
    """What a GuardReport was computed on.

    Without this a GuardReport is a bag of findings that can be attached to
    any result at all, including one from an easier split, and the attaching
    is invisible. With it, a report can check that the guards in front of it
    were run on the same dataset, the same split and the same held-out samples
    as the rows it is about to headline.
    """

    split: Split
    split_name: str
    dataset: str
    dataset_version: str
    n_train: int
    n_test: int
    fingerprint: str

    def describe(self) -> str:
        return (
            f"guards computed on {self.split_name} "
            f"({self.split.name}, {self.n_test} held-out samples of "
            f"{self.dataset}@{self.dataset_version}) "
            f"fingerprint {self.fingerprint[:12]}"
        )

    def disagreement_with(self, protocol: Protocol) -> str | None:
        """Why these guards do not describe a result under protocol, or None."""
        if self.split is not protocol.split:
            return (
                f"the guards were run on a {self.split.name} split but the "
                f"rows are {protocol.split.name}"
            )
        if self.dataset != protocol.dataset:
            return (
                f"the guards were run on dataset {self.dataset!r} but the rows "
                f"are on {protocol.dataset!r}"
            )
        if self.dataset_version != protocol.dataset_version:
            return (
                f"the guards were run on {self.dataset}@{self.dataset_version} "
                f"but the rows are on {protocol.dataset}@"
                f"{protocol.dataset_version}"
            )
        return None


@dataclass(frozen=True, slots=True)
class GuardReport:
    """Every guard's finding for one evaluation set, bound to that set.

    Both fields are required. An empty findings tuple used to mean "no guard
    found anything", which is indistinguishable from "no guard ever ran", and
    the second one is the state an evaluation harness must never accept.
    """

    findings: tuple[GuardFinding, ...]
    binding: GuardBinding

    def __post_init__(self) -> None:
        present = {f.guard for f in self.findings}
        missing = sorted(g.value for g in GuardName if g not in present)
        if missing:
            raise ValueError(
                f"a guard report must contain a finding for every guard; "
                f"missing {missing}. An empty or partial report is how a result "
                f"passes with guards that were never run."
            )

    @property
    def blocking(self) -> tuple[GuardFinding, ...]:
        return tuple(f for f in self.findings if f.blocking)

    @property
    def waived(self) -> tuple[GuardFinding, ...]:
        return tuple(
            f
            for f in self.findings
            if not f.passed and f.waiver is not None and not f.waiver.is_expired()
        )

    @property
    def clean(self) -> bool:
        return not self.blocking and not self.waived

    def render(self) -> str:
        lines = [f"  {self.binding.describe()}"]
        lines.extend(f.describe() for f in self.findings)
        return "\n".join(lines)

    def to_json(self) -> list[dict[str, object]]:
        return [
            {
                "guard": f.guard.value,
                "passed": f.passed,
                "waived": f.waiver is not None and not f.waiver.is_expired(),
                "message": f.message,
                "detail": dict(f.detail),
                "split": self.binding.split.name,
                "split_name": self.binding.split_name,
                "fingerprint": self.binding.fingerprint,
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
    split: Split,
    max_share: float = 0.5,
    waiver: Waiver | None = None,
) -> GuardFinding:
    """Fire when one participant dominates an evaluation that claims a population.

    max_share defaults to one half, which is already generous: a test set that
    is half one person is a study of that person with a control group.

    Split aware, and this is the whole difficulty. On CROSS_USER and
    CROSS_DEVICE the claim is about people who contributed nothing, so one
    person filling the test set means the number is about that person. For
    leave one user out this fires on every individual fold, and it is supposed
    to: the quotable number from LOUO is the aggregate over folds, so pass the
    concatenation of every fold's test indices, not one fold's.

    On WITHIN_SESSION and CROSS_SESSION a single-participant test set is the
    protocol rather than a defect, because both hold the person fixed on
    purpose. Firing there would mean this guard fires on every honest within
    user split, which is how a guard gets routinely waived and stops
    protecting the case it exists for. It passes instead, and the finding
    states in plain words that the number is about one named person, which is
    the same fact stated where it is useful rather than where it is noise.
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
        "split": split.name,
    }
    if split.is_within_user:
        return GuardFinding(
            guard=GuardName.PARTICIPANT_CONCENTRATION,
            passed=True,
            message=(
                f"{split.name} holds the person fixed by design, so a test set "
                f"concentrated on {top_person!r} ({share:.0%} of "
                f"{len(people)} samples, {len(counts)} participants) is the "
                f"protocol rather than a shortcut. Nothing here supports a "
                f"claim about anyone else; the cross-user split is what would."
            ),
            detail=detail,
        )
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
    split: Split,
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

    Split aware, in the only way that keeps the guard honest.

    On a population split (CROSS_USER, CROSS_DEVICE) every training
    participant is in the comparison set, including any that also appear in
    the test set. An earlier version excluded those, which inverted the guard:
    with the test set sitting on subject T0's pose cloud it fired at 0.00,
    and moving a single one of T0's training frames into the test set made T0
    an excluded participant, leaving the comparison to run against T1 and
    pass at 1.07. Worse contamination produced a cleaner report. A participant
    on both sides of a population split is now the finding, not an exemption
    from it.

    On a within user split (WITHIN_SESSION, CROSS_SESSION) the test poses
    matching the same person's training poses is the protocol working, and
    there is frequently no second training participant at all because
    cross_session_split trains on one person by design. Running an inter
    subject geometry test there produces a failure on every honest split,
    which is a guard nobody reads. It passes with a finding that says what the
    split does and does not support instead.

    A population split with fewer than two training participants fails
    outright, without a numeric test. There is no population to generalise
    over, so whatever the protocol is called it is a single-subject study.
    """
    if train.size == 0 or test.size == 0:
        raise ValueError("label distribution check needs a non-empty split")
    test_people = set(dataset.participant_of(test))
    all_train_means = _mean_pose_per_participant(dataset, train)

    if split.is_within_user:
        return GuardFinding(
            guard=GuardName.SINGLE_SUBJECT_LABEL_MATCH,
            passed=True,
            message=(
                f"{split.name} is a within-user protocol, so the test poses "
                f"sitting near the training poses of {sorted(test_people)} is "
                f"the design rather than a subject-identity shortcut. This "
                f"guard tests inter-subject collapse and has nothing to "
                f"measure here; the cross-user split is where it bites."
            ),
            detail={
                "n_train_participants": len(all_train_means),
                "split": split.name,
                "applicable": "no",
            },
        )

    shared = sorted(test_people & set(all_train_means))
    if shared:
        return GuardFinding(
            guard=GuardName.SINGLE_SUBJECT_LABEL_MATCH,
            passed=False,
            message=(
                f"participants {shared} contribute to both the training and "
                f"the test half of a {split.name} split. Recognising one of "
                f"them is sufficient to score well, and the contamination "
                f"flatters the number, so this is reported as the failure it "
                f"is rather than excluded from the comparison."
            ),
            detail={
                "n_train_participants": len(all_train_means),
                "shared_participants": ", ".join(shared),
                "split": split.name,
            },
            waiver=waiver,
        )

    train_means = all_train_means
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
                "split": split.name,
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
    split: Split,
    reported_decimals: int = 2,
    min_clusters: int | None = None,
    cluster_kind: str = "participant and session pairs",
    waiver: Waiver | None = None,
) -> GuardFinding:
    """Fire when the evaluation set cannot support the precision being printed.

    Two failure modes, one guard.

    Too few independent units. Frames at 80 per second are near duplicates, so
    the independent unit is the session, not the frame. On a population split
    min_clusters is POPULATION_MIN_CLUSTERS, which is low and still catches
    the single-subject-five-sessions shape of arXiv 2405.15085. On a within
    user split there is exactly one held-out session by construction, so the
    caller clusters by contiguous time block instead and the floor is
    WITHIN_USER_MIN_CLUSTERS; cluster_kind is then named in the finding so
    nobody reads a block-based standard error as a session-based one.

    Too many digits. The standard error is computed across clusters, not
    across frames, because computing it across frames divides by a sample size
    that does not exist and produces confidence intervals a hundred times too
    narrow. reported_decimals defaults to two because that is what the field
    prints: 4.81 mm, 10.71 mm, 14.88 mm. Most honest datasets in this field
    do not support two decimals, and saying so is the point.

    A single cluster fails rather than raising. The standard error of one
    observation is infinite, and an earlier version converted that infinity to
    an integer and raised OverflowError from inside the guard, which is the
    same as no guard at all once a caller wraps the call in a try.
    """
    if min_clusters is None:
        min_clusters = (
            POPULATION_MIN_CLUSTERS if split.is_population else WITHIN_USER_MIN_CLUSTERS
        )
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
        "cluster_kind": cluster_kind,
        "min_clusters": min_clusters,
        "split": split.name,
    }

    if k < 2 or k < min_clusters:
        return GuardFinding(
            guard=GuardName.EVALUATION_SIZE,
            passed=False,
            message=(
                f"{k} independent evaluation units ({cluster_kind}), under the "
                f"floor of {max(2, min_clusters)}. The {int(errors.size)} "
                f"frames are not independent samples; consecutive frames at "
                f"frame rate are near duplicates, and the spread across one "
                f"unit is not a standard error of anything."
            ),
            detail=detail,
            waiver=waiver,
        )

    sem = float(np.std(cluster_means, ddof=1) / math.sqrt(k))
    detail["cluster_sem_mm"] = sem
    justified = reported_decimals if sem <= 0.0 else math.floor(-math.log10(2.0 * sem))
    detail["justified_decimals"] = justified
    caveat = (
        ""
        if split.is_population
        else (
            f" These {cluster_kind} come from one held-out session and are not "
            f"independent of each other, so this standard error is optimistic "
            f"and the digits it justifies are an upper bound."
        )
    )

    if reported_decimals <= justified:
        return GuardFinding(
            guard=GuardName.EVALUATION_SIZE,
            passed=True,
            message=(
                f"{k} evaluation units ({cluster_kind}), cluster standard "
                f"error {sem:.3f} mm, which supports {justified} decimal "
                f"places and {reported_decimals} are being printed.{caveat}"
            ),
            detail=detail,
        )
    return GuardFinding(
        guard=GuardName.EVALUATION_SIZE,
        passed=False,
        message=(
            f"{reported_decimals} decimal places are being printed but the "
            f"cluster standard error of {sem:.3f} mm across {k} units "
            f"({cluster_kind}) supports only {justified}. The extra digits are "
            f"noise presented as resolution.{caveat}"
        ),
        detail=detail,
        waiver=waiver,
    )


def evaluation_clusters(
    dataset: Dataset, indices: SplitIndices, *, n_blocks: int = 10
) -> tuple[list[str], str]:
    """The independent-ish unit of this split, and its name.

    A population split clusters by participant and session, which is the
    independent unit. A within user split holds out exactly one session, so
    that clustering leaves one unit and no measure of spread at all. It
    clusters by contiguous time block instead, in timestamp order, and returns
    a name that says so, because a block is not independent of its neighbours
    and every message built from these clusters has to admit that.
    """
    if indices.split.is_population:
        return (
            [
                f"{dataset.meta[int(i)].participant_key}/"
                f"{dataset.meta[int(i)].session_key}"
                for i in indices.test
            ],
            "participant and session pairs",
        )
    order = np.argsort(
        [dataset.meta[int(i)].timestamp_s for i in indices.test], kind="stable"
    )
    blocks = max(1, min(n_blocks, int(indices.test.size)))
    labels = [""] * int(indices.test.size)
    for rank, position in enumerate(order.tolist()):
        labels[position] = f"time-block-{rank * blocks // int(indices.test.size):02d}"
    return labels, "contiguous time blocks within one held-out session"


def run_guards(
    dataset: Dataset,
    indices: SplitIndices,
    errors_mm: NDArray[np.floating],
    protocol: Protocol,
    *,
    waivers: Sequence[Waiver] = (),
    max_participant_share: float = 0.5,
    label_match_ratio: float = 0.20,
    reported_decimals: int = 2,
    min_clusters: int | None = None,
    raise_on_violation: bool = True,
) -> GuardReport:
    """Run every guard, and raise unless each one passes or is waived.

    Called by default from the reporting path. raise_on_violation exists so
    that a caller can collect findings and render them, which the report does,
    but the report then refuses to produce a headline while any guard is
    blocking, so turning the exception off does not buy a clean number.

    errors_mm is the per sample error over the test set, which is
    metrics.joint_errors output or its per sample mean.

    protocol is required rather than inferred. The returned GuardReport carries
    a GuardBinding naming the dataset, the version, the split and a fingerprint
    of the exact held-out samples, and a Report refuses guards whose binding
    does not match its rows. Without that, a guard report computed on an easier
    split attaches to a harder one and nothing notices.
    """
    if protocol.split is not indices.split:
        raise ValueError(
            f"protocol declares {protocol.split.name} but the split "
            f"{indices.name!r} is {indices.split.name}; guards judged against "
            f"the wrong protocol are worse than no guards"
        )
    by_name = {w.guard: w for w in waivers}
    clusters, cluster_kind = evaluation_clusters(dataset, indices)
    findings = (
        check_participant_concentration(
            dataset,
            indices.test,
            split=indices.split,
            max_share=max_participant_share,
            waiver=by_name.get(GuardName.PARTICIPANT_CONCENTRATION),
        ),
        check_label_distribution_match(
            dataset,
            indices.train,
            indices.test,
            split=indices.split,
            ratio=label_match_ratio,
            waiver=by_name.get(GuardName.SINGLE_SUBJECT_LABEL_MATCH),
        ),
        check_evaluation_size(
            errors_mm,
            clusters,
            split=indices.split,
            reported_decimals=reported_decimals,
            min_clusters=min_clusters,
            cluster_kind=cluster_kind,
            waiver=by_name.get(GuardName.EVALUATION_SIZE),
        ),
    )
    report = GuardReport(
        findings=findings,
        binding=GuardBinding(
            split=indices.split,
            split_name=indices.name,
            dataset=protocol.dataset,
            dataset_version=protocol.dataset_version,
            n_train=indices.n_train,
            n_test=indices.n_test,
            fingerprint=split_fingerprint(dataset, indices),
        ),
    )
    if raise_on_violation and report.blocking:
        raise GuardViolation(report.blocking)
    return report
