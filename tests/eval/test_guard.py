"""Each guard on a constructed bad case, and quiet on a good one."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset, make_pose
from wristsonar.eval.guard import (
    GuardBinding,
    GuardName,
    GuardReport,
    GuardViolation,
    Waiver,
    check_evaluation_size,
    check_label_distribution_match,
    check_participant_concentration,
    run_guards,
)
from wristsonar.eval.splits import (
    Dataset,
    SampleMeta,
    SplitIndices,
    cross_device_split,
    cross_session_split,
    cross_user_folds,
)
from wristsonar.protocol import GroundTruth, Protocol, Split

REASON = (
    "single fold of a leave-one-user-out sweep, aggregated across every fold "
    "before any number leaves this machine"
)


def _protocol(split: Split, subjects: int = 6) -> Protocol:
    return Protocol(
        split=split,
        dataset="synthetic",
        dataset_version="0",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=subjects,
    )


def _waivers() -> list[Waiver]:
    return [
        Waiver(
            guard=guard,
            reason=REASON,
            approved_by="adrian",
            expires_on=date.today() + timedelta(days=30),
        )
        for guard in GuardName
    ]


def _dataset_from_curls(
    curls: dict[str, float],
    *,
    n_frames: int = 20,
    n_sessions: int = 2,
    device_of: dict[str, str] | None = None,
) -> Dataset:
    """One participant per named curl offset, so mean poses land where told."""
    rng = np.random.default_rng(3)
    poses: list[np.ndarray] = []
    meta: list[SampleMeta] = []
    for person, offset in curls.items():
        for s in range(n_sessions):
            for f in range(n_frames):
                curl = np.full(5, offset) + 0.01 * rng.standard_normal(5)
                poses.append(make_pose(curl))
                meta.append(
                    SampleMeta(
                        participant=person,
                        session=f"{person}-S{s}",
                        device=("watch-a" if device_of is None else device_of[person]),
                        timestamp_s=float(f),
                    )
                )
    return Dataset.build(poses=np.stack(poses), meta=meta)


def test_participant_concentration_fires_on_a_single_subject_test_set() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]

    finding = check_participant_concentration(data, fold.test, split=Split.CROSS_USER)
    assert not finding.passed
    assert finding.blocking
    assert "single-subject result" in finding.message
    assert "2405.15085" in finding.message
    assert finding.detail["n_participants"] == 1


def test_participant_concentration_is_quiet_on_a_balanced_evaluation_set() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    everything = np.arange(data.n_samples, dtype=np.intp)

    finding = check_participant_concentration(
        data, everything, split=Split.CROSS_USER
    )
    assert finding.passed
    assert finding.detail["n_participants"] == 4


def test_participant_concentration_fires_on_an_unbalanced_mixture() -> None:
    data = make_dataset(n_participants=3, n_sessions=1, n_frames=20)
    test = np.concatenate(
        [
            data.indices_where(participant="P00"),
            data.indices_where(participant="P01")[:2],
        ]
    )
    finding = check_participant_concentration(data, test, split=Split.CROSS_USER)
    assert not finding.passed
    assert "of the evaluation set" in finding.message


def test_participant_concentration_is_quiet_on_a_within_user_protocol() -> None:
    # A cross-session split is one person by design. Firing here would mean
    # firing on every honest within-user split, which is how a guard gets
    # switched off permanently.
    data = make_dataset(n_participants=4, n_sessions=3, n_frames=20)
    fold = cross_session_split(data, participant="P00")

    finding = check_participant_concentration(data, fold.test, split=fold.split)
    assert finding.passed
    assert not finding.blocking
    assert "holds the person fixed by design" in finding.message


def test_label_distribution_fires_when_the_test_sits_on_a_training_subject() -> None:
    # P00 is the held-out subject and is an exact copy of training subject P01.
    data = _dataset_from_curls({"P00": 0.5, "P01": 0.5, "P02": 1.4, "P03": 2.3})
    test = data.indices_where(participant="P00")
    train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant != "P00"],
        dtype=np.intp,
    )

    finding = check_label_distribution_match(
        data, train, test, split=Split.CROSS_USER
    )
    assert not finding.passed
    assert finding.detail["nearest_train_participant"] == "P01"
    assert "would be enough to score well" in finding.message


def test_contaminating_the_split_cannot_make_the_guard_quieter() -> None:
    """The inversion this guard used to have, checked directly.

    Excluding participants present on both sides meant that moving one of the
    held-out subject's training frames into the test set removed that subject
    from the comparison, and the guard then compared against somebody else and
    passed. Worse contamination produced a cleaner report.
    """
    data = _dataset_from_curls({"P00": 0.5, "P01": 0.5, "P02": 1.4, "P03": 2.3})
    p00 = data.indices_where(participant="P00")
    clean_test = p00
    clean_train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant != "P00"],
        dtype=np.intp,
    )
    clean = check_label_distribution_match(
        data, clean_train, clean_test, split=Split.CROSS_USER
    )

    # One of P01's training frames moves across, so P01 is now on both sides.
    p01 = data.indices_where(participant="P01")
    dirty_test = np.sort(np.concatenate([clean_test, p01[:1]]))
    dirty_train = np.asarray(
        [i for i in clean_train if i != int(p01[0])], dtype=np.intp
    )
    dirty = check_label_distribution_match(
        data, dirty_train, dirty_test, split=Split.CROSS_USER
    )

    assert not clean.passed
    assert not dirty.passed
    assert "both the training and the test half" in dirty.message


def test_label_distribution_is_quiet_on_a_genuinely_new_subject() -> None:
    data = _dataset_from_curls(
        {"P00": 0.25, "P01": 0.0, "P02": 0.5, "P03": 1.0, "P04": 1.5}
    )
    test = data.indices_where(participant="P00")
    train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant != "P00"],
        dtype=np.intp,
    )

    finding = check_label_distribution_match(
        data, train, test, split=Split.CROSS_USER
    )
    assert finding.passed
    assert finding.detail["n_train_participants"] == 4


def test_label_distribution_does_not_fire_on_a_within_user_split() -> None:
    data = make_dataset(n_participants=4, n_sessions=3, n_frames=20)
    fold = cross_session_split(data, participant="P00")

    finding = check_label_distribution_match(
        data, fold.train, fold.test, split=fold.split
    )
    assert finding.passed
    assert finding.detail["applicable"] == "no"


def test_a_single_training_participant_fails_outright() -> None:
    data = _dataset_from_curls({"P00": 0.3, "P01": 0.9})
    test = data.indices_where(participant="P00")
    train = data.indices_where(participant="P01")

    finding = check_label_distribution_match(
        data, train, test, split=Split.CROSS_USER
    )
    assert not finding.passed
    assert "single-subject study" in finding.message


def test_evaluation_size_fires_on_too_few_independent_units() -> None:
    errors = np.full(4000, 12.0)
    clusters = ["P00/S0"] * 2000 + ["P00/S1"] * 2000

    finding = check_evaluation_size(errors, clusters, split=Split.CROSS_USER)
    assert not finding.passed
    assert "independent evaluation units" in finding.message
    assert finding.detail["n_clusters"] == 2
    assert finding.detail["n_samples"] == 4000


def test_a_single_cluster_fails_instead_of_raising() -> None:
    """The standard error of one observation is infinite, not an exception.

    Converting that infinity to an integer raised OverflowError from inside
    the guard, and a guard that raises is a guard that gets wrapped in a try.
    """
    errors = np.full(50, 9.0)
    clusters = ["P00/S0"] * 50

    finding = check_evaluation_size(
        errors, clusters, split=Split.CROSS_SESSION, min_clusters=1
    )
    assert not finding.passed
    assert finding.detail["n_clusters"] == 1
    assert "not a standard error of anything" in finding.message


def test_evaluation_size_fires_on_too_many_printed_digits() -> None:
    rng = np.random.default_rng(11)
    clusters: list[str] = []
    values: list[float] = []
    for c in range(12):
        level = 12.0 + 3.0 * rng.standard_normal()
        for _ in range(50):
            clusters.append(f"P{c:02d}/S0")
            values.append(level)

    finding = check_evaluation_size(
        np.asarray(values), clusters, split=Split.CROSS_USER, reported_decimals=2
    )
    assert not finding.passed
    assert "noise presented as resolution" in finding.message
    assert int(finding.detail["justified_decimals"]) < 2


def test_evaluation_size_is_quiet_when_the_spread_supports_the_digits() -> None:
    rng = np.random.default_rng(5)
    clusters: list[str] = []
    values: list[float] = []
    for c in range(20):
        level = 12.0 + 0.002 * rng.standard_normal()
        for _ in range(50):
            clusters.append(f"P{c:02d}/S0")
            values.append(level)

    finding = check_evaluation_size(
        np.asarray(values), clusters, split=Split.CROSS_USER, reported_decimals=2
    )
    assert finding.passed
    assert finding.detail["n_clusters"] == 20


def test_evaluation_size_accepts_per_joint_error_arrays() -> None:
    errors = np.full((100, 21), 9.0)
    clusters = [f"P{i % 10:02d}/S0" for i in range(100)]
    finding = check_evaluation_size(errors, clusters, split=Split.CROSS_USER)
    assert finding.passed


def test_run_guards_raises_by_default() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    errors = np.full(fold.n_test, 11.0)
    protocol = _protocol(Split.CROSS_USER, subjects=4)

    with pytest.raises(GuardViolation) as excinfo:
        run_guards(data, fold, errors, protocol)
    assert "shortcut-learning guards fired" in str(excinfo.value)

    report = run_guards(data, fold, errors, protocol, raise_on_violation=False)
    assert report.blocking
    assert not report.clean
    assert len(report.findings) == 3
    assert len(report.to_json()) == 3
    assert report.binding.split is Split.CROSS_USER
    assert report.binding.dataset_version == "0"


def test_run_guards_refuses_a_protocol_that_disagrees_with_the_split() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    with pytest.raises(ValueError, match="protocol declares"):
        run_guards(
            data,
            fold,
            np.full(fold.n_test, 11.0),
            _protocol(Split.WITHIN_SESSION, subjects=4),
        )


def test_a_guard_report_cannot_be_empty_or_partial() -> None:
    binding = GuardBinding(
        split=Split.CROSS_USER,
        split_name="fabricated",
        dataset="synthetic",
        dataset_version="0",
        n_train=1,
        n_test=1,
        fingerprint="0" * 64,
    )
    with pytest.raises(ValueError, match="finding for every guard"):
        GuardReport(findings=(), binding=binding)


def test_a_waiver_needs_a_real_reason_a_name_and_an_expiry() -> None:
    soon = date.today() + timedelta(days=30)
    with pytest.raises(ValueError, match="at least 40 characters"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason="too small",
            approved_by="someone",
            expires_on=soon,
        )
    with pytest.raises(ValueError, match="distinct words"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason="x" * 60,
            approved_by="someone",
            expires_on=soon,
        )
    with pytest.raises(ValueError, match="name the person"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason=REASON,
            approved_by="   ",
            expires_on=soon,
        )
    with pytest.raises(ValueError, match="not in the future"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason=REASON,
            approved_by="adrian",
            expires_on=date.today(),
        )
    with pytest.raises(ValueError, match="at most 90 days"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason=REASON,
            approved_by="adrian",
            expires_on=date.today() + timedelta(days=400),
        )


def test_a_waived_guard_stops_blocking_but_stays_visible() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    errors = np.full(fold.n_test, 11.0)

    report = run_guards(
        data, fold, errors, _protocol(Split.CROSS_USER, subjects=4), waivers=_waivers()
    )
    assert not report.blocking
    assert report.waived
    assert not report.clean
    rendered = report.render()
    assert "WAIVED" in rendered
    assert "adrian" in rendered


def test_an_expired_waiver_stops_suppressing_its_guard() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    waiver = Waiver(
        guard=GuardName.PARTICIPANT_CONCENTRATION,
        reason=REASON,
        approved_by="adrian",
        expires_on=date.today() + timedelta(days=1),
    )
    finding = check_participant_concentration(
        data, fold.test, split=Split.CROSS_USER, waiver=waiver
    )
    assert not finding.blocking

    # The same waiver, one day after it lapsed, suppresses nothing. A guard
    # switched off before a deadline has to be argued again after it.
    lapsed = object.__new__(Waiver)
    object.__setattr__(lapsed, "guard", waiver.guard)
    object.__setattr__(lapsed, "reason", waiver.reason)
    object.__setattr__(lapsed, "approved_by", waiver.approved_by)
    object.__setattr__(lapsed, "expires_on", date.today() - timedelta(days=1))
    relapsed = check_participant_concentration(
        data, fold.test, split=Split.CROSS_USER, waiver=lapsed
    )
    assert relapsed.blocking
    assert "EXPIRED" in lapsed.describe()


def test_every_guard_stays_quiet_on_an_honestly_built_split() -> None:
    # Five participants per device model, disjoint people on each side, pose
    # distributions spread across the population, ten independent units.
    people = [f"P{i:02d}" for i in range(10)]
    data = _dataset_from_curls(
        {person: 0.2 * i for i, person in enumerate(people)},
        n_frames=30,
        n_sessions=2,
        device_of={
            person: ("watch-a" if i < 5 else "watch-b")
            for i, person in enumerate(people)
        },
    )
    fold = cross_device_split(data, held_out_device="watch-b")
    errors = np.full(fold.n_test, 12.0)

    report = run_guards(
        data, fold, errors, _protocol(Split.CROSS_DEVICE, subjects=10)
    )
    assert report.clean
    assert all(f.passed for f in report.findings)
    assert len({f.guard for f in report.findings}) == len(GuardName)


def test_every_guard_stays_quiet_on_a_clean_cross_session_split() -> None:
    """The split the documentation calls the first honest number can headline.

    Every guard used to fire here at once: one participant by construction,
    no second training participant to compare against, and one held-out
    session so one cluster. All three are properties of the protocol rather
    than evidence of a shortcut, and a guard that fires on every honest run
    gets waived by reflex.
    """
    data = make_dataset(n_participants=4, n_sessions=3, n_frames=40)
    fold = cross_session_split(data, participant="P00")
    rng = np.random.default_rng(2)
    errors = 12.0 + 0.01 * rng.standard_normal(fold.n_test)

    report = run_guards(
        data, fold, errors, _protocol(Split.CROSS_SESSION, subjects=4)
    )
    assert report.clean
    assert not report.blocking
    size = next(f for f in report.findings if f.guard is GuardName.EVALUATION_SIZE)
    assert "not independent" in size.message


def test_a_split_indices_test_set_of_one_cluster_still_reports_a_binding() -> None:
    data = make_dataset(n_participants=3, n_sessions=2, n_frames=10)
    fold = SplitIndices(
        name="hand-made",
        split=Split.WITHIN_SESSION,
        train=data.indices_where(session=data.sessions[0])[:7],
        test=data.indices_where(session=data.sessions[0])[7:],
    )
    report = run_guards(
        data,
        fold,
        np.full(fold.n_test, 7.0),
        _protocol(Split.WITHIN_SESSION, subjects=3),
        raise_on_violation=False,
    )
    assert report.binding.n_test == fold.n_test
    assert len(report.binding.fingerprint) == 64
