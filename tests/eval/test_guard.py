"""Each guard on a constructed bad case, and quiet on a good one."""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset, make_pose
from wristsonar.eval.guard import (
    GuardName,
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
    cross_device_split,
    cross_user_folds,
)


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

    finding = check_participant_concentration(data, fold.test)
    assert not finding.passed
    assert finding.blocking
    assert "single-subject result" in finding.message
    assert "2405.15085" in finding.message
    assert finding.detail["n_participants"] == 1


def test_participant_concentration_is_quiet_on_a_balanced_evaluation_set() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    everything = np.arange(data.n_samples, dtype=np.intp)

    finding = check_participant_concentration(data, everything)
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
    finding = check_participant_concentration(data, test)
    assert not finding.passed
    assert "of the evaluation set" in finding.message


def test_label_distribution_fires_when_the_test_sits_on_a_training_subject() -> None:
    # P00 is the held-out subject and is an exact copy of training subject P01.
    data = _dataset_from_curls({"P00": 0.5, "P01": 0.5, "P02": 1.4, "P03": 2.3})
    test = data.indices_where(participant="P00")
    train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant != "P00"],
        dtype=np.intp,
    )

    finding = check_label_distribution_match(data, train, test)
    assert not finding.passed
    assert finding.detail["nearest_train_participant"] == "P01"
    assert "would be enough to score well" in finding.message


def test_label_distribution_is_quiet_on_a_genuinely_new_subject() -> None:
    data = _dataset_from_curls(
        {"P00": 0.25, "P01": 0.0, "P02": 0.5, "P03": 1.0, "P04": 1.5}
    )
    test = data.indices_where(participant="P00")
    train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant != "P00"],
        dtype=np.intp,
    )

    finding = check_label_distribution_match(data, train, test)
    assert finding.passed
    assert finding.detail["n_train_participants"] == 4


def test_a_single_training_participant_fails_outright() -> None:
    data = _dataset_from_curls({"P00": 0.3, "P01": 0.9})
    test = data.indices_where(participant="P00")
    train = data.indices_where(participant="P01")

    finding = check_label_distribution_match(data, train, test)
    assert not finding.passed
    assert "single-subject study" in finding.message


def test_evaluation_size_fires_on_too_few_independent_units() -> None:
    errors = np.full(4000, 12.0)
    clusters = ["P00/S0"] * 2000 + ["P00/S1"] * 2000

    finding = check_evaluation_size(errors, clusters)
    assert not finding.passed
    assert "independent evaluation units" in finding.message
    assert finding.detail["n_clusters"] == 2
    assert finding.detail["n_samples"] == 4000


def test_evaluation_size_fires_on_too_many_printed_digits() -> None:
    rng = np.random.default_rng(11)
    clusters: list[str] = []
    values: list[float] = []
    for c in range(12):
        level = 12.0 + 3.0 * rng.standard_normal()
        for _ in range(50):
            clusters.append(f"P{c:02d}/S0")
            values.append(level)

    finding = check_evaluation_size(np.asarray(values), clusters, reported_decimals=2)
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

    finding = check_evaluation_size(np.asarray(values), clusters, reported_decimals=2)
    assert finding.passed
    assert finding.detail["n_clusters"] == 20


def test_evaluation_size_accepts_per_joint_error_arrays() -> None:
    errors = np.full((100, 21), 9.0)
    clusters = [f"P{i % 10:02d}/S0" for i in range(100)]
    finding = check_evaluation_size(errors, clusters)
    assert finding.passed


def test_run_guards_raises_by_default() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    errors = np.full(fold.n_test, 11.0)

    with pytest.raises(GuardViolation) as excinfo:
        run_guards(data, fold, errors)
    assert "shortcut-learning guards fired" in str(excinfo.value)

    report = run_guards(data, fold, errors, raise_on_violation=False)
    assert report.blocking
    assert not report.clean
    assert len(report.findings) == 3
    assert len(report.to_json()) == 3


def test_a_waiver_needs_a_real_reason_and_a_name() -> None:
    with pytest.raises(ValueError, match="at least 40 characters"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason="too small",
            approved_by="someone",
        )
    with pytest.raises(ValueError, match="name the person"):
        Waiver(
            guard=GuardName.EVALUATION_SIZE,
            reason="x" * 60,
            approved_by="   ",
        )


def test_a_waived_guard_stops_blocking_but_stays_visible() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    errors = np.full(fold.n_test, 11.0)
    waivers = [
        Waiver(
            guard=guard,
            reason=(
                "single fold of a leave-one-user-out sweep, aggregated across "
                "all four folds before anything is quoted"
            ),
            approved_by="adrian",
        )
        for guard in GuardName
    ]

    report = run_guards(data, fold, errors, waivers=waivers)
    assert not report.blocking
    assert report.waived
    assert not report.clean
    rendered = report.render()
    assert "WAIVED" in rendered
    assert "adrian" in rendered


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

    report = run_guards(data, fold, errors)
    assert report.clean
    assert all(f.passed for f in report.findings)
    assert len({f.guard for f in report.findings}) == len(GuardName)
