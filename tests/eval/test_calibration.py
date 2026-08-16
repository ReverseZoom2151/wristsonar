"""The calibration curve, and the shape it has to have."""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset
from wristsonar.eval.baselines import PerSubjectMeanPoseBaseline
from wristsonar.eval.calibration import (
    CALIBRATION_BUDGETS_MIN,
    sweep_calibration,
)
from wristsonar.eval.metrics import PoseArray
from wristsonar.eval.splits import (
    Dataset,
    IndexArray,
    LeakageError,
    SplitIndices,
    cross_user_folds,
)
from wristsonar.protocol import GroundTruth, Protocol, Split

SAMPLE_SECONDS = 1.0


def _fitter(dataset: Dataset, train: IndexArray, test: IndexArray) -> PoseArray:
    """A personalising predictor: per-subject mean, global mean when unseen.

    Chosen because its behaviour under calibration is known in advance, so the
    curve's shape is a property of the sweep rather than of a model nobody can
    reason about.
    """
    baseline = PerSubjectMeanPoseBaseline()
    baseline.fit(dataset, train)
    return baseline.predict(dataset, test)


def _protocol() -> Protocol:
    return Protocol(
        split=Split.CROSS_USER,
        dataset="synthetic",
        dataset_version="0",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=4,
    )


def _setup() -> tuple[Dataset, SplitIndices, IndexArray]:
    data = make_dataset(n_participants=4, n_sessions=3, n_frames=120)
    held_out = "P00"
    test = data.indices_where(participant=held_out, session="p00/p00-s2")
    pool = np.sort(
        np.concatenate(
            [
                data.indices_where(participant=held_out, session="p00/p00-s0"),
                data.indices_where(participant=held_out, session="p00/p00-s1"),
            ]
        )
    )
    train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant != held_out],
        dtype=np.intp,
    )
    indices = SplitIndices(
        name=f"cross-user[held-out={held_out}]",
        split=Split.CROSS_USER,
        train=train,
        test=test,
    )
    return data, indices, pool


def test_curve_has_one_point_per_budget_and_stamps_each_protocol() -> None:
    data, indices, pool = _setup()
    curve = sweep_calibration(
        data,
        indices,
        _fitter,
        _protocol(),
        sample_seconds=SAMPLE_SECONDS,
        calibration_pool=pool,
    )

    assert len(curve.points) == len(CALIBRATION_BUDGETS_MIN)
    assert [p.minutes_requested for p in curve.points] == list(CALIBRATION_BUDGETS_MIN)

    zero = curve.zero_shot
    assert zero is not None
    assert zero.n_calibration_samples == 0
    assert zero.metrics.mpjpe.protocol.is_zero_shot

    two_minutes = curve.field_norm
    assert two_minutes is not None
    assert two_minutes.n_calibration_samples == 120
    assert two_minutes.metrics.mpjpe.protocol.calibration_minutes == pytest.approx(2.0)
    assert not two_minutes.metrics.mpjpe.protocol.is_zero_shot


def test_curve_falls_with_calibration_and_reports_the_gain() -> None:
    data, indices, pool = _setup()
    curve = sweep_calibration(
        data,
        indices,
        _fitter,
        _protocol(),
        sample_seconds=SAMPLE_SECONDS,
        calibration_pool=pool,
    )
    values = [p.metrics.mpjpe.value for p in curve.points]

    # Zero-shot is the worst point and the first minute buys the most.
    assert values[0] > values[1]
    assert curve.is_monotone
    assert curve.best is curve.points[-1]

    gain = curve.personalisation_gain_mm
    assert gain is not None
    assert gain > 0.0
    assert "personalisation buys" in curve.render()
    assert curve.to_json()[0]["minutes_requested"] == 0.0


def test_a_budget_larger_than_the_pool_is_marked_truncated() -> None:
    data, indices, pool = _setup()
    curve = sweep_calibration(
        data,
        indices,
        _fitter,
        _protocol(),
        sample_seconds=SAMPLE_SECONDS,
        calibration_pool=pool,
    )
    twenty = curve.points[-1]

    assert twenty.minutes_requested == 20.0
    assert twenty.truncated
    assert twenty.minutes_achieved == pytest.approx(pool.size / 60.0)
    assert "TRUNCATED" in twenty.describe()


def test_calibrating_on_the_test_frames_raises() -> None:
    data, indices, _pool = _setup()
    with pytest.raises(LeakageError, match="shares"):
        sweep_calibration(
            data,
            indices,
            _fitter,
            _protocol(),
            sample_seconds=SAMPLE_SECONDS,
            calibration_pool=indices.test,
        )


def _two_person_setup() -> tuple[Dataset, SplitIndices, IndexArray]:
    """Two people in the test set, both with held-in data of their own."""
    data = make_dataset(n_participants=4, n_sessions=3, n_frames=60)
    held_out = ("P00", "P01")

    def session(person: str, number: int) -> str:
        key = person.lower()
        return f"{key}/{key}-s{number}"

    test = np.sort(
        np.concatenate(
            [
                data.indices_where(participant=person, session=session(person, 2))
                for person in held_out
            ]
        )
    )
    pool = np.sort(
        np.concatenate(
            [
                data.indices_where(participant=person, session=session(person, s))
                for person in held_out
                for s in (0, 1)
            ]
        )
    )
    train = np.asarray(
        [i for i in range(data.n_samples) if data.meta[i].participant not in held_out],
        dtype=np.intp,
    )
    indices = SplitIndices(
        name="cross-user[held-out=P00,P01]",
        split=Split.CROSS_USER,
        train=train,
        test=test,
    )
    return data, indices, pool


def test_the_budget_is_spent_per_user_and_not_in_recording_order() -> None:
    """Two minutes of per-user data means two minutes each.

    Spending the budget in global recording order gives the whole allowance to
    whoever recorded earliest, leaves the other test participant with nothing,
    and still reports the result as per-user calibration.
    """
    data, indices, pool = _two_person_setup()
    curve = sweep_calibration(
        data,
        indices,
        _fitter,
        _protocol(),
        sample_seconds=SAMPLE_SECONDS,
        budgets=(1.0,),
        calibration_pool=pool,
    )
    point = curve.points[0]

    assert point.n_calibration_samples == 120  # sixty seconds for each of two
    assert point.minutes_achieved == pytest.approx(1.0)
    assert not point.truncated


def test_a_calibration_pool_from_another_person_is_refused() -> None:
    data, indices, _pool = _two_person_setup()
    outsider = data.indices_where(participant="P02")
    with pytest.raises(LeakageError, match="not in the test set"):
        sweep_calibration(
            data,
            indices,
            _fitter,
            _protocol(),
            sample_seconds=SAMPLE_SECONDS,
            budgets=(1.0,),
            calibration_pool=outsider,
        )


def test_leave_one_user_out_has_no_pool_and_says_so() -> None:
    data = make_dataset(n_participants=3, n_sessions=2, n_frames=20)
    fold = cross_user_folds(data)[0]

    # Zero-shot alone is fine, because it needs no held-in data.
    curve = sweep_calibration(
        data, fold, _fitter, _protocol(), sample_seconds=SAMPLE_SECONDS, budgets=(0.0,)
    )
    assert len(curve.points) == 1

    with pytest.raises(ValueError, match="held-in pool"):
        sweep_calibration(
            data,
            fold,
            _fitter,
            _protocol(),
            sample_seconds=SAMPLE_SECONDS,
            budgets=(0.0, 2.0),
        )
