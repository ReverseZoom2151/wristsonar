"""How well the trivial predictors do, which is the uncomfortable part."""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset
from wristsonar.eval.baselines import (
    MeanPoseBaseline,
    NearestNeighbourBaseline,
    PerSubjectMeanPoseBaseline,
    default_baselines,
    evaluate_baselines,
)
from wristsonar.eval.metrics import Alignment, mpjpe
from wristsonar.eval.splits import (
    Dataset,
    cross_session_split,
    cross_user_folds,
)
from wristsonar.protocol import GroundTruth, Protocol, Split


def _protocol(split: Split, subjects: int) -> Protocol:
    return Protocol(
        split=split,
        dataset="synthetic",
        dataset_version="0",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=subjects,
    )


def test_mean_pose_scores_plausibly_and_embarrassingly_well() -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=30)
    fold = cross_user_folds(data)[0]
    baseline = MeanPoseBaseline()
    pred = baseline.fit_predict(data, fold)

    error = mpjpe(pred, data.poses[fold.test], alignment=Alignment.ROOT)
    # A predictor that never reads the microphone lands in the same order of
    # magnitude as published acoustic systems, which is the entire argument.
    assert 1.0 < error < 60.0
    assert pred.shape == (fold.n_test, 21, 3)
    assert np.allclose(pred[0], pred[-1])


def test_per_subject_mean_beats_global_mean_when_the_subject_is_seen() -> None:
    data = make_dataset(n_participants=5, n_sessions=3, n_frames=30)
    fold = cross_session_split(data, participant="P00")
    truth = data.poses[fold.test]

    global_error = mpjpe(
        MeanPoseBaseline().fit_predict(data, fold), truth, alignment=Alignment.ROOT
    )
    per_subject = PerSubjectMeanPoseBaseline()
    subject_error = mpjpe(
        per_subject.fit_predict(data, fold), truth, alignment=Alignment.ROOT
    )

    assert per_subject.unseen_fraction == 0.0
    assert subject_error <= global_error


def test_per_subject_mean_collapses_to_the_global_mean_across_users() -> None:
    data = make_dataset(n_participants=5, n_sessions=2, n_frames=20)
    fold = cross_user_folds(data)[0]
    per_subject = PerSubjectMeanPoseBaseline()
    subject_pred = per_subject.fit_predict(data, fold)
    global_pred = MeanPoseBaseline().fit_predict(data, fold)

    # Every fallback fired, which is what a clean cross-user split guarantees.
    assert per_subject.unseen_fraction == 1.0
    assert np.allclose(subject_pred, global_pred)


def test_nearest_neighbour_refuses_to_run_without_features() -> None:
    data = make_dataset(n_participants=3, n_sessions=2, n_frames=10)
    stripped = Dataset(poses=data.poses, meta=data.meta, features=None)
    fold = cross_user_folds(stripped)[0]

    with pytest.raises(ValueError, match="no input features"):
        NearestNeighbourBaseline().fit_predict(stripped, fold)

    # And the sweep drops the row rather than inventing a score for it.
    results = evaluate_baselines(stripped, fold, _protocol(Split.CROSS_USER, 3))
    assert "nearest-neighbour" not in results
    assert "mean-pose" in results
    assert "per-subject-mean-pose" in results


def test_nearest_neighbour_reads_the_features() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=25)
    fold = cross_session_split(data, participant="P00")
    truth = data.poses[fold.test]

    nn_error = mpjpe(
        NearestNeighbourBaseline().fit_predict(data, fold),
        truth,
        alignment=Alignment.ROOT,
    )
    assert nn_error > 0.0
    assert np.isfinite(nn_error)


def test_evaluate_baselines_returns_protocol_stamped_metrics() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=15)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, 4)
    results = evaluate_baselines(data, fold, protocol)

    assert set(results) == {"mean-pose", "per-subject-mean-pose", "nearest-neighbour"}
    for metrics in results.values():
        assert metrics.mpjpe.protocol.split is Split.CROSS_USER
        assert metrics.mpjpe.samples == fold.n_test
        assert metrics.breakdown.per_joint_mm["wrist"] == pytest.approx(0.0, abs=1e-9)


def test_default_baselines_are_fresh_instances_each_call() -> None:
    first = default_baselines()
    second = default_baselines()
    assert [b.name for b in first] == [b.name for b in second]
    assert all(a is not b for a, b in zip(first, second, strict=True))
