from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset
from wristsonar.eval.experiment import (
    ExperimentError,
    run_cross_user_experiment,
    run_experiment,
)
from wristsonar.eval.guard import GuardName, Waiver
from wristsonar.eval.splits import cross_user_folds
from wristsonar.protocol import GroundTruth, Protocol, Split


def _protocol() -> Protocol:
    return Protocol(
        split=Split.CROSS_USER,
        dataset="synthetic",
        dataset_version="1",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=6,
    )


def _waivers() -> list[Waiver]:
    return [
        Waiver(
            guard=guard,
            reason=(
                "Synthetic unit data exercises report assembly, not a hardware claim."
            ),
            approved_by="test-suite",
        )
        for guard in GuardName
    ]


def test_experiment_runs_model_baselines_and_guards_on_the_same_split(
    tmp_path: Path,
) -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=12)
    indices = cross_user_folds(data)[0]

    result = run_experiment(
        data,
        indices,
        _protocol(),
        name="near-perfect",
        predictor=lambda dataset, train, test: dataset.poses[test] + 0.0001,
        waivers=_waivers(),
    )
    result.write(tmp_path)

    assert result.report.models[0].name == "near-perfect"
    assert len(result.report.baselines) == 3
    assert result.report.guards is not None
    assert (tmp_path / "predictions.npy").is_file()
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["test_indices"] == indices.test.tolist()
    assert saved["prediction_sha256"] == result.prediction_sha256


def test_experiment_refuses_to_label_a_cross_user_split_as_within_session() -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=10)
    indices = cross_user_folds(data)[0]
    wrong = Protocol(
        split=Split.WITHIN_SESSION,
        dataset="synthetic",
        dataset_version="1",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=6,
    )

    with pytest.raises(ExperimentError, match="protocol declares"):
        run_experiment(
            data,
            indices,
            wrong,
            name="wrong",
            predictor=lambda dataset, train, test: dataset.poses[test],
        )


def test_experiment_rejects_wrong_prediction_shape() -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=10)
    indices = cross_user_folds(data)[0]

    with pytest.raises(ExperimentError, match="predictor returned"):
        run_experiment(
            data,
            indices,
            _protocol(),
            name="wrong-shape",
            predictor=lambda dataset, train, test: np.zeros((1, 21, 3)),
        )


def test_cross_user_execution_aggregates_every_held_out_person(tmp_path: Path) -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=12)

    result = run_cross_user_experiment(
        data,
        _protocol(),
        name="near-perfect",
        predictor=lambda dataset, train, test: dataset.poses[test] + 0.0001,
        waivers=_waivers(),
    )
    result.write(tmp_path)

    assert len(result.folds) == 6
    assert result.test_indices.tolist() == list(range(data.n_samples))
    assert result.predictions.shape == data.poses.shape
    assert result.report.guards is not None
    assert (tmp_path / "folds" / "00" / "report.json").is_file()


def test_cross_user_execution_requires_a_cross_user_protocol() -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=10)
    wrong = Protocol(
        split=Split.CROSS_SESSION,
        dataset="synthetic",
        dataset_version="1",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=6,
    )

    with pytest.raises(ExperimentError, match="CROSS_USER"):
        run_cross_user_experiment(
            data,
            wrong,
            name="wrong",
            predictor=lambda dataset, train, test: dataset.poses[test],
        )
