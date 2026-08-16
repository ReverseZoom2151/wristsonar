from __future__ import annotations

import inspect
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.eval.synthetic import make_dataset
from wristsonar.eval.experiment import (
    ExperimentError,
    Predictor,
    SplitSuiteResult,
    run_cross_user_experiment,
    run_experiment,
    run_split_suite,
)
from wristsonar.eval.guard import GuardName, Waiver
from wristsonar.eval.metrics import PoseArray
from wristsonar.eval.report import MultiSplitReport
from wristsonar.eval.splits import Dataset, cross_user_folds
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
            expires_on=date.today() + timedelta(days=30),
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


def _suite_protocol(subjects: int) -> Protocol:
    """The split field is replaced per split by the suite, so any value serves."""
    return Protocol(
        split=Split.WITHIN_SESSION,
        dataset="synthetic",
        dataset_version="1",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=subjects,
    )


def _near_perfect(
    dataset: Dataset, train: NDArray[np.intp], test: NDArray[np.intp]
) -> PoseArray:
    predicted: PoseArray = dataset.poses[test] + 0.0001
    return predicted


def _kind_to_one_participant(favoured: str) -> Predictor:
    """A predictor that is excellent on one person and poor on everyone else.

    The shape that makes fold selection tempting: one leave-one-user-out fold
    looks like a result and the rest do not.
    """

    def predict(
        dataset: Dataset, train: NDArray[np.intp], test: NDArray[np.intp]
    ) -> PoseArray:
        out: PoseArray = dataset.poses[test].copy()
        for row, index in enumerate(test):
            good = dataset.meta[int(index)].participant == favoured
            out[row] += 0.0001 if good else 0.02
        return out

    return predict


def test_the_split_suite_headlines_only_with_every_split_present(
    tmp_path: Path,
) -> None:
    data = make_dataset(
        n_participants=4, n_sessions=2, n_frames=8, devices=("watch-a", "watch-b")
    )

    suite = run_split_suite(
        data,
        _suite_protocol(subjects=4),
        name="wristsonar",
        predictor=_near_perfect,
        waivers=_waivers(),
    )
    suite.write(tmp_path)

    assert suite.absences == ()
    assert set(suite.families) == set(Split)
    assert suite.report.missing_splits == ()
    assert suite.report.can_headline, suite.report.refusal_reasons
    # The headline is the strongest split's, never the most flattering one's.
    assert "split=cross-device" in suite.report.headline()
    cross_user_dir = tmp_path / "splits" / "cross_user"
    assert (cross_user_dir / "folds" / "00" / "report.json").is_file()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["can_headline"] is True
    assert payload["absences"] == []
    assert len(payload["folds"]["CROSS_USER"]) == 4


def test_a_dataset_that_cannot_support_a_split_says_which_and_why() -> None:
    one_device = make_dataset(
        n_participants=4, n_sessions=2, n_frames=8, devices=("watch-a",)
    )

    suite = run_split_suite(
        one_device,
        _suite_protocol(subjects=4),
        name="wristsonar",
        predictor=_near_perfect,
        waivers=_waivers(),
    )

    assert Split.CROSS_DEVICE not in suite.families
    absence = suite.absence_for(Split.CROSS_DEVICE)
    assert absence is not None
    assert "watch-a" in absence.reason
    # Stated, not silent, and still not quotable.
    assert not suite.report.can_headline
    assert "CROSS_DEVICE" in " ".join(suite.report.refusal_reasons)
    assert absence.describe() in suite.report.render_text()

    one_session = make_dataset(
        n_participants=4, n_sessions=1, n_frames=8, devices=("watch-a",)
    )
    thin = run_split_suite(
        one_session,
        _suite_protocol(subjects=4),
        name="wristsonar",
        predictor=_near_perfect,
        waivers=_waivers(),
    )
    absent = {a.split for a in thin.absences}
    assert absent == {Split.CROSS_SESSION, Split.CROSS_DEVICE}
    assert set(thin.families) == {Split.WITHIN_SESSION, Split.CROSS_USER}
    reasons = [a.reason for a in thin.absences]
    assert any("remounted" in reason for reason in reasons)
    assert thin.to_dict()["can_headline"] is False


def test_the_split_suite_evaluates_one_model_on_every_split() -> None:
    data = make_dataset(
        n_participants=4, n_sessions=2, n_frames=8, devices=("watch-a", "watch-b")
    )
    suite = run_split_suite(
        data,
        _suite_protocol(subjects=4),
        name="wristsonar",
        predictor=_near_perfect,
        waivers=_waivers(),
    )

    assert suite.report.model_names() == ("wristsonar",)
    for family in suite.families.values():
        assert [row.name for row in family.report.models] == ["wristsonar"]

    # There is no per-split model name to get wrong, so the only way to build a
    # mismatched set is to assemble one by hand, and that is refused too.
    other = run_split_suite(
        data,
        _suite_protocol(subjects=4),
        name="wristsonar-lite",
        predictor=_near_perfect,
        waivers=_waivers(),
    )
    mixed = MultiSplitReport(title="two systems in one set")
    mixed.add(suite.families[Split.CROSS_SESSION].report)
    mixed.add(other.families[Split.CROSS_USER].report)
    with pytest.raises(ExperimentError, match="one model on every split"):
        SplitSuiteResult(report=mixed, families={}, absences=())


def test_the_split_suite_leaves_no_room_to_pick_a_favourable_fold() -> None:
    data = make_dataset(
        n_participants=4, n_sessions=2, n_frames=8, devices=("watch-a", "watch-b")
    )

    suite = run_split_suite(
        data,
        _suite_protocol(subjects=4),
        name="wristsonar",
        predictor=_kind_to_one_participant("P00"),
        waivers=_waivers(),
    )

    family = suite.families[Split.CROSS_USER]
    per_fold = sorted(fold.report.models[0].mpjpe.value for fold in family.folds)
    aggregate = family.report.models[0].mpjpe.value

    # Every participant contributes the same number of frames here, so pooling
    # is exactly the mean over folds. The flattering fold is nowhere near it.
    assert aggregate == pytest.approx(sum(per_fold) / len(per_fold))
    assert aggregate > per_fold[0] * 10
    assert family.report.notes.count("mm") == 2
    assert "best fold is not a result" in family.report.notes

    # Nothing in the entry point names a fold, so there is no argument through
    # which a fold could be chosen after the numbers are visible.
    parameters = inspect.signature(run_split_suite).parameters
    assert not any("fold" in parameter for parameter in parameters)
