"""One execution path for a protocol-bound acoustic pose experiment.

Metrics, baselines and guards are useful only when they consume the exact same
partition and protocol as the learned model.  This module owns that join.  It
does not expose a convenience mode for skipping baselines or guards: a caller
can collect a blocking report, but cannot turn it into a headline by omitting
the uncomfortable rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wristsonar.eval.baselines import Baseline, default_baselines, evaluate_baselines
from wristsonar.eval.guard import (
    GuardBinding,
    GuardFinding,
    GuardName,
    GuardReport,
    Waiver,
    check_evaluation_size,
    check_label_distribution_match,
    check_participant_concentration,
    run_guards,
    split_fingerprint,
)
from wristsonar.eval.metrics import Alignment, PoseArray, evaluate, joint_errors
from wristsonar.eval.report import Report
from wristsonar.eval.splits import (
    Dataset,
    SplitIndices,
    assert_no_leakage,
    cross_user_folds,
)
from wristsonar.protocol import Protocol, Split

__all__ = [
    "CrossUserExperimentResult",
    "ExperimentError",
    "ExperimentResult",
    "run_cross_user_experiment",
    "run_experiment",
]

Predictor = Callable[[Dataset, NDArray[np.intp], NDArray[np.intp]], PoseArray]


class ExperimentError(ValueError):
    """The attempted experiment contradicts its own declared protocol."""


def _prediction_digest(predictions: NDArray[np.float64]) -> str:
    """Stable content identity for the exact predictions behind a report."""
    return hashlib.sha256(np.ascontiguousarray(predictions).tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """A report plus the precise prediction array and split that generated it."""

    report: Report
    predictions: PoseArray
    indices: SplitIndices
    prediction_sha256: str

    def __post_init__(self) -> None:
        if self.predictions.shape[0] != self.indices.n_test:
            raise ExperimentError(
                "prediction count must equal the split test count, got "
                f"{self.predictions.shape[0]} and {self.indices.n_test}"
            )
        if self.prediction_sha256 != _prediction_digest(self.predictions):
            raise ExperimentError("prediction digest does not match the array")
        # The report carries guards and the guards carry the split they were
        # run on. Checking the two against each other here is what stops one
        # fold's guards from being attached to another fold's numbers, which
        # the rendered report has no way of showing.
        guards = self.report.guards
        if guards is None:
            raise ExperimentError(
                "an experiment result must carry its guard report; a result "
                "with no guards is a number with nothing checking it"
            )
        if (
            guards.binding.split_name != self.indices.name
            or guards.binding.n_test != self.indices.n_test
        ):
            raise ExperimentError(
                f"the guard report describes {guards.binding.split_name!r} with "
                f"{guards.binding.n_test} held-out samples, but this result is "
                f"{self.indices.name!r} with {self.indices.n_test}"
            )

    def to_dict(self) -> dict[str, object]:
        """Machine-readable result that names the exact held-out examples."""
        return {
            "report": self.report.to_dict(),
            "split_name": self.indices.name,
            "train_indices": self.indices.train.tolist(),
            "test_indices": self.indices.test.tolist(),
            "prediction_sha256": self.prediction_sha256,
        }

    def write(self, directory: Path) -> None:
        """Write report, split membership and predictions as one inseparable run."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.txt").write_text(
            self.report.render_text() + "\n", encoding="utf-8"
        )
        (directory / "report.json").write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        np.save(directory / "predictions.npy", self.predictions)


@dataclass(frozen=True, slots=True)
class CrossUserExperimentResult:
    """All leave-one-user-out folds plus their only quotable aggregate."""

    report: Report
    folds: tuple[ExperimentResult, ...]
    test_indices: NDArray[np.intp]
    predictions: PoseArray
    prediction_sha256: str

    def __post_init__(self) -> None:
        if not self.folds:
            raise ExperimentError("cross-user aggregation needs at least one fold")
        if self.test_indices.shape != (self.predictions.shape[0],):
            raise ExperimentError("aggregate test indices and predictions disagree")
        if len(set(self.test_indices.tolist())) != self.test_indices.size:
            raise ExperimentError("cross-user folds overlap in their held-out examples")
        if self.prediction_sha256 != _prediction_digest(self.predictions):
            raise ExperimentError(
                "aggregate prediction digest does not match the array"
            )

    def write(self, directory: Path) -> None:
        """Persist the aggregate and each fold, never just the flattering mean."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.txt").write_text(
            self.report.render_text() + "\n", encoding="utf-8"
        )
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "report": self.report.to_dict(),
                    "test_indices": self.test_indices.tolist(),
                    "prediction_sha256": self.prediction_sha256,
                    "folds": [fold.indices.name for fold in self.folds],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        np.save(directory / "predictions.npy", self.predictions)
        for number, fold in enumerate(self.folds):
            fold.write(directory / "folds" / f"{number:02d}")


def run_experiment(
    dataset: Dataset,
    indices: SplitIndices,
    protocol: Protocol,
    *,
    name: str,
    predictor: Predictor,
    title: str | None = None,
    baselines: Sequence[Baseline] | None = None,
    waivers: Sequence[Waiver] = (),
    alignment: Alignment = Alignment.ROOT,
) -> ExperimentResult:
    """Run model, trivial baselines and shortcut guards on one exact split.

    Guard failures are attached to the returned report rather than raised here.
    That gives a researcher evidence about what failed, while ``Report`` still
    refuses to call the outcome headline-worthy.  A caller who wants a hard
    stop can inspect ``result.report.guards.blocking`` after persisting it.
    """
    if not name.strip():
        raise ExperimentError("model name must not be empty")
    if protocol.split is not indices.split:
        raise ExperimentError(
            f"protocol declares {protocol.split.name}, but split {indices.name!r} "
            f"is {indices.split.name}"
        )
    assert_no_leakage(dataset, indices)
    predicted = np.asarray(
        predictor(dataset, indices.train, indices.test), dtype=np.float64
    )
    truth = dataset.poses[indices.test]
    if predicted.shape != truth.shape:
        raise ExperimentError(
            f"predictor returned {predicted.shape}, expected {truth.shape}"
        )
    if not np.isfinite(predicted).all():
        raise ExperimentError("predictor returned non-finite poses")

    report = Report(title=title or f"{name} [{indices.name}]")
    report.add_baselines(
        evaluate_baselines(
            dataset,
            indices,
            protocol,
            baselines=list(baselines) if baselines else None,
            alignment=alignment,
        )
    )
    report.add_model(name, evaluate(predicted, truth, protocol, alignment=alignment))
    per_sample_error = joint_errors(predicted, truth, alignment=alignment).mean(axis=1)
    guards: GuardReport = run_guards(
        dataset,
        indices,
        per_sample_error,
        protocol,
        waivers=waivers,
        raise_on_violation=False,
    )
    report.set_guards(guards)
    pose_predictions: PoseArray = predicted
    return ExperimentResult(
        report=report,
        predictions=pose_predictions,
        indices=indices,
        prediction_sha256=_prediction_digest(pose_predictions),
    )


def run_cross_user_experiment(
    dataset: Dataset,
    protocol: Protocol,
    *,
    name: str,
    predictor: Predictor,
    title: str | None = None,
    waivers: Sequence[Waiver] = (),
    alignment: Alignment = Alignment.ROOT,
) -> CrossUserExperimentResult:
    """Evaluate leave-one-user-out as one aggregate, with every fold retained.

    ``run_experiment`` intentionally treats a single held-out participant as a
    guarded, non-headline result.  This function is the route for the real
    cross-user claim: each person is held out once, predictions are joined back
    in original example order, and baselines are refit separately inside every
    fold.  A global mean baseline fitted once would train on the person it is
    evaluating in all but one fold, which is a quiet but serious leak.
    """
    if protocol.split is not Split.CROSS_USER:
        raise ExperimentError(
            "run_cross_user_experiment requires a CROSS_USER protocol, got "
            f"{protocol.split.name}"
        )
    if not name.strip():
        raise ExperimentError("model name must not be empty")

    folds = cross_user_folds(dataset)
    by_guard = {waiver.guard: waiver for waiver in waivers}
    fold_results: list[ExperimentResult] = []
    all_indices: list[NDArray[np.intp]] = []
    all_predictions: list[PoseArray] = []
    baseline_predictions: dict[str, list[PoseArray]] = {}
    label_findings: list[GuardFinding] = []

    for fold in folds:
        prediction = np.asarray(
            predictor(dataset, fold.train, fold.test), dtype=np.float64
        )
        truth = dataset.poses[fold.test]
        if prediction.shape != truth.shape:
            raise ExperimentError(
                f"predictor returned {prediction.shape} for {fold.name!r}, "
                f"expected {truth.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ExperimentError(
                f"predictor returned non-finite poses for {fold.name!r}"
            )

        # Each baseline is a fresh object per fold. Its fitted state belongs to
        # that fold's train half and must never cross the held-out user boundary.
        for baseline in default_baselines(
            include_nearest_neighbour=dataset.features is not None
        ):
            baseline_prediction = baseline.fit_predict(dataset, fold)
            baseline_predictions.setdefault(baseline.name, []).append(
                baseline_prediction
            )

        fold_error = joint_errors(prediction, truth, alignment=alignment).mean(
            axis=1
        )
        fold_guards = run_guards(
            dataset,
            fold,
            fold_error,
            protocol,
            waivers=waivers,
            raise_on_violation=False,
        )
        # A one-user held-out fold is expected to fail concentration. Preserve
        # all other diagnostics locally; the aggregate will evaluate
        # concentration and independent-session precision over every person.
        fold_report = Report(title=f"{name} [{fold.name}]")
        for baseline_name, baseline_chunks in baseline_predictions.items():
            if len(baseline_chunks) == len(fold_results) + 1:
                fold_report.add_baseline(
                    baseline_name,
                    evaluate(
                        baseline_chunks[-1], truth, protocol, alignment=alignment
                    ),
                )
        fold_report.add_model(
            name, evaluate(prediction, truth, protocol, alignment=alignment)
        )
        fold_report.set_guards(fold_guards)
        fold_results.append(
            ExperimentResult(
                report=fold_report,
                predictions=prediction,
                indices=fold,
                prediction_sha256=_prediction_digest(prediction),
            )
        )
        all_indices.append(fold.test)
        all_predictions.append(prediction)
        label_findings.append(
            check_label_distribution_match(
                dataset,
                fold.train,
                fold.test,
                split=fold.split,
                waiver=by_guard.get(GuardName.SINGLE_SUBJECT_LABEL_MATCH),
            )
        )

    joined_indices = np.concatenate(all_indices)
    joined_predictions = np.concatenate(all_predictions)
    order = np.argsort(joined_indices, kind="stable")
    test_indices = np.asarray(joined_indices[order], dtype=np.intp)
    predictions = np.asarray(joined_predictions[order], dtype=np.float64)
    truth = dataset.poses[test_indices]

    report = Report(title=title or f"{name} [cross-user leave-one-out aggregate]")
    for baseline_name, chunks in baseline_predictions.items():
        joined = np.concatenate(chunks)[order]
        report.add_baseline(
            baseline_name, evaluate(joined, truth, protocol, alignment=alignment)
        )
    report.add_model(
        name, evaluate(predictions, truth, protocol, alignment=alignment)
    )
    aggregate_errors = joint_errors(predictions, truth, alignment=alignment).mean(
        axis=1
    )
    clusters = [
        f"{dataset.meta[int(index)].participant_key}/"
        f"{dataset.meta[int(index)].session_key}"
        for index in test_indices
    ]
    # The aggregate is not one SplitIndices, so its binding is built from the
    # folds it is made of. Every fold's fingerprint goes in, which is what ties
    # this guard report to this exact sweep rather than to any cross-user run.
    fingerprint = hashlib.sha256(
        "".join(
            split_fingerprint(dataset, fold_result.indices)
            for fold_result in fold_results
        ).encode("utf-8")
    ).hexdigest()
    aggregate_guards = GuardReport(
        findings=(
            check_participant_concentration(
                dataset,
                test_indices,
                split=Split.CROSS_USER,
                waiver=by_guard.get(GuardName.PARTICIPANT_CONCENTRATION),
            ),
            *label_findings,
            check_evaluation_size(
                aggregate_errors,
                clusters,
                split=Split.CROSS_USER,
                waiver=by_guard.get(GuardName.EVALUATION_SIZE),
            ),
        ),
        binding=GuardBinding(
            split=Split.CROSS_USER,
            split_name=f"{len(fold_results)}-fold cross-user aggregate",
            dataset=protocol.dataset,
            dataset_version=protocol.dataset_version,
            n_train=int(sum(f.indices.n_train for f in fold_results)),
            n_test=int(test_indices.size),
            fingerprint=fingerprint,
        ),
    )
    report.set_guards(aggregate_guards)
    return CrossUserExperimentResult(
        report=report,
        folds=tuple(fold_results),
        test_indices=test_indices,
        predictions=predictions,
        prediction_sha256=_prediction_digest(predictions),
    )
