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

from wristsonar.eval.baselines import Baseline, evaluate_baselines
from wristsonar.eval.guard import GuardReport, Waiver, run_guards
from wristsonar.eval.metrics import Alignment, PoseArray, evaluate, joint_errors
from wristsonar.eval.report import Report
from wristsonar.eval.splits import Dataset, SplitIndices, assert_no_leakage
from wristsonar.protocol import Protocol

__all__ = ["ExperimentError", "ExperimentResult", "run_experiment"]

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
