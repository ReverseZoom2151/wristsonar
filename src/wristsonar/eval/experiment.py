"""One execution path for a protocol-bound acoustic pose experiment.

Metrics, baselines and guards are useful only when they consume the exact same
partition and protocol as the learned model.  This module owns that join.  It
does not expose a convenience mode for skipping baselines or guards: a caller
can collect a blocking report, but cannot turn it into a headline by omitting
the uncomfortable rows.

The same argument applies one level up.  A single split is one question, and a
harness that only runs one question at a time leaves the complete set to be
assembled by hand, which is the assembly that gets skipped on the day the
deadline is close.  ``run_split_suite`` is therefore the ordinary entry point:
it runs every split this dataset can support, with one model identity and one
set of baselines, and returns the ``MultiSplitReport`` that refuses a headline
unless the set is complete.  Running a single flattering split is still
possible through ``run_experiment``, but it is now the longer route.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
from wristsonar.eval.report import MultiSplitReport, Report
from wristsonar.eval.splits import (
    Dataset,
    SplitIndices,
    assert_no_leakage,
    cross_device_split,
    cross_session_split,
    cross_user_folds,
    within_session_split,
)
from wristsonar.protocol import Protocol, Split

__all__ = [
    "CrossUserExperimentResult",
    "ExperimentError",
    "ExperimentResult",
    "FoldBuilder",
    "FoldFamilyResult",
    "SplitAbsence",
    "SplitSuiteResult",
    "run_cross_user_experiment",
    "run_experiment",
    "run_split_suite",
]

Predictor = Callable[[Dataset, NDArray[np.intp], NDArray[np.intp]], PoseArray]

FoldBuilder = Callable[[Dataset], tuple[SplitIndices, ...]]
"""Cuts every fold of one split's leave-one-out family, or raises ValueError.

The ValueError is the contract that lets a dataset which cannot express a split
be recorded as a stated absence rather than crashing the sweep.
"""


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
class FoldFamilyResult:
    """Every fold of one leave-one-out family plus its only quotable aggregate.

    A family is several partitions answering one question: leave-one-user-out
    is one fold per participant, leave-one-session-out is one fold per session,
    leave-one-device-out is one fold per device model.  Each fold on its own is
    a single held-out unit, which is the shape the guards exist to distrust,
    so the number that leaves this object is the aggregate over pooled
    predictions and the folds are kept beside it rather than instead of it.
    """

    report: Report
    folds: tuple[ExperimentResult, ...]
    test_indices: NDArray[np.intp]
    predictions: PoseArray
    prediction_sha256: str

    def __post_init__(self) -> None:
        if not self.folds:
            raise ExperimentError("fold aggregation needs at least one fold")
        if self.test_indices.shape != (self.predictions.shape[0],):
            raise ExperimentError("aggregate test indices and predictions disagree")
        if len(set(self.test_indices.tolist())) != self.test_indices.size:
            raise ExperimentError("the folds overlap in their held-out examples")
        if self.prediction_sha256 != _prediction_digest(self.predictions):
            raise ExperimentError(
                "aggregate prediction digest does not match the array"
            )

    @property
    def split(self) -> Split:
        """The one split every fold in this family was cut for."""
        return self.folds[0].indices.split

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


CrossUserExperimentResult = FoldFamilyResult
"""The leave-one-user-out family, under the name it was published as here.

Kept because cross-user is the family callers reach for first and the name
appears in existing call sites, not because the container ever held anything
specific to participants.
"""


@dataclass(frozen=True, slots=True)
class SplitAbsence:
    """A split the dataset cannot support, and the reason it cannot.

    The distinction this type exists to preserve: a split missing because the
    recordings only ever used one device model is a fact about the study, while
    a split missing because nobody ran it is a fact about the run. Both leave
    the same hole in a MultiSplitReport, and only one of them can be fixed by
    running more code. An absence is therefore recorded and printed rather than
    swallowed, and it still leaves can_headline False, because a claim that
    cannot be checked is not a claim that has been checked.
    """

    split: Split
    reason: str

    def describe(self) -> str:
        return f"{self.split.name} cannot be measured on this dataset: {self.reason}"


@dataclass(frozen=True, slots=True)
class SplitSuiteResult:
    """Every split that could be run, every split that could not, in one object."""

    report: MultiSplitReport
    families: dict[Split, FoldFamilyResult]
    absences: tuple[SplitAbsence, ...]

    def __post_init__(self) -> None:
        # MultiSplitReport refuses a set whose splits evaluate different model
        # names. Checking it here as well means the suite cannot be assembled
        # into that state at all, rather than being assembled and then found
        # unquotable at rendering time.
        names = self.report.model_names()
        if len(names) > 1:
            raise ExperimentError(
                f"a split suite must evaluate one model on every split, got "
                f"{list(names)}. A system measured on the easy splits and a "
                f"different one measured on the hard splits is not one result."
            )
        unaccounted = [
            split.name
            for split in Split
            if split not in self.families and self.absence_for(split) is None
        ]
        if unaccounted:
            raise ExperimentError(
                f"splits {unaccounted} are neither measured nor recorded as "
                f"unsupportable; a split that is simply absent cannot be told "
                f"apart from one nobody bothered to run"
            )

    def absence_for(self, split: Split) -> SplitAbsence | None:
        for absence in self.absences:
            if absence.split is split:
                return absence
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "report": self.report.to_dict(),
            "can_headline": self.report.can_headline,
            "absences": [
                {"split": a.split.name, "reason": a.reason} for a in self.absences
            ],
            "folds": {
                split.name: [fold.indices.name for fold in family.folds]
                for split, family in sorted(
                    self.families.items(), key=lambda kv: kv[0].value
                )
            },
        }

    def write(self, directory: Path) -> None:
        """Write the set, then every family under it, as one inseparable run."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.txt").write_text(
            self.report.render_text() + "\n", encoding="utf-8"
        )
        (directory / "report.json").write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        for split, family in self.families.items():
            family.write(directory / "splits" / split.name.lower())


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


def _run_fold_family(
    dataset: Dataset,
    folds: Sequence[SplitIndices],
    protocol: Protocol,
    *,
    name: str,
    predictor: Predictor,
    title: str | None = None,
    waivers: Sequence[Waiver] = (),
    alignment: Alignment = Alignment.ROOT,
) -> FoldFamilyResult:
    """Turn a leave-one-out family into the single report it is allowed to be.

    How the family becomes one number, and why it is this way.  Every fold's
    predictions are pooled, put back in original example order, and evaluated
    once against the pooled ground truth.  The aggregate is therefore a mean
    over held-out examples, weighted by how many examples each fold held out,
    and no fold can be preferred over another because no fold is ever compared
    to another.  The alternative, scoring each fold and combining the scores,
    requires choosing a combiner, and every such choice is a place where the
    best fold can be selected once the numbers are visible.  There is no
    argument to this function that names a fold, and none that would let a
    caller drop one.

    The spread the pooling hides is put back on the report as a note: the
    minimum, median and maximum per-fold MPJPE, so that a wide spread across
    people or devices is visible next to the aggregate rather than only in the
    per-fold files.  Every fold is retained on the result and written to disk,
    which is what makes the pooled number checkable.

    Baselines are refit inside every fold.  A single global fit would train on
    the unit it is evaluating in all but one fold, which is a quiet leak that
    would flatter the baselines and therefore make the model look worse in a
    way nobody would question.
    """
    if not folds:
        raise ExperimentError(
            f"a {protocol.split.name} fold family needs at least one fold"
        )
    if not name.strip():
        raise ExperimentError("model name must not be empty")
    wrong = [fold.name for fold in folds if fold.split is not protocol.split]
    if wrong:
        raise ExperimentError(
            f"protocol declares {protocol.split.name}, but folds {wrong} are "
            f"cut for a different split"
        )

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
        # that fold's train half and must never cross the held-out boundary.
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
        # A fold that holds out a single unit is expected to fail
        # concentration. Preserve all other diagnostics locally; the aggregate
        # evaluates concentration and precision over every held-out unit.
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

    label = protocol.split.name.lower().replace("_", "-")
    fold_values = sorted(
        fold_result.report.models[0].mpjpe.value for fold_result in fold_results
    )
    report = Report(
        title=title or f"{name} [{label} leave-one-out aggregate]",
        notes=(
            f"pooled over {len(fold_results)} folds and evaluated once, so no "
            f"fold can be preferred to another. Per-fold model MPJPE ran "
            f"{fold_values[0]:.2f} to {fold_values[-1]:.2f} mm, median "
            f"{float(np.median(fold_values)):.2f} mm; the pooled figure is the "
            f"quotable one and the best fold is not a result."
        ),
    )
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
    # this guard report to this exact sweep rather than to any sweep of the
    # same shape.
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
                split=protocol.split,
                waiver=by_guard.get(GuardName.PARTICIPANT_CONCENTRATION),
            ),
            *label_findings,
            check_evaluation_size(
                aggregate_errors,
                clusters,
                split=protocol.split,
                waiver=by_guard.get(GuardName.EVALUATION_SIZE),
            ),
        ),
        binding=GuardBinding(
            split=protocol.split,
            split_name=f"{len(fold_results)}-fold {label} aggregate",
            dataset=protocol.dataset,
            dataset_version=protocol.dataset_version,
            n_train=int(sum(f.indices.n_train for f in fold_results)),
            n_test=int(test_indices.size),
            fingerprint=fingerprint,
        ),
    )
    report.set_guards(aggregate_guards)
    return FoldFamilyResult(
        report=report,
        folds=tuple(fold_results),
        test_indices=test_indices,
        predictions=predictions,
        prediction_sha256=_prediction_digest(predictions),
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
) -> FoldFamilyResult:
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
    return _run_fold_family(
        dataset,
        cross_user_folds(dataset),
        protocol,
        name=name,
        predictor=predictor,
        title=title,
        waivers=waivers,
        alignment=alignment,
    )


def _within_session_folds(
    dataset: Dataset, *, test_fraction: float
) -> tuple[SplitIndices, ...]:
    """One contiguous tail per session, for every session that can be cut."""
    folds = [
        within_session_split(dataset, session=session, test_fraction=test_fraction)
        for session in dataset.sessions
        if dataset.indices_where(session=session).size >= 2
    ]
    if not folds:
        raise ValueError(
            "no session holds two samples, so nothing can be held out inside "
            "one continuous wearing"
        )
    return tuple(folds)


def _cross_session_folds(dataset: Dataset) -> tuple[SplitIndices, ...]:
    """Leave one session out, for every session of every remounted participant.

    Participants with a single session contribute no folds, because a
    cross-session split needs a remount and cutting one session in half would
    be a within-session split under a stronger label. That exclusion is decided
    from the metadata before any prediction is made, so it cannot be a choice
    about the numbers.
    """
    folds: list[SplitIndices] = []
    for participant in dataset.participants:
        own = dataset.indices_where(participant=participant)
        sessions = sorted({dataset.meta[int(i)].session_key for i in own})
        if len(sessions) < 2:
            continue
        folds.extend(
            cross_session_split(
                dataset, participant=participant, held_out_session=session
            )
            for session in sessions
        )
    if not folds:
        raise ValueError(
            "no participant recorded a second session, so the device was never "
            "remounted and there is no cross-session question this dataset can "
            "answer"
        )
    return tuple(folds)


def _cross_device_folds(dataset: Dataset) -> tuple[SplitIndices, ...]:
    """Leave one device model out, once for each model in the dataset."""
    devices = dataset.devices
    if len(devices) < 2:
        raise ValueError(
            f"every recording came from device model {devices[0]!r}, so no "
            f"device can be held out and nothing here says how the system "
            f"behaves on hardware it has never heard"
        )
    return tuple(
        cross_device_split(dataset, held_out_device=device) for device in devices
    )


def run_split_suite(
    dataset: Dataset,
    protocol: Protocol,
    *,
    name: str,
    predictor: Predictor,
    title: str | None = None,
    waivers: Sequence[Waiver] = (),
    alignment: Alignment = Alignment.ROOT,
    within_session_test_fraction: float = 0.2,
) -> SplitSuiteResult:
    """Run one model and its baselines across every split, as one result.

    This is the entry point the project's phase 1 exit criterion is written
    against, and it exists so that producing the complete set is the easy path.
    Assembling a MultiSplitReport by hand means four separate runs, four
    protocols kept in step with each other and four guard reports attached to
    the right rows, and a guard rail that needs that much assembly is one that
    gets skipped exactly when the deadline makes it matter.

    One model identity for the whole sweep.  ``name`` is a single argument and
    is used unchanged on every split, so the mismatch MultiSplitReport refuses
    cannot be produced from here at all.  ``protocol`` is likewise given once,
    with only its split field replaced per split, so the ground truth source,
    the dataset version and the calibration budget cannot drift between the
    easy splits and the hard ones.

    Each split is a leave-one-out family rather than a single partition, and
    ``_run_fold_family`` pools the folds into one report; see its docstring for
    why the pooled number cannot be the best fold.

    A split the dataset cannot support becomes a SplitAbsence carrying the
    reason the constructor gave, not an exception and not a silent gap.  The
    honest splits are required by MultiSplitReport, so an absence still leaves
    can_headline False, and the reason is printed with the set so that a reader
    can tell a study that had one device model from a run that skipped the
    cross-device number.
    """
    if not name.strip():
        raise ExperimentError("model name must not be empty")

    builders: tuple[tuple[Split, FoldBuilder], ...] = (
        (
            Split.WITHIN_SESSION,
            lambda data: _within_session_folds(
                data, test_fraction=within_session_test_fraction
            ),
        ),
        (Split.CROSS_SESSION, _cross_session_folds),
        (Split.CROSS_USER, cross_user_folds),
        (Split.CROSS_DEVICE, _cross_device_folds),
    )

    families: dict[Split, FoldFamilyResult] = {}
    absences: list[SplitAbsence] = []
    combined = MultiSplitReport(title=title or f"{name}, every split")
    for split, build in builders:
        try:
            folds = build(dataset)
        except ValueError as error:
            # Only a dataset that cannot express the split lands here.
            # LeakageError is an AssertionError and is deliberately not caught:
            # a leaking split is a broken invariant, not a stated absence.
            absences.append(SplitAbsence(split=split, reason=str(error)))
            continue
        family = _run_fold_family(
            dataset,
            folds,
            replace(protocol, split=split),
            name=name,
            predictor=predictor,
            waivers=waivers,
            alignment=alignment,
        )
        families[split] = family
        combined.add(family.report)

    combined.notes = _absence_notes(tuple(absences))
    return SplitSuiteResult(
        report=combined, families=families, absences=tuple(absences)
    )


def _absence_notes(absences: Sequence[SplitAbsence]) -> str:
    if not absences:
        return (
            "every split in the enum was measurable on this dataset, so no "
            "split is missing for want of data"
        )
    body = "\n  ".join(f"- {absence.describe()}" for absence in absences)
    return (
        "Splits this dataset cannot support. These are absent because the "
        "recordings cannot express them, not because nobody ran them, and "
        "neither reason permits a headline:\n  " + body
    )
