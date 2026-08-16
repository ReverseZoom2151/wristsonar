"""Evaluation harness: the actual product of this project.

The binding problem in acoustic hand pose sensing is measurement, not
modelling. Range resolution is about 5.7 cm and the reported millimetre
figures come from regressing a coarse echo signature onto the low dimensional
manifold hands occupy. Under those conditions the thing that decides whether a
number means anything is the protocol that produced it, so the harness is
first class and the model is not.

Everything this package exports at the top level carries a Protocol. Nothing
exported here returns a bare float, which is why mpjpe, pa_mpjpe, pck,
per_joint_mpjpe and joint_errors are deliberately absent from the list below:
they are the raw reductions, they return unstamped numbers and arrays by
design, and a caller who genuinely wants one imports it from
wristsonar.eval.metrics and thereby says so. The stamped route is
metrics.evaluate, which returns a MetricSet in which every figure is a
Measurement.
"""

from __future__ import annotations

from wristsonar.eval.baselines import (
    Baseline,
    MeanPoseBaseline,
    NearestNeighbourBaseline,
    PerSubjectMeanPoseBaseline,
    evaluate_baselines,
)
from wristsonar.eval.calibration import (
    CALIBRATION_BUDGETS_MIN,
    CalibrationCurve,
    CalibrationPoint,
    sweep_calibration,
)
from wristsonar.eval.experiment import (
    CrossUserExperimentResult,
    ExperimentError,
    ExperimentResult,
    FoldFamilyResult,
    SplitAbsence,
    SplitSuiteResult,
    run_cross_user_experiment,
    run_experiment,
    run_split_suite,
)
from wristsonar.eval.guard import (
    GuardBinding,
    GuardFinding,
    GuardName,
    GuardReport,
    GuardViolation,
    Waiver,
    check_evaluation_size,
    check_label_distribution_match,
    check_participant_concentration,
    run_guards,
)
from wristsonar.eval.metrics import (
    FINGERTIP_INDICES,
    FINGERTIP_NAMES,
    NON_ROOT_INDICES,
    ROOT_INDEX,
    Alignment,
    JointBreakdown,
    MetricSet,
    evaluate,
)
from wristsonar.eval.report import MultiSplitReport, Report, ResultRow
from wristsonar.eval.splits import (
    Dataset,
    LeakageError,
    SampleMeta,
    SplitIndices,
    assert_no_leakage,
    cross_device_split,
    cross_session_split,
    cross_user_folds,
    normalise_identity,
    within_session_split,
)

__all__ = [
    "CALIBRATION_BUDGETS_MIN",
    "FINGERTIP_INDICES",
    "FINGERTIP_NAMES",
    "NON_ROOT_INDICES",
    "ROOT_INDEX",
    "Alignment",
    "Baseline",
    "CalibrationCurve",
    "CalibrationPoint",
    "CrossUserExperimentResult",
    "Dataset",
    "ExperimentError",
    "ExperimentResult",
    "FoldFamilyResult",
    "GuardBinding",
    "GuardFinding",
    "GuardName",
    "GuardReport",
    "GuardViolation",
    "JointBreakdown",
    "LeakageError",
    "MeanPoseBaseline",
    "MetricSet",
    "MultiSplitReport",
    "NearestNeighbourBaseline",
    "PerSubjectMeanPoseBaseline",
    "Report",
    "ResultRow",
    "SampleMeta",
    "SplitAbsence",
    "SplitIndices",
    "SplitSuiteResult",
    "Waiver",
    "assert_no_leakage",
    "check_evaluation_size",
    "check_label_distribution_match",
    "check_participant_concentration",
    "cross_device_split",
    "cross_session_split",
    "cross_user_folds",
    "evaluate",
    "evaluate_baselines",
    "normalise_identity",
    "run_cross_user_experiment",
    "run_experiment",
    "run_guards",
    "run_split_suite",
    "sweep_calibration",
    "within_session_split",
]
