"""Evaluation harness: the actual product of this project.

The binding problem in acoustic hand pose sensing is measurement, not
modelling. Range resolution is about 5.7 cm and the reported millimetre
figures come from regressing a coarse echo signature onto the low dimensional
manifold hands occupy. Under those conditions the thing that decides whether a
number means anything is the protocol that produced it, so the harness is
first class and the model is not.

Everything reportable here carries a Protocol. There is no code path that
emits a bare float as a result.
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
from wristsonar.eval.guard import (
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
    ROOT_INDEX,
    Alignment,
    JointBreakdown,
    MetricSet,
    evaluate,
    joint_errors,
    mpjpe,
    pa_mpjpe,
    pck,
    per_joint_mpjpe,
)
from wristsonar.eval.report import Report, ResultRow
from wristsonar.eval.splits import (
    Dataset,
    LeakageError,
    SampleMeta,
    SplitIndices,
    assert_no_leakage,
    cross_device_split,
    cross_session_split,
    cross_user_folds,
    within_session_split,
)

__all__ = [
    "CALIBRATION_BUDGETS_MIN",
    "FINGERTIP_INDICES",
    "FINGERTIP_NAMES",
    "ROOT_INDEX",
    "Alignment",
    "Baseline",
    "CalibrationCurve",
    "CalibrationPoint",
    "Dataset",
    "GuardFinding",
    "GuardName",
    "GuardReport",
    "GuardViolation",
    "JointBreakdown",
    "LeakageError",
    "MeanPoseBaseline",
    "MetricSet",
    "NearestNeighbourBaseline",
    "PerSubjectMeanPoseBaseline",
    "Report",
    "ResultRow",
    "SampleMeta",
    "SplitIndices",
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
    "joint_errors",
    "mpjpe",
    "pa_mpjpe",
    "pck",
    "per_joint_mpjpe",
    "run_guards",
    "sweep_calibration",
    "within_session_split",
]
