"""Metric correctness, including what each alignment quietly removes."""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.synthetic import make_pose
from wristsonar.eval.metrics import (
    FINGERTIP_NAMES,
    Alignment,
    JointBreakdown,
    evaluate,
    joint_errors,
    mpjpe,
    pa_mpjpe,
    pck,
    per_joint_mpjpe,
    procrustes_align,
)
from wristsonar.protocol import GroundTruth, Protocol, Split
from wristsonar.types import JOINT_NAMES, N_JOINTS, HandPose


def _protocol() -> Protocol:
    return Protocol(
        split=Split.CROSS_USER,
        dataset="synthetic",
        dataset_version="0",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=6,
    )


def test_mpjpe_matches_hand_computed_value() -> None:
    true = np.zeros((1, N_JOINTS, 3), dtype=np.float64)
    pred = np.zeros((1, N_JOINTS, 3), dtype=np.float64)
    pred[0, 3] = [0.003, 0.004, 0.0]  # 5 mm
    pred[0, 7] = [0.0, 0.0, 0.012]  # 12 mm

    expected = (5.0 + 12.0) / N_JOINTS
    assert mpjpe(pred, true, alignment=Alignment.NONE) == pytest.approx(expected)
    # Both wrists sit at the origin, so root alignment changes nothing here.
    assert mpjpe(pred, true, alignment=Alignment.ROOT) == pytest.approx(expected)

    errors = joint_errors(pred, true, alignment=Alignment.NONE)
    assert errors.shape == (1, N_JOINTS)
    assert errors[0, 3] == pytest.approx(5.0)
    assert errors[0, 7] == pytest.approx(12.0)
    assert errors[0, 0] == pytest.approx(0.0)


def test_root_alignment_removes_a_pure_translation() -> None:
    true = make_pose(np.full(5, 0.2))[None, ...]
    offset = np.array([0.05, -0.02, 0.011], dtype=np.float64)
    pred = true + offset

    unaligned = mpjpe(pred, true, alignment=Alignment.NONE)
    aligned = mpjpe(pred, true, alignment=Alignment.ROOT)

    assert unaligned == pytest.approx(float(np.linalg.norm(offset)) * 1000.0)
    assert aligned == pytest.approx(0.0, abs=1e-9)
    assert unaligned > 50.0


def test_procrustes_removes_a_known_rotation_scale_and_translation() -> None:
    true = np.stack([make_pose(np.full(5, 0.3)), make_pose(np.full(5, 0.7))])
    angle = 0.4
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation = np.array([0.02, 0.03, -0.01], dtype=np.float64)
    pred = 1.3 * np.einsum("ij,nkj->nki", rot, true) + translation

    assert mpjpe(pred, true, alignment=Alignment.NONE) > 10.0
    assert pa_mpjpe(pred, true) == pytest.approx(0.0, abs=1e-6)

    recovered = procrustes_align(pred, true)
    assert np.allclose(recovered, true, atol=1e-9)


def test_procrustes_refuses_to_fix_a_reflection() -> None:
    true = make_pose(np.full(5, 0.4))[None, ...]
    mirrored = true.copy()
    mirrored[..., 0] *= -1.0
    # A reflection is a different hand, not a misalignment, so the fit must
    # leave a large residual rather than mirroring it away.
    assert pa_mpjpe(mirrored, true) > 5.0


def test_pck_counts_joints_inside_the_threshold() -> None:
    true = np.zeros((1, N_JOINTS, 3), dtype=np.float64)
    pred = np.zeros((1, N_JOINTS, 3), dtype=np.float64)
    pred[0, 7] = [0.0, 0.0, 0.012]

    assert pck(pred, true, threshold_m=0.01, alignment=Alignment.NONE) == pytest.approx(
        (N_JOINTS - 1) / N_JOINTS
    )
    assert pck(pred, true, threshold_m=0.02, alignment=Alignment.NONE) == 1.0

    with pytest.raises(ValueError, match="threshold must be positive"):
        pck(pred, true, threshold_m=0.0)


def test_per_joint_output_exposes_fingertip_dominance() -> None:
    true = np.zeros((4, N_JOINTS, 3), dtype=np.float64)
    pred = np.zeros((4, N_JOINTS, 3), dtype=np.float64)
    tip_indices = [JOINT_NAMES.index(name) for name in FINGERTIP_NAMES]
    pred[:, tip_indices, 2] = 0.020

    per_joint = per_joint_mpjpe(pred, true, alignment=Alignment.NONE)
    assert set(per_joint) == set(JOINT_NAMES)
    breakdown = JointBreakdown(per_joint_mm=per_joint, alignment=Alignment.NONE)

    assert breakdown.fingertip_mm == pytest.approx(20.0)
    assert breakdown.non_fingertip_mm == pytest.approx(0.0)
    # The mean over 21 joints reports under 5 mm for a 20 mm fingertip error.
    assert mpjpe(pred, true, alignment=Alignment.NONE) < 5.0
    assert breakdown.worst(1)[0][0] in FINGERTIP_NAMES
    assert "fingertips" in breakdown.render()


def test_evaluate_stamps_the_alignment_into_the_protocol() -> None:
    true = np.stack([make_pose(np.full(5, 0.3))])
    pred = true + 0.001
    metrics = evaluate(pred, true, _protocol(), alignment=Alignment.ROOT)

    assert "root aligned" in metrics.mpjpe.protocol.notes
    assert "Procrustes aligned" in metrics.pa_mpjpe.protocol.notes
    assert metrics.n_frames == 1
    assert metrics.mpjpe.samples == 1
    assert metrics.mpjpe.unit == "mm"
    assert "MPJPE" in metrics.summary()


def test_pose_objects_and_arrays_are_interchangeable() -> None:
    joints = make_pose(np.full(5, 0.25)).astype(np.float32)
    pose = HandPose(joints=joints)
    assert mpjpe([pose], joints[None, ...]) == pytest.approx(0.0, abs=1e-9)


def test_shape_and_emptiness_are_rejected() -> None:
    true = np.zeros((2, N_JOINTS, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="disagree on shape"):
        mpjpe(np.zeros((3, N_JOINTS, 3)), true)
    with pytest.raises(ValueError, match="empty set of poses"):
        mpjpe(np.zeros((0, N_JOINTS, 3)), np.zeros((0, N_JOINTS, 3)))
    with pytest.raises(ValueError, match="non finite"):
        mpjpe(np.full((2, N_JOINTS, 3), np.nan), true)
