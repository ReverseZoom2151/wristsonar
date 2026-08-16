"""Pose error metrics, and an explicit account of what each one hides.

Three traps live in this module, and all three are load bearing.

Root alignment. Subtracting the wrist before measuring removes every error in
where the hand is and leaves only error in how the hand is shaped. A wrist
worn sensor has no idea where the wrist is in the world, so wrist relative
poses are the honest representation for this hardware, but the moment a number
is quoted next to a camera system that does estimate global position the two
are answering different questions.

Procrustes alignment. PA-MPJPE additionally fits a rotation, a translation and
usually a scale per frame before measuring. Every one of those degrees of
freedom is error that a real application would still suffer. A pointing
interface cares which way the finger points, and PA-MPJPE has already thrown
that away. An unaligned number and an aligned number are different claims and
this module refuses to let one stand in for the other: the alignment is
stamped into the Protocol notes of every Measurement it produces.

Fingertip dominance. Fingertips are the distal end of the kinematic chain, so
they accumulate the error of every joint proximal to them, and they are also
the joints an interface actually uses. In this literature they typically carry
two to three times the mean error. A mean over twenty one joints therefore
reports mostly the near stationary metacarpals, which barely move relative to
the wrist and are close to free. Per joint output is mandatory here rather
than optional, and JointBreakdown separates fingertips from the rest so the
dilution is visible without having to read twenty one rows.

Units. Poses arrive in metres because that is what types.HandPose stores.
Everything reported leaves in millimetres because that is what the field
quotes, and the conversion happens once, here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from wristsonar.protocol import Measurement, Protocol
from wristsonar.types import JOINT_NAMES, N_JOINTS, HandPose

__all__ = [
    "FINGERTIP_INDICES",
    "FINGERTIP_NAMES",
    "MM_PER_M",
    "ROOT_INDEX",
    "Alignment",
    "JointBreakdown",
    "MetricSet",
    "PoseArray",
    "as_pose_array",
    "evaluate",
    "joint_errors",
    "mpjpe",
    "pa_mpjpe",
    "pck",
    "per_joint_mpjpe",
    "procrustes_align",
]

PoseArray = NDArray[np.float64]
"""Shape (n_frames, N_JOINTS, 3), metres."""

MM_PER_M = 1000.0

ROOT_INDEX = JOINT_NAMES.index("wrist")

FINGERTIP_NAMES: tuple[str, ...] = tuple(
    name for name in JOINT_NAMES if name.endswith("_tip")
)
FINGERTIP_INDICES: tuple[int, ...] = tuple(
    JOINT_NAMES.index(name) for name in FINGERTIP_NAMES
)

DEFAULT_PCK_THRESHOLD_M = 0.02
"""Two centimetres.

Chosen because it is roughly a fingertip width and well inside the 5.7 cm
range resolution of the sensor, which is the point: a system passing PCK at
2 cm is being carried by its shape prior, not by the echo.
"""


class Alignment(Enum):
    """What was removed from the error before it was measured."""

    NONE = "none"
    """Raw joint positions as predicted. The strictest reading, and the only
    one that is directly comparable across systems with different pose
    conventions only if both are genuinely in the same frame."""

    ROOT = "root"
    """Wrist subtracted from both poses. Removes global translation error."""

    PROCRUSTES = "procrustes"
    """Per frame similarity fit: rotation, translation and scale removed.
    Flatters the result the most and is the number most often quoted."""

    def describe(self) -> str:
        if self is Alignment.NONE:
            return "unaligned, global position error included"
        if self is Alignment.ROOT:
            return "root aligned, global translation error removed"
        return "Procrustes aligned, rotation translation and scale removed"

    @property
    def removes_error(self) -> bool:
        """Whether a real application would still suffer what this removes."""
        return self is not Alignment.NONE


def as_pose_array(poses: Sequence[HandPose] | NDArray[np.floating]) -> PoseArray:
    """Coerce poses to a float64 array of shape (n, N_JOINTS, 3).

    Accepts a sequence of HandPose or an array that is already stacked. A
    single pose of shape (N_JOINTS, 3) is promoted to a batch of one.
    """
    if isinstance(poses, np.ndarray):
        arr = np.asarray(poses, dtype=np.float64)
    else:
        if len(poses) == 0:
            raise ValueError("cannot evaluate an empty set of poses")
        arr = np.stack([np.asarray(p.joints, dtype=np.float64) for p in poses])
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected poses of shape (n, {N_JOINTS}, 3), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("cannot evaluate an empty set of poses")
    return arr


def _check_pair(pred: PoseArray, true: PoseArray) -> None:
    if pred.shape != true.shape:
        raise ValueError(
            f"prediction and ground truth disagree on shape: "
            f"{pred.shape} against {true.shape}"
        )
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains non finite values")
    if not np.isfinite(true).all():
        raise ValueError("ground truth contains non finite values")


def _root_align(poses: PoseArray) -> PoseArray:
    return poses - poses[:, ROOT_INDEX : ROOT_INDEX + 1, :]


def procrustes_align(
    source: PoseArray,
    target: PoseArray,
    *,
    with_scale: bool = True,
) -> PoseArray:
    """Fit each source frame onto its target frame by a similarity transform.

    Rotation and translation always, isotropic scale by default because that
    is what PA-MPJPE conventionally includes. Reflection is explicitly
    excluded: a fit that mirrors the hand would turn a left hand prediction
    into a passing right hand one, which is not an alignment but a different
    hand.

    Returns the transformed source. This is exposed rather than hidden inside
    pa_mpjpe so that a caller can see exactly how much the alignment moved the
    prediction, which is the size of the claim the aligned number drops.
    """
    _check_pair(source, target)
    mu_s = source.mean(axis=1, keepdims=True)
    mu_t = target.mean(axis=1, keepdims=True)
    s0 = source - mu_s
    t0 = target - mu_t

    cov = np.einsum("nji,njk->nik", s0, t0)
    u, sing, vt = np.linalg.svd(cov)
    v = np.transpose(vt, (0, 2, 1))
    ut = np.transpose(u, (0, 2, 1))
    det = np.linalg.det(v @ ut)
    correction = np.ones_like(sing)
    correction[:, -1] = np.sign(det)
    d = np.zeros_like(cov)
    idx = np.arange(3)
    d[:, idx, idx] = correction
    rot = v @ d @ ut

    if with_scale:
        var = np.sum(s0**2, axis=(1, 2))
        trace = np.sum(sing * correction, axis=1)
        scale = np.where(var > 0.0, trace / np.where(var > 0.0, var, 1.0), 1.0)
        scale = scale[:, None, None]
    else:
        scale = np.ones((source.shape[0], 1, 1), dtype=np.float64)

    aligned: PoseArray = scale * np.einsum("nij,nkj->nki", rot, s0) + mu_t
    return aligned


def _apply_alignment(
    pred: PoseArray, true: PoseArray, alignment: Alignment
) -> tuple[PoseArray, PoseArray]:
    if alignment is Alignment.NONE:
        return pred, true
    if alignment is Alignment.ROOT:
        return _root_align(pred), _root_align(true)
    return procrustes_align(pred, true), true


def joint_errors(
    pred: Sequence[HandPose] | NDArray[np.floating],
    true: Sequence[HandPose] | NDArray[np.floating],
    *,
    alignment: Alignment = Alignment.ROOT,
) -> NDArray[np.float64]:
    """Per frame per joint Euclidean error in millimetres, shape (n, N_JOINTS).

    Everything else in this module is a reduction of this array, so a caller
    who wants a statistic that is not offered can compute it without going
    around the alignment handling.
    """
    p = as_pose_array(pred)
    t = as_pose_array(true)
    _check_pair(p, t)
    pa, ta = _apply_alignment(p, t, alignment)
    errors: NDArray[np.float64] = (
        np.linalg.norm(pa - ta, axis=2).astype(np.float64) * MM_PER_M
    )
    return errors


def mpjpe(
    pred: Sequence[HandPose] | NDArray[np.floating],
    true: Sequence[HandPose] | NDArray[np.floating],
    *,
    alignment: Alignment = Alignment.ROOT,
) -> float:
    """Mean per joint position error in millimetres.

    The default is root alignment because wrist relative is what this hardware
    can honestly claim, but note that the default is already a choice that
    removes error. Pass Alignment.NONE if the prediction is meant to be in a
    global frame.

    The mean is over all frames and all twenty one joints at once, which is
    the field convention and which dilutes the fingertips with the near static
    metacarpals. Read the JointBreakdown before quoting this.
    """
    return float(joint_errors(pred, true, alignment=alignment).mean())


def pa_mpjpe(
    pred: Sequence[HandPose] | NDArray[np.floating],
    true: Sequence[HandPose] | NDArray[np.floating],
) -> float:
    """Procrustes aligned MPJPE in millimetres.

    A separate function rather than a flag so that it cannot be reached by
    accident. This number has had per frame rotation, translation and scale
    fitted away, so it describes shape agreement only. If the product points
    at something, rotation is the product and this metric has deleted it.
    """
    return mpjpe(pred, true, alignment=Alignment.PROCRUSTES)


def pck(
    pred: Sequence[HandPose] | NDArray[np.floating],
    true: Sequence[HandPose] | NDArray[np.floating],
    *,
    threshold_m: float = DEFAULT_PCK_THRESHOLD_M,
    alignment: Alignment = Alignment.ROOT,
) -> float:
    """Fraction of joints within threshold_m of ground truth, in 0 to 1.

    Threshold choice is the whole metric. At 2 cm this is comfortably inside
    the sensor's 5.7 cm range resolution, so a high score is evidence about
    the learned hand prior rather than about the echo, and the threshold must
    always be quoted with the number.
    """
    if threshold_m <= 0:
        raise ValueError("PCK threshold must be positive")
    errors = joint_errors(pred, true, alignment=alignment)
    return float((errors <= threshold_m * MM_PER_M).mean())


def per_joint_mpjpe(
    pred: Sequence[HandPose] | NDArray[np.floating],
    true: Sequence[HandPose] | NDArray[np.floating],
    *,
    alignment: Alignment = Alignment.ROOT,
) -> dict[str, float]:
    """Mean error in millimetres for each named joint."""
    errors = joint_errors(pred, true, alignment=alignment)
    means = errors.mean(axis=0)
    return {name: float(means[i]) for i, name in enumerate(JOINT_NAMES)}


@dataclass(frozen=True, slots=True)
class JointBreakdown:
    """Per joint error, plus the fingertip against non fingertip contrast.

    Mandatory output rather than a diagnostic extra. The single mean over
    twenty one joints is the number the field quotes and it is the number that
    hides the failure, because fifteen of the twenty one joints are proximal
    and move very little relative to the wrist.
    """

    per_joint_mm: dict[str, float]
    alignment: Alignment

    @property
    def fingertip_mm(self) -> float:
        return float(np.mean([self.per_joint_mm[name] for name in FINGERTIP_NAMES]))

    @property
    def non_fingertip_mm(self) -> float:
        rest = [
            v for k, v in self.per_joint_mm.items() if k not in set(FINGERTIP_NAMES)
        ]
        return float(np.mean(rest))

    @property
    def dilution_factor(self) -> float:
        """How much worse the fingertips are than everything else.

        A factor near one usually means the predictor is not tracking fingers
        at all, for example a mean pose baseline whose error is dominated by
        pose variance everywhere. A factor of two to three is what a working
        system in this literature looks like.
        """
        rest = self.non_fingertip_mm
        if rest <= 0.0:
            return float("inf")
        return self.fingertip_mm / rest

    def worst(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.per_joint_mm.items(), key=lambda kv: -kv[1])[:n]

    def render(self) -> str:
        lines = [f"per joint error in mm ({self.alignment.describe()})"]
        for name in JOINT_NAMES:
            marker = " *" if name in FINGERTIP_NAMES else "  "
            lines.append(f"  {name:<12}{marker} {self.per_joint_mm[name]:8.2f}")
        lines.append(
            f"  fingertips (*) {self.fingertip_mm:.2f} mm, "
            f"rest {self.non_fingertip_mm:.2f} mm, "
            f"ratio {self.dilution_factor:.2f}"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Everything measured on one prediction set, each item protocol stamped.

    Both the aligned and the unaligned MPJPE are always present. They are
    different claims, and carrying both means a report cannot silently quote
    the flattering one.
    """

    mpjpe: Measurement
    pa_mpjpe: Measurement
    pck: Measurement
    breakdown: JointBreakdown
    alignment: Alignment
    pck_threshold_m: float
    n_frames: int

    @property
    def protocol(self) -> Protocol:
        return self.mpjpe.protocol

    def summary(self) -> str:
        return (
            f"MPJPE {self.mpjpe.value:.2f} mm "
            f"({self.alignment.value}), "
            f"PA-MPJPE {self.pa_mpjpe.value:.2f} mm, "
            f"PCK@{self.pck_threshold_m * 100:g}cm {self.pck.value:.3f}, "
            f"fingertips {self.breakdown.fingertip_mm:.2f} mm"
        )


def evaluate(
    pred: Sequence[HandPose] | NDArray[np.floating],
    true: Sequence[HandPose] | NDArray[np.floating],
    protocol: Protocol,
    *,
    alignment: Alignment = Alignment.ROOT,
    pck_threshold_m: float = DEFAULT_PCK_THRESHOLD_M,
) -> MetricSet:
    """Measure a prediction set and stamp the protocol onto every number.

    The alignment is written into the returned Protocol's notes, so a
    Measurement lifted out of here and pasted somewhere else still says what
    was removed from it before it was computed.
    """
    p = as_pose_array(pred)
    t = as_pose_array(true)
    _check_pair(p, t)
    n = int(p.shape[0])

    def stamped(extra: str) -> Protocol:
        note = protocol.notes
        joined = f"{note}; {extra}" if note else extra
        return replace(protocol, notes=joined)

    mpjpe_value = mpjpe(p, t, alignment=alignment)
    pa_value = pa_mpjpe(p, t)
    pck_value = pck(p, t, threshold_m=pck_threshold_m, alignment=alignment)

    return MetricSet(
        mpjpe=Measurement(
            value=mpjpe_value,
            unit="mm",
            protocol=stamped(f"mpjpe {alignment.describe()}"),
            samples=n,
        ),
        pa_mpjpe=Measurement(
            value=pa_value,
            unit="mm",
            protocol=stamped(f"pa-mpjpe {Alignment.PROCRUSTES.describe()}"),
            samples=n,
        ),
        pck=Measurement(
            value=pck_value,
            unit=f"fraction@{pck_threshold_m * 100:g}cm",
            protocol=stamped(f"pck {alignment.describe()}"),
            samples=n,
        ),
        breakdown=JointBreakdown(
            per_joint_mm=per_joint_mpjpe(p, t, alignment=alignment),
            alignment=alignment,
        ),
        alignment=alignment,
        pck_threshold_m=pck_threshold_m,
        n_frames=n,
    )
