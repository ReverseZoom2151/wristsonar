"""Synthetic hands, offline and deterministic.

The generator is built to have the properties this harness is supposed to
detect. Hand size is a per participant constant, resting curl has a per
participant bias, mounting geometry shifts the features per session, and the
poses live on a low dimensional manifold parameterised by five curl values.
That is a caricature of the real thing, but it is a caricature with the same
failure surfaces: a per subject mean pose is a strong predictor, and a feature
vector carries a session fingerprint that a model could latch onto instead of
the fingers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wristsonar.eval.splits import Dataset, SampleMeta
from wristsonar.types import N_JOINTS

FINGER_BASE_X: tuple[float, ...] = (-0.035, -0.015, 0.0, 0.017, 0.033)
FINGER_SEGMENTS: tuple[tuple[float, float, float], ...] = (
    (0.035, 0.030, 0.024),
    (0.042, 0.026, 0.020),
    (0.045, 0.028, 0.021),
    (0.041, 0.025, 0.019),
    (0.033, 0.020, 0.017),
)


def make_pose(curl: NDArray[np.float64], scale: float = 1.0) -> NDArray[np.float64]:
    """A wrist relative hand with each finger bent by its own curl angle."""
    if curl.shape != (5,):
        raise ValueError("curl must have one angle per finger")
    joints = np.zeros((N_JOINTS, 3), dtype=np.float64)
    index = 1
    for finger, segments in enumerate(FINGER_SEGMENTS):
        position = np.array([FINGER_BASE_X[finger], 0.02, 0.0], dtype=np.float64)
        joints[index] = position
        index += 1
        angle = 0.0
        for length in segments:
            angle += float(curl[finger])
            direction = np.array([0.0, np.cos(angle), -np.sin(angle)], dtype=np.float64)
            position = position + length * direction
            joints[index] = position
            index += 1
    return joints * scale


def make_dataset(
    *,
    n_participants: int = 6,
    n_sessions: int = 2,
    n_frames: int = 40,
    devices: tuple[str, ...] = ("watch-a",),
    seed: int = 7,
    feature_noise: float = 0.02,
    session_shift: float = 0.3,
) -> Dataset:
    """A dataset with participants, sessions, devices, poses and features.

    Features are the curl vector plus a per session offset plus noise, which
    is the shape of the real problem: the signal carries the pose, and it also
    carries a mounting fingerprint that is constant within a session.
    """
    rng = np.random.default_rng(seed)
    poses: list[NDArray[np.float64]] = []
    features: list[NDArray[np.float64]] = []
    meta: list[SampleMeta] = []

    for p in range(n_participants):
        scale = 0.9 + 0.2 * rng.random()
        bias = 0.25 * rng.standard_normal(5)
        for s in range(n_sessions):
            shift = session_shift * rng.standard_normal(5)
            device = devices[(p + s) % len(devices)]
            for f in range(n_frames):
                curl = bias + 0.45 * rng.random(5)
                poses.append(make_pose(curl, scale))
                features.append(
                    np.concatenate(
                        [
                            curl + shift + feature_noise * rng.standard_normal(5),
                            np.array([scale], dtype=np.float64),
                        ]
                    )
                )
                meta.append(
                    SampleMeta(
                        participant=f"P{p:02d}",
                        session=f"P{p:02d}-S{s}",
                        device=device,
                        timestamp_s=float(f),
                    )
                )

    return Dataset.build(
        poses=np.stack(poses),
        meta=meta,
        features=np.stack(features),
    )
