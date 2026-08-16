"""Turn WatchHand's verified sessions into supervised acoustic windows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.data.watchhand import (
    MODEL_CROP_BINS,
    MODEL_WINDOW_FRAMES,
    SessionData,
)
from wristsonar.types import N_JOINTS

__all__ = ["PoseWindows", "WindowExample"]


@dataclass(frozen=True, slots=True)
class WindowExample:
    """One input window and the pose at its final frame.

    The target is deliberately the final frame rather than a 96-frame sequence.
    That gives an online model a causal contract: it never receives echo data
    recorded after the pose it predicts.
    """

    features: NDArray[np.float32]
    target: NDArray[np.float32]
    participant: str
    session: str
    timestamp_s: float

    def __post_init__(self) -> None:
        if self.features.shape[0] != 2:
            raise ValueError(
                "expected original and differential channels, got "
                f"{self.features.shape}"
            )
        if self.target.shape != (N_JOINTS, 3):
            raise ValueError(
                f"expected ({N_JOINTS}, 3) target, got {self.target.shape}"
            )


class PoseWindows:
    """A causal, session-labelled view over generated WatchHand landmarks."""

    @staticmethod
    def from_session(
        session: SessionData,
        landmarks: NDArray[np.float32] | None,
        *,
        width: int = MODEL_WINDOW_FRAMES,
        stride: int = 1,
        crop_bins: int = MODEL_CROP_BINS,
    ) -> Iterator[WindowExample]:
        """Yield examples without silently interpolating video labels.

        WatchHand's labels originate at video rate, while echo profiles are at
        80 Hz.  The release provides the clock alignment, so each acoustic
        window chooses the most recent video landmark rather than inventing a
        sub-frame pose.  Consumers that want interpolation must state and test
        that separate labelling method.
        """
        poses = session.hand_poses(landmarks)
        if not poses:
            return
        profile_times = np.asarray(
            [session.timestamp_for(i) for i in range(session.n_frames)],
            dtype=np.float64,
        )
        target_times = session.frame_timestamps
        if target_times.shape[0] != len(poses):
            raise ValueError(
                "WatchHand landmark count must match its video frame timestamps: "
                f"{len(poses)} landmarks, {target_times.shape[0]} timestamps"
            )
        targets = np.stack([p.joints for p in poses]).astype(np.float32)

        for start, features in session.windows(
            width=width, stride=stride, crop_bins=crop_bins
        ):
            frame = start + width - 1
            at = profile_times[frame]
            label = int(np.searchsorted(target_times, at, side="right") - 1)
            if label < 0:
                continue
            yield WindowExample(
                features=features,
                target=targets[label],
                participant=f"sub{session.ref.participant}",
                session=session.ref.identity,
                timestamp_s=float(at),
            )
