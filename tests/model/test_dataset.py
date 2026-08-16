from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np

from wristsonar.data.watchhand import SessionData, SessionRef, Study
from wristsonar.model.dataset import PoseWindows
from wristsonar.preprocess import WATCHHAND_PREPROCESSING

_SIXTEEN_FRAME_WINDOW = WATCHHAND_PREPROCESSING.with_window_frames(16)


def test_pose_windows_are_causal_and_use_the_aligned_previous_label() -> None:
    ref = SessionRef(Study.MAIN, 1, 0, PurePosixPath("session"), "session")
    profiles = np.ones((600, 100), dtype=np.float32)
    diff = np.ones((600, 100), dtype=np.float32)
    session = SessionData(
        ref=ref,
        profiles=profiles,
        diff_profiles=diff,
        frame_length=600,
        sample_rate=48_000,
        audio_sync_index=0,
        gt_sync_timestamp=0.0,
        frame_timestamps=np.asarray([0.0, 0.8, 1.6], dtype=np.float64),
        gestures=(),
    )
    labels = np.zeros((3, 21, 3), dtype=np.float32)
    labels[1, 1, 0] = 1.0
    labels[2, 1, 0] = 2.0

    examples = list(
        PoseWindows.from_session(
            session, labels, descriptor=_SIXTEEN_FRAME_WINDOW, stride=32
        )
    )

    assert examples
    assert examples[0].features.shape == _SIXTEEN_FRAME_WINDOW.window_shape
    assert examples[0].target.shape == (21, 3)
    assert examples[0].participant == "study1/sub1"
    assert examples[0].device == "unknown-device"
    assert examples[0].timestamp_s < 0.8
    assert examples[-1].target[1, 0] in (1.0, 2.0)
