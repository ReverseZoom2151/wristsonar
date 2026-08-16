from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from wristsonar.prepare.landmarks import LandmarkPreparationError, prepare_landmarks


class NoopDetector:
    version = "test-detector/1"

    def detect(self, _frame: NDArray[np.uint8]) -> None:
        return None


def test_landmark_preparation_refuses_to_fill_missing_video_labels(
    tmp_path: Path,
) -> None:
    with pytest.raises(LandmarkPreparationError, match="preparation support"):
        prepare_landmarks(
            tmp_path / "missing.mp4",
            np.asarray([0.0], dtype=np.float64),
            tmp_path / "landmarks.npy",
            NoopDetector(),
        )
