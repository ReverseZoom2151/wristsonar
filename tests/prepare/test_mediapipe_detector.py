from __future__ import annotations

import pytest

from wristsonar.prepare.landmarks import LandmarkPreparationError, MediaPipeHandDetector


def test_mediapipe_detector_explains_the_optional_dependency(
    without_mediapipe: None,
) -> None:
    with pytest.raises(LandmarkPreparationError, match=r"\[prepare\]"):
        MediaPipeHandDetector()


def test_mediapipe_detector_rejects_multi_hand_configuration_before_import() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MediaPipeHandDetector(max_hands=2)
