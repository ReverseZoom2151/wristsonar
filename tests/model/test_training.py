from __future__ import annotations

import numpy as np
import pytest

from wristsonar.model.torch_model import ModelUnavailableError
from wristsonar.model.training import TrainingConfig, TrainingError, train_pose_cnn


def test_training_config_rejects_invalid_optimization_parameters() -> None:
    with pytest.raises(TrainingError, match="positive"):
        TrainingConfig(epochs=0)
    with pytest.raises(TrainingError, match="learning_rate"):
        TrainingConfig(learning_rate=0)


def test_training_rejects_mismatched_supervision_before_torch_is_needed() -> None:
    x = np.zeros((2, 2, 60, 96), dtype=np.float32)
    wrong_y = np.zeros((2, 20, 3), dtype=np.float32)

    with pytest.raises(TrainingError, match="targets must"):
        train_pose_cnn(x, wrong_y, x, wrong_y)


def test_training_explains_the_optional_torch_dependency(
    without_torch: None,
) -> None:
    x = np.zeros((2, 2, 3, 4), dtype=np.float32)
    y = np.zeros((2, 21, 3), dtype=np.float32)

    with pytest.raises(ModelUnavailableError, match=r"\[train\]"):
        train_pose_cnn(x, y, x, y, config=TrainingConfig(epochs=1))
