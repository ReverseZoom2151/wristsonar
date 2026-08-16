"""Trainable pose regression over verified acoustic windows.

The optional Torch dependency is deliberately kept out of the import path for
the rest of Wristsonar.  Dataset inspection and evaluation must work on a
clean checkout, while training opts in through ``wristsonar[train]``.
"""

from wristsonar.model.checkpoint import CheckpointBundle, CheckpointMetadata
from wristsonar.model.dataset import PoseWindows, WindowExample
from wristsonar.model.normalization import (
    FeatureNormalizer,
    NormalizationError,
    PoseNormalizer,
)
from wristsonar.model.predictor import NormalizedPosePredictor, PredictorError
from wristsonar.model.torch_model import ModelUnavailableError, create_pose_cnn

__all__ = [
    "CheckpointBundle",
    "CheckpointMetadata",
    "FeatureNormalizer",
    "ModelUnavailableError",
    "NormalizationError",
    "NormalizedPosePredictor",
    "PoseNormalizer",
    "PoseWindows",
    "PredictorError",
    "WindowExample",
    "create_pose_cnn",
]
