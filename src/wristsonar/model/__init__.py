"""Trainable pose regression over verified acoustic windows.

The optional Torch dependency is deliberately kept out of the import path for
the rest of Wristsonar.  Dataset inspection and evaluation must work on a
clean checkout, while training opts in through ``wristsonar[train]``.
"""

from wristsonar.model.checkpoint import CheckpointBundle, CheckpointMetadata
from wristsonar.model.corpus import SessionWindowSource, TrainingCorpus
from wristsonar.model.dataset import PoseWindows, WindowExample
from wristsonar.model.load import (
    CheckpointLoadError,
    LoadedPoseCheckpoint,
    load_torch_checkpoint,
    pose_cnn_width,
)
from wristsonar.model.normalization import (
    FeatureNormalizer,
    NormalizationError,
    PoseNormalizer,
)
from wristsonar.model.predictor import NormalizedPosePredictor, PredictorError
from wristsonar.model.torch_model import ModelUnavailableError, create_pose_cnn
from wristsonar.model.training import (
    TrainedPoseModel,
    TrainingConfig,
    TrainingError,
    TrainingHistory,
    export_torch_checkpoint,
    train_pose_cnn,
)

__all__ = [
    "CheckpointBundle",
    "CheckpointLoadError",
    "CheckpointMetadata",
    "FeatureNormalizer",
    "LoadedPoseCheckpoint",
    "ModelUnavailableError",
    "NormalizationError",
    "NormalizedPosePredictor",
    "PoseNormalizer",
    "PoseWindows",
    "PredictorError",
    "SessionWindowSource",
    "TrainedPoseModel",
    "TrainingConfig",
    "TrainingCorpus",
    "TrainingError",
    "TrainingHistory",
    "WindowExample",
    "create_pose_cnn",
    "export_torch_checkpoint",
    "load_torch_checkpoint",
    "pose_cnn_width",
    "train_pose_cnn",
]
