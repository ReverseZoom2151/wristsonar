"""A compact causal CNN for the 2 by 60 by 96 WatchHand input."""

from __future__ import annotations

from typing import Any

from wristsonar.types import N_JOINTS

__all__ = ["ModelUnavailableError", "create_pose_cnn"]


class ModelUnavailableError(RuntimeError):
    """Raised when training support was not installed explicitly."""


def create_pose_cnn(*, channels: int = 2, width: int = 32) -> Any:
    """Build the reproducible baseline model.

    This is intentionally a small convolutional baseline, not a claim to have
    reproduced FastViT-T12.  It provides a runnable, causal reference that the
    public-data reproduction must beat before architecture changes are useful.
    """
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ModelUnavailableError(
            "install training support with: pip install -e '.[train]'"
        ) from error

    class PoseCNN(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(channels, width, kernel_size=5, padding=2),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(width * 2),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width * 2, width * 2),
                nn.GELU(),
                nn.Linear(width * 2, N_JOINTS * 3),
            )

        def forward(self, x: Any) -> Any:
            if x.ndim != 4 or x.shape[1] != channels:
                raise ValueError(
                    f"expected (batch, {channels}, bins, frames), got {tuple(x.shape)}"
                )
            return self.head(self.features(x)).reshape(-1, N_JOINTS, 3)

    torch.manual_seed(0)
    return PoseCNN()
