"""A compact causal CNN for the 2 by 60 by 96 WatchHand input."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from wristsonar.types import N_JOINTS

__all__ = ["ModelUnavailableError", "create_pose_cnn", "load_pose_cnn"]


class ModelUnavailableError(RuntimeError):
    """Raised when training support was not installed explicitly."""


def create_pose_cnn(
    *, channels: int = 2, width: int = 32, seed: int | None = None
) -> Any:
    """Build the reproducible baseline model.

    This is intentionally a small convolutional baseline, not a claim to have
    reproduced FastViT-T12.  It provides a runnable, causal reference that the
    public-data reproduction must beat before architecture changes are useful.

    ``seed`` seeds the initialisation and nothing else.  It defaults to None,
    meaning this function leaves the global Torch generator exactly as it found
    it.  An earlier version called ``torch.manual_seed(0)`` unconditionally
    here, which silently overrode the caller's seed a line after training set
    it, so every run initialised identically and a seed sweep measured only
    batch order.  It also reset the caller's generator on the inference path,
    since checkpoint loading builds a model too.
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

    if seed is None:
        return PoseCNN()
    # Build under a private generator state so the caller's stream is restored.
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        return PoseCNN()
    finally:
        torch.set_rng_state(state)


def load_pose_cnn(weights: Path, *, width: int = 32) -> Any:
    """Load CPU-portable state written by ``export_torch_checkpoint``.

    Weight integrity belongs to :class:`CheckpointBundle`, not here.  This
    function owns the narrower Torch responsibility: construct the known
    architecture, reject arbitrary pickle objects through ``weights_only``,
    and reject missing or incompatible parameter tensors loudly.
    """
    try:
        import torch
    except ImportError as error:
        raise ModelUnavailableError(
            "install training support with: pip install -e '.[train]'"
        ) from error
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights do not exist: {weights}")
    model = create_pose_cnn(width=width)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        # A truncated or non-Torch file surfaces from the unpickler rather than
        # from Torch, so without these the caller gets a bare pickle traceback
        # that never names the offending path. Reporting a corrupt checkpoint
        # loudly is exactly what this function promises to do.
        pickle.UnpicklingError,
        EOFError,
    ) as error:
        raise ValueError(f"invalid pose-cnn weights {weights}: {error}") from error
    model.eval()
    return model
