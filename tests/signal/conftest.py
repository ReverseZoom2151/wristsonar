"""Shared fixtures for the signal layer, plus a source path shim.

The package is a src layout and is not necessarily installed when these tests
run. Putting `src` on the path here rather than requiring an editable install
keeps the instruction for running this layer to a single pytest command, which
is the point of a layer that has no device and no dataset behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import settings
from numpy.typing import NDArray

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wristsonar.signal import (  # noqa: E402
    analytic_reference,
    bin_metres_for,
    matched_filter,
)
from wristsonar.types import ChirpConfig  # noqa: E402

settings.register_profile("signal", deadline=None, max_examples=40)
settings.load_profile("signal")


@pytest.fixture(scope="session")
def config() -> ChirpConfig:
    """The default sweep. Every physical constant in these tests derives from it."""
    return ChirpConfig()


@pytest.fixture(scope="session")
def reference(config: ChirpConfig) -> NDArray[np.complex128]:
    return analytic_reference(config)


@pytest.fixture(scope="session")
def bin_metres(config: ChirpConfig) -> float:
    return bin_metres_for(config.sample_rate)


def profile_of(
    frame: NDArray[np.float64], reference: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """One-liner used by most tests: frame in, range envelope out."""
    return matched_filter(frame, reference)
