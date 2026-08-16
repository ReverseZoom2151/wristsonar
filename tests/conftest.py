"""Fixtures shared across the suite.

The optional extras present a testing trap. Several modules promise a helpful
error when `[train]` or `[prepare]` was never installed, and the obvious way to
test that promise is to assert the error is raised. That works only on a
machine where the package happens to be absent, so the test passes in CI and
fails for any contributor who has Torch or OpenCV installed for another
project. The suite is supposed to pass everywhere with nothing downloaded, and
a test whose result depends on unrelated site-packages breaks that guarantee.

These fixtures simulate the absence instead of relying on it. Binding a name to
None in `sys.modules` makes a subsequent `import` of it raise ImportError,
which is exactly the condition the code under test is meant to handle.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest


def _hide(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)
        # Submodules resolve independently, so a stale entry for one of them
        # would let a masked import succeed through the back door.
        for loaded in list(sys.modules):
            if loaded.startswith(f"{name}."):
                monkeypatch.setitem(sys.modules, loaded, None)


@pytest.fixture
def without_torch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run a test as though `pip install -e '.[train]'` had never happened."""
    _hide(monkeypatch, "torch")
    yield


@pytest.fixture
def without_opencv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run a test as though `pip install -e '.[prepare]'` had never happened."""
    _hide(monkeypatch, "cv2")
    yield


@pytest.fixture
def without_mediapipe(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """As above, for the other half of the `prepare` extra.

    Separate from `without_opencv` because the two are installed together but
    fail independently, and a test that means to exercise one should not pass
    because the other happens to be missing.
    """
    _hide(monkeypatch, "mediapipe")
    yield
