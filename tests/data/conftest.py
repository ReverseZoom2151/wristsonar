"""Synthetic WatchHand trees, so the suite never needs the 175 GB release.

Everything here mirrors the published layout exactly: the same directory name
grammar, the same config.json keys, the same `label,start,end,name` records
format, the same (600, n_frames) float32 profiles. The only difference is that
sessions are a few seconds instead of six minutes.

Building the fixture rather than checking in a sample is a deliberate choice.
A checked-in sample would be a redistribution of participant data, and it
would drift silently from the schema this module claims to parse. Generating
it means the schema is written down in exactly one place, here, where a
mismatch with the real dataset shows up as a failing parse rather than a
passing test against a stale copy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

SAMPLE_RATE = 48_000
FRAME_LENGTH = 600
N_RANGE_BINS = 600
GT_FPS = 30.0

_BASE_TIMESTAMP = 1_742_260_996.362494
"""Taken from a real session so the magnitudes exercise float64 timestamp
handling the way the dataset does. Nothing depends on the exact value."""


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """One synthetic session's shape."""

    stem: str
    n_frames: int
    audio_sync_index: int
    gt_sync_offset_s: float
    record_name: str


def _profile(n_bins: int, n_frames: int, seed: int) -> NDArray[np.float32]:
    """A profile with a strong static near-field peak and a moving reflector.

    Shaped rather than pure noise because two tests depend on structure:
    bin-zero estimation needs a dominant static peak, and normalisation needs
    a peak that is not one already.
    """
    rng = np.random.default_rng(seed)
    array = rng.normal(0.0, 0.05, size=(n_bins, n_frames))
    array[3, :] += 40.0
    moving = 12 + (6 * np.sin(np.linspace(0.0, 6.0, n_frames))).astype(int)
    array[moving, np.arange(n_frames)] += 8.0
    return np.asarray(array, dtype=np.float32)


def _write_records(path: Path, start: float, n_gestures: int) -> None:
    names = ("1", "2", "Five", "wrist_up")
    lines: list[str] = []
    cursor = start
    for index in range(n_gestures):
        end = cursor + 2.0
        lines.append(f"{index},{cursor:.6f},{end:.6f},{names[index % len(names)]}")
        cursor = end
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_frame_times(path: Path, start: float, n_video_frames: int) -> None:
    stamps = start + np.arange(n_video_frames) / GT_FPS
    path.write_text(
        "\n".join(f"{value:.6f}" for value in stamps) + "\n", encoding="utf-8"
    )


def write_session_dir(directory: Path, specs: list[SessionSpec]) -> None:
    """Write one folder holding several sessions, matching the real layout."""
    directory.mkdir(parents=True, exist_ok=True)

    audio_files: list[str] = []
    audio_sync: list[int] = []
    gt_files: list[str] = []
    gt_sync: list[float] = []
    videos: list[str] = []

    for index, spec in enumerate(specs):
        original = _profile(N_RANGE_BINS, spec.n_frames, seed=index)
        np.save(directory / f"{spec.stem}_fmcw_16bit_profiles.npy", original)
        # The real release ships a differential that is shorter than the
        # original, so the fixture is too. Code that assumes equal widths
        # should fail here rather than on the real data.
        diff = np.diff(np.abs(original), axis=1).astype(np.float32)
        np.save(directory / f"{spec.stem}_fmcw_16bit_diff_profiles.npy", diff)

        start = _BASE_TIMESTAMP + index * 300.0
        _write_records(directory / f"{spec.record_name}_records.txt", start, 4)
        _write_frame_times(directory / f"{spec.record_name}_frame_time.txt", start, 90)

        audio_files.append(f"{spec.stem}.raw")
        audio_sync.append(spec.audio_sync_index)
        gt_files.append(f"{spec.record_name}_records.txt")
        gt_sync.append(start + spec.gt_sync_offset_s)
        videos.append(f"{spec.record_name}.mp4")

    config: dict[str, Any] = {
        "tasks": ["classification", "hand_landmarks"],
        "audio": {
            "config": {
                "sampling_rate": SAMPLE_RATE,
                "n_channels": 1,
                "channels_of_interest": [],
                "signal": "FMCW",
                "tx_file": ["fmcw18000_b3000_l600_s48k.wav"],
                "frame_length": FRAME_LENGTH,
                "sample_depth": 16,
                "bandpass_range": [[18000, 21000]],
            },
            "files": audio_files,
            "syncing_poses": audio_sync,
        },
        "ground_truth": {
            "files": gt_files,
            "syncing_poses": gt_sync,
            "anchor_length": 88,
            "videos": videos,
        },
        "sessions": [[{"start": _BASE_TIMESTAMP, "duration": 20.0}] for _ in specs],
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def build_tree(root: Path, *, n_frames: int = 200) -> Path:
    """A miniature dataset: two study 1 folders and one nested study 4 folder."""
    specs = [
        SessionSpec("audio001", n_frames, 310, 0.5, "record_20250318_102306_311752"),
        SessionSpec("audio002", n_frames, 247, 0.4, "record_20250318_102707_892381"),
    ]
    write_session_dir(root / "Study-1" / "sub1_samsung_left_video", specs)
    write_session_dir(root / "Study-1" / "sub2_xiaomi_right_video", specs[:1])
    write_session_dir(
        root / "Study-4" / "sub1_raw" / "sub1_normal",
        [SessionSpec("audio001", n_frames, 200, 0.3, "record_20250401_090000_000001")],
    )
    # Incidental files the real tree carries and the manifest must ignore.
    (root / "Study-1" / ".DS_Store").write_bytes(b"\x00" * 16)
    (
        root
        / "Study-1"
        / "sub1_samsung_left_video"
        / "audio001_fmcw_16bit_profiles.png"
    ).write_bytes(b"\x89PNG\r\n\x1a\n")
    return root


@pytest.fixture
def watchhand_root(tmp_path: Path) -> Path:
    """A synthetic dataset root, fresh per test."""
    return build_tree(tmp_path / "watchhand")


@pytest.fixture
def make_watchhand_tree() -> Callable[..., Path]:
    """Factory for tests that need a tree of a different size.

    Handed out as a fixture rather than imported, so that tests/ needs no
    __init__.py and conftest stays the only place that knows the schema.
    """
    return build_tree


@pytest.fixture
def add_session_dir() -> Callable[[Path, list[SessionSpec]], None]:
    """Factory for adding a session folder after a manifest has been built."""
    return write_session_dir


@pytest.fixture
def session_spec() -> type[SessionSpec]:
    return SessionSpec
