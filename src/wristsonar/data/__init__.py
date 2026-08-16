"""Dataset access, with integrity checking that cannot be switched off.

Loading is deliberately awkward without a manifest. Every public entry point
here takes one, because an evaluation run against unpinned bytes produces a
number that nobody can later attach to a dataset.
"""

from __future__ import annotations

from wristsonar.data.manifest import (
    MANIFEST_SCHEMA,
    FileRecord,
    IntegrityError,
    LazyVerifier,
    Manifest,
    VerificationReport,
    build_manifest,
    sha256_file,
)
from wristsonar.data.watchhand import (
    BUTTERWORTH_ORDER,
    DATASET_NAME,
    MODEL_CROP_BINS,
    MODEL_WINDOW_FRAMES,
    N_RANGE_BINS,
    PROFILE_BIN_METRES,
    WATCHHAND_CHIRP,
    WATCHHAND_GROUND_TRUTH,
    Device,
    GestureLabel,
    Hand,
    MissingLandmarksError,
    SessionData,
    SessionRef,
    Study,
    WatchHandDataset,
    WatchHandError,
    build_watchhand_manifest,
    estimate_bin_zero_offset,
    watchhand_protocol,
)

__all__ = [
    "BUTTERWORTH_ORDER",
    "DATASET_NAME",
    "MANIFEST_SCHEMA",
    "MODEL_CROP_BINS",
    "MODEL_WINDOW_FRAMES",
    "N_RANGE_BINS",
    "PROFILE_BIN_METRES",
    "WATCHHAND_CHIRP",
    "WATCHHAND_GROUND_TRUTH",
    "Device",
    "FileRecord",
    "GestureLabel",
    "Hand",
    "IntegrityError",
    "LazyVerifier",
    "Manifest",
    "MissingLandmarksError",
    "SessionData",
    "SessionRef",
    "Study",
    "VerificationReport",
    "WatchHandDataset",
    "WatchHandError",
    "build_manifest",
    "build_watchhand_manifest",
    "estimate_bin_zero_offset",
    "sha256_file",
    "watchhand_protocol",
]
