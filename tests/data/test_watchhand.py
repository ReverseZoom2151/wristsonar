"""Loader tests, run entirely against synthetic trees.

Two things are being defended here. First, that the loader parses the real
layout, which the fixture reproduces key for key. Second, and more important,
that the constants describing the input representation match what the dataset
and paper specify, because those constants are the contract a live capture
app has to satisfy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wristsonar.data.manifest import IntegrityError, Manifest, build_manifest
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
    Hand,
    MissingLandmarksError,
    SessionRef,
    Study,
    WatchHandDataset,
    WatchHandError,
    build_watchhand_manifest,
    estimate_bin_zero_offset,
    watchhand_protocol,
)
from wristsonar.protocol import GroundTruth, Split
from wristsonar.types import N_JOINTS, SPEED_OF_SOUND, EchoProfile, HandPose


def _dataset(root: Path) -> WatchHandDataset:
    return WatchHandDataset(root, build_watchhand_manifest(root, version="test-1"))


# --------------------------------------------------------------------------
# The physical constants. If any of these drift, training and capture diverge.
# --------------------------------------------------------------------------


def test_chirp_matches_the_shipped_transmit_waveform() -> None:
    """config.json names `fmcw18000_b3000_l600_s48k.wav`. Read it literally."""
    assert WATCHHAND_CHIRP.f_start == 18_000.0
    assert WATCHHAND_CHIRP.bandwidth == 3_000.0
    assert WATCHHAND_CHIRP.n_samples == 600
    assert WATCHHAND_CHIRP.sample_rate == 48_000
    assert WATCHHAND_CHIRP.frame_rate == pytest.approx(80.0)


def test_bin_metres_reproduces_the_published_3_57_mm() -> None:
    assert pytest.approx(SPEED_OF_SOUND / 96_000) == PROFILE_BIN_METRES
    assert pytest.approx(3.57, abs=0.005) == PROFILE_BIN_METRES * 1000


def test_crop_window_covers_a_hand_and_not_the_room() -> None:
    assert pytest.approx(0.214, abs=0.001) == MODEL_CROP_BINS * PROFILE_BIN_METRES
    assert MODEL_WINDOW_FRAMES / WATCHHAND_CHIRP.frame_rate == pytest.approx(1.2)


def test_range_resolution_is_far_coarser_than_the_reported_error() -> None:
    """A guard against the field's central confusion, kept executable.

    c/2B at 3 kHz is about 5.7 cm. The paper reports 7.87 mm. The gap is a
    learned prior over hand shape, not measurement, and any code that starts
    treating a bin as a measurement of geometry should trip this.
    """
    assert WATCHHAND_CHIRP.range_resolution > 0.05
    assert WATCHHAND_CHIRP.range_resolution / 10 > PROFILE_BIN_METRES


def test_ground_truth_is_video_fitted_and_not_configurable() -> None:
    assert WATCHHAND_GROUND_TRUTH is GroundTruth.VIDEO_FITTED
    protocol = watchhand_protocol(Split.CROSS_USER, subjects=24, version="v1")
    assert protocol.ground_truth is GroundTruth.VIDEO_FITTED
    assert protocol.dataset == DATASET_NAME
    assert protocol.is_zero_shot
    assert BUTTERWORTH_ORDER == 5


def test_protocol_carries_calibration_and_split_through() -> None:
    protocol = watchhand_protocol(
        Split.CROSS_SESSION, subjects=24, version="v1", calibration_minutes=2.0
    )
    assert not protocol.is_zero_shot
    assert protocol.split.is_honest
    assert "cross-session" in protocol.describe()


# --------------------------------------------------------------------------
# Layout parsing.
# --------------------------------------------------------------------------


def test_sessions_are_discovered_with_full_identity(watchhand_root: Path) -> None:
    refs = _dataset(watchhand_root).sessions()

    assert len(refs) == 4
    assert [r.identity for r in refs] == sorted(r.identity for r in refs)

    first = refs[0]
    assert first.study is Study.MAIN
    assert first.participant == 1
    assert first.device is Device.SAMSUNG
    assert first.hand is Hand.LEFT
    assert first.session_index == 0
    assert first.profile_path.endswith("audio001_fmcw_16bit_profiles.npy")
    assert "study1/p01/samsung/left" in first.identity


def test_study_four_nesting_is_traversed(watchhand_root: Path) -> None:
    refs = _dataset(watchhand_root).sessions(studies=[Study.DYNAMIC])
    assert len(refs) == 1
    assert refs[0].condition == "normal"
    assert refs[0].device is None, "study 4 folder names do not record the watch"
    assert "unknown-device" in refs[0].identity


def test_session_filtering_by_study(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    assert len(dataset.sessions(studies=[Study.MAIN])) == 3
    assert dataset.sessions(studies=[Study.NOISE]) == ()


def test_session_index_is_derived_from_the_filename_not_the_config(
    watchhand_root: Path,
) -> None:
    """config['audio']['files'] holds `audioNNN.raw`, and no .raw was released.

    The published loading example reads the profile name from that list, which
    cannot work. Indices come from the filenames on disk instead.
    """
    refs = _dataset(watchhand_root).sessions(studies=[Study.MAIN])
    stems = {r.audio_stem: r.session_index for r in refs}
    assert stems["audio001"] == 0
    assert stems["audio002"] == 1


# --------------------------------------------------------------------------
# Loading and synchronisation.
# --------------------------------------------------------------------------


def test_load_returns_correctly_shaped_arrays(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])

    assert session.profiles.shape[0] == N_RANGE_BINS
    assert session.profiles.dtype == np.float32
    assert session.diff_profiles.shape[1] < session.profiles.shape[1]
    assert session.frame_rate == pytest.approx(80.0)
    assert len(session.gestures) == 4
    assert session.frame_timestamps.shape == (90,)


def test_sync_formula_round_trips(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])

    for index in (0, 7, 123):
        assert session.profile_index_for(session.timestamp_for(index)) == index


def test_sync_matches_the_published_formula(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    timestamp = float(session.frame_timestamps[10])

    expected = round(
        (
            (timestamp - session.gt_sync_timestamp) * session.sample_rate
            + session.audio_sync_index
        )
        / session.frame_length
    )
    assert session.profile_index_for(timestamp) == expected


def test_gesture_lookup_by_timestamp(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    label = session.gestures[1]

    assert session.gesture_at(label.start_s) is label
    assert session.gesture_at(label.duration_s * 0.5 + label.start_s) is label
    assert session.gesture_at(session.gestures[-1].end_s + 1e6) is None


# --------------------------------------------------------------------------
# The project's own types come out the other end.
# --------------------------------------------------------------------------


def test_frames_come_out_as_echo_profiles(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    profile = session.echo_profile(5)

    assert isinstance(profile, EchoProfile)
    assert profile.n_bins == MODEL_CROP_BINS
    assert profile.bin_metres == PROFILE_BIN_METRES
    assert profile.max_range == pytest.approx(0.214, abs=0.001)
    assert not profile.differential
    assert float(np.max(np.abs(profile.samples))) == pytest.approx(1.0)


def test_uncropped_and_unnormalised_frames_are_available(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    profile = session.echo_profile(5, crop_bins=None, normalise=False)

    assert profile.n_bins == N_RANGE_BINS
    assert float(np.max(np.abs(profile.samples))) > 1.0


def test_differential_frames_are_flagged_as_such(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    profile = session.echo_profile(0, differential=True)
    assert profile.differential


def test_frame_index_out_of_range_raises(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    with pytest.raises(IndexError, match="out of range"):
        session.echo_profile(session.n_frames + 1)


def test_windows_are_two_channel_model_inputs(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    starts, arrays = zip(*session.windows(stride=16), strict=True)

    assert arrays[0].shape == (2, MODEL_CROP_BINS, MODEL_WINDOW_FRAMES)
    assert arrays[0].dtype == np.float32
    assert starts[0] == 0 and starts[1] == 16
    assert float(np.max(np.abs(arrays[0]))) == pytest.approx(1.0)


def test_windows_are_empty_when_the_session_is_too_short(
    tmp_path: Path, make_watchhand_tree: Callable[..., Path]
) -> None:
    root = make_watchhand_tree(tmp_path / "tiny", n_frames=32)
    dataset = _dataset(root)
    session = dataset.load(dataset.sessions()[0])
    assert list(session.windows()) == []


def test_frames_iterates_every_column(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    assert sum(1 for _ in session.frames()) == session.n_frames


# --------------------------------------------------------------------------
# The ground truth that is not in the release.
# --------------------------------------------------------------------------


def test_missing_landmarks_raise_a_specific_and_actionable_error(
    watchhand_root: Path,
) -> None:
    dataset = _dataset(watchhand_root)
    ref = dataset.sessions()[0]
    session = dataset.load(ref)

    assert dataset.landmarks(ref) is None
    with pytest.raises(MissingLandmarksError, match="MediaPipe"):
        session.hand_poses(dataset.landmarks(ref))


def test_generated_landmarks_become_wrist_relative_hand_poses(
    watchhand_root: Path,
) -> None:
    dataset = _dataset(watchhand_root)
    ref = dataset.sessions()[0]
    session = dataset.load(ref)

    rng = np.random.default_rng(0)
    landmarks = rng.normal(0.0, 0.05, size=(4, N_JOINTS, 3)).astype(np.float32)
    poses = session.hand_poses(landmarks, handedness="left")

    assert len(poses) == 4
    assert all(isinstance(p, HandPose) for p in poses)
    assert poses[0].handedness == "left"
    assert poses[0].wrist_relative
    assert np.allclose(poses[0].joint("wrist"), 0.0), "wrist must be the origin"
    np.testing.assert_allclose(
        poses[0].joint("index_tip"),
        landmarks[0, 8] - landmarks[0, 0],
        atol=1e-6,
    )


def test_wrongly_shaped_landmarks_are_refused(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    with pytest.raises(WatchHandError, match="shape"):
        session.hand_poses(np.zeros((4, 20, 3), dtype=np.float32))


def test_landmark_sidecar_is_read_when_present(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    ref = dataset.sessions()[0]
    np.save(
        watchhand_root / ref.landmarks_path,
        np.zeros((3, N_JOINTS, 3), dtype=np.float32),
    )
    loaded = dataset.landmarks(ref)
    assert loaded is not None and loaded.shape == (3, N_JOINTS, 3)


# --------------------------------------------------------------------------
# Integrity gating. The loader must refuse altered data.
# --------------------------------------------------------------------------


def test_tampered_profile_refuses_to_load(watchhand_root: Path) -> None:
    manifest = build_watchhand_manifest(watchhand_root, version="test-1")
    dataset = WatchHandDataset(watchhand_root, manifest)
    ref = dataset.sessions()[0]

    payload = bytearray((watchhand_root / ref.profile_path).read_bytes())
    payload[-1] ^= 0xFF
    (watchhand_root / ref.profile_path).write_bytes(bytes(payload))

    with pytest.raises(IntegrityError, match="contents have changed"):
        dataset.load(ref)


def test_session_added_after_the_manifest_refuses_to_load(
    watchhand_root: Path,
    add_session_dir: Callable[[Path, list[Any]], None],
    session_spec: Callable[..., Any],
) -> None:
    """The silent-extra-data case: enumeration finds it, verification rejects it."""
    manifest = build_watchhand_manifest(watchhand_root, version="test-1")
    add_session_dir(
        watchhand_root / "Study-1" / "sub3_google_left_video",
        [session_spec("audio001", 128, 1, 0.1, "record_20250401_000000_000001")],
    )
    dataset = WatchHandDataset(watchhand_root, manifest)
    new = next(r for r in dataset.sessions() if r.participant == 3)

    with pytest.raises(IntegrityError, match="not in manifest"):
        dataset.load(new)


def test_manifest_for_another_dataset_is_refused(watchhand_root: Path) -> None:
    other = build_manifest(watchhand_root, dataset="grab-n-go", version="v1")
    with pytest.raises(IntegrityError, match="refusing to cross-load"):
        WatchHandDataset(watchhand_root, other)


def test_missing_root_is_refused(tmp_path: Path, watchhand_root: Path) -> None:
    manifest = build_watchhand_manifest(watchhand_root, version="test-1")
    with pytest.raises(WatchHandError, match="does not exist"):
        WatchHandDataset(tmp_path / "absent", manifest)


def test_manifest_records_session_identities(watchhand_root: Path) -> None:
    manifest = build_watchhand_manifest(watchhand_root, version="2026-08")
    assert len(manifest.identities) == 4
    assert any("samsung" in identity for identity in manifest.identities)
    assert manifest.version == "2026-08"


def test_manifest_excludes_previews_and_videos(watchhand_root: Path) -> None:
    manifest = build_watchhand_manifest(watchhand_root, version="v1")
    assert not any(r.path.endswith((".png", ".mp4")) for r in manifest.files)


def test_wrong_frame_length_in_config_is_rejected(watchhand_root: Path) -> None:
    """A config that does not describe this chirp must not load as if it did."""
    import json

    manifest_before = build_watchhand_manifest(watchhand_root, version="v1")
    ref = WatchHandDataset(watchhand_root, manifest_before).sessions()[0]
    config_path = watchhand_root / ref.directory / "config.json"
    blob = json.loads(config_path.read_text(encoding="utf-8"))
    blob["audio"]["config"]["frame_length"] = 512
    config_path.write_text(json.dumps(blob), encoding="utf-8")

    dataset = WatchHandDataset(
        watchhand_root, build_watchhand_manifest(watchhand_root, version="v2")
    )
    with pytest.raises(WatchHandError, match="frame_length"):
        dataset.load(ref)


# --------------------------------------------------------------------------
# Recovering the one preprocessing parameter nobody wrote down.
# --------------------------------------------------------------------------


def test_bin_zero_offset_is_measured_from_the_static_direct_path() -> None:
    """The fixture plants the direct path at bin 3. Find it without being told."""
    rng = np.random.default_rng(7)
    profiles = rng.normal(0.0, 0.05, size=(600, 200)).astype(np.float32)
    profiles[3, :] += 40.0
    assert estimate_bin_zero_offset(profiles) == 3


def test_bin_zero_offset_on_real_shaped_fixture(watchhand_root: Path) -> None:
    dataset = _dataset(watchhand_root)
    session = dataset.load(dataset.sessions()[0])
    assert estimate_bin_zero_offset(np.asarray(session.profiles)) == 3


def test_bin_zero_offset_rejects_a_single_frame() -> None:
    with pytest.raises(ValueError, match="bins, frames"):
        estimate_bin_zero_offset(np.zeros(600, dtype=np.float32))


# --------------------------------------------------------------------------
# Small parsing details that bite once and are then forgotten.
# --------------------------------------------------------------------------


def test_session_ref_rejects_impossible_identities() -> None:
    from pathlib import PurePosixPath

    with pytest.raises(ValueError, match="one based"):
        SessionRef(
            study=Study.MAIN,
            participant=0,
            session_index=0,
            directory=PurePosixPath("Study-1/x"),
            audio_stem="audio001",
        )


def test_manifest_survives_a_reread_of_the_written_file(
    watchhand_root: Path, tmp_path: Path
) -> None:
    manifest = build_watchhand_manifest(watchhand_root, version="v1")
    path = tmp_path / "watchhand.manifest.json"
    manifest.write(path)
    dataset = WatchHandDataset(watchhand_root, Manifest.read(path))
    assert dataset.load(dataset.sessions()[0]).n_frames > 0
    assert dataset.version == "v1"
