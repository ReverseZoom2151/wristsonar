"""Reader for the WatchHand dataset (CHI 2026, KAIST WIT Lab with Cornell).

The single fact that shapes this module: WatchHand ships no raw audio. Every
session is a precomputed C-FMCW echo profile, float32, shape (600, n_frames).
That would normally be fatal for a project whose other half is a live capture
app, because a model trained on someone else's preprocessing cannot be fed by
a microphone unless that preprocessing is reproducible.

It is reproducible here, and the parameters are not guessed. `config.json`
names the exact transmit waveform, `fmcw18000_b3000_l600_s48k.wav`, which
pins a linear 18 to 21 kHz sweep of 600 samples at 48 kHz. The paper pins the
rest: a fifth-order Butterworth bandpass over the same band, cross-correlation
against one transmitted sweep, one range bin per sample so that a bin is
c / (2 * fs) = 3.573 mm, and the differential profile as the frame-to-frame
difference of correlation magnitude. `wristsonar.dsp` is expected to
implement exactly that, and the constants it must match live here rather than
being retyped at each call site.

Two things remain outside our control and are named as such rather than
papered over:

* The absolute amplitude scale of the shipped profiles depends on how the
  16-bit PCM input was scaled before correlation, which is not documented.
  Any per-profile normalisation cancels it, which is why `normalise` exists
  and defaults to on.
* The lag that corresponds to bin zero is not documented either. It is
  recoverable from the data, because the direct speaker-to-microphone path is
  a large static peak at essentially zero range, so `estimate_bin_zero_offset`
  measures it instead of assuming it.

The other surprise worth reading before building on this: the pose ground
truth is not in the release. Sessions carry the study video, per-frame video
timestamps, and gesture class labels. The 21 MediaPipe landmarks the paper
regresses were produced by running MediaPipe over those videos and were not
published. `SessionData.hand_poses` therefore reads a landmark sidecar that
you generate, and raises a specific error when it is absent, instead of
returning something that looks like ground truth and is not.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from wristsonar.data.manifest import IntegrityError, LazyVerifier, Manifest
from wristsonar.protocol import GroundTruth, Protocol, Split
from wristsonar.types import (
    N_JOINTS,
    SPEED_OF_SOUND,
    ChirpConfig,
    EchoProfile,
    HandPose,
)

__all__ = [
    "BUTTERWORTH_ORDER",
    "DATASET_NAME",
    "MODEL_CROP_BINS",
    "MODEL_WINDOW_FRAMES",
    "N_RANGE_BINS",
    "PROFILE_BIN_METRES",
    "WATCHHAND_CHIRP",
    "WATCHHAND_GROUND_TRUTH",
    "Device",
    "GestureLabel",
    "Hand",
    "MissingLandmarksError",
    "SessionData",
    "SessionRef",
    "Study",
    "WatchHandDataset",
    "WatchHandError",
    "estimate_bin_zero_offset",
    "watchhand_protocol",
]

DATASET_NAME: Final = "watchhand"

WATCHHAND_CHIRP: Final = ChirpConfig(
    f_start=18_000.0,
    f_end=21_000.0,
    duration_s=600 / 48_000,
    sample_rate=48_000,
)
"""The sweep the watches actually transmitted.

Read off the transmit waveform filename in every config.json,
`fmcw18000_b3000_l600_s48k.wav`: start 18000 Hz, bandwidth 3000 Hz, length
600 samples, 48 kHz. A live capture app that emits anything else is not
producing this dataset's input, however similar the numbers look.
"""

N_RANGE_BINS: Final = 600
"""Rows in a shipped profile. One bin per sample of the correlation lag."""

PROFILE_BIN_METRES: Final = SPEED_OF_SOUND / (2.0 * WATCHHAND_CHIRP.sample_rate)
"""3.5729 mm. One sample of round-trip delay, halved into one-way distance.

The paper quotes 3.57 mm, which is this number, so the range axis needs no
calibration constant beyond the speed of sound.
"""

BUTTERWORTH_ORDER: Final = 5
"""Order of the bandpass applied before correlation, from the paper."""

MODEL_CROP_BINS: Final = 60
"""Bins the paper keeps, about 21.4 cm, covering wrist to middle fingertip.

Everything past this is room. See `EchoProfile.crop` for why that matters.
"""

MODEL_WINDOW_FRAMES: Final = 96
"""Frames per model input, 1.2 s at 80 frames per second."""

WATCHHAND_GROUND_TRUTH: Final = GroundTruth.VIDEO_FITTED
"""MediaPipe Hands on a webcam facing the palm, at 30 fps.

Not motion capture. Errors reported against this dataset are errors against
another estimator, and the published 7.87 mm cross-session figure inherits
whatever MediaPipe's own error is. The release states no bound on it.
"""

_PROFILE_SUFFIX: Final = "_fmcw_16bit_profiles.npy"
_DIFF_SUFFIX: Final = "_fmcw_16bit_diff_profiles.npy"


class WatchHandError(RuntimeError):
    """Anything wrong with the dataset layout or contents."""


class MissingLandmarksError(WatchHandError):
    """Raised when pose ground truth is asked for and none was generated.

    Its own class because this is not a corrupt-file problem, it is the
    dataset genuinely not containing what the caller assumed. Silently
    returning zeros or interpolating would produce a trainable target and a
    meaningless number.
    """


class Device(Enum):
    """The three watch models. Cross-device is a real generalisation axis.

    Speaker and microphone response in the 18 to 21 kHz band differs by tens
    of dB between models, so which watch a participant wore is part of a
    session's identity, not a footnote.
    """

    SAMSUNG = "samsung"
    XIAOMI = "xiaomi"
    GOOGLE = "google"


class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"


class Study(Enum):
    """The four sub-studies, which differ in layout and in what they hold."""

    MAIN = 1
    POSTURE = 2
    NOISE = 3
    DYNAMIC = 4


_STUDY_DIR_ALIASES: Final[dict[Study, tuple[str, ...]]] = {
    Study.MAIN: ("Study-1", "Study1"),
    Study.POSTURE: ("Study-2", "Study2"),
    Study.NOISE: ("Study-3", "Study3"),
    Study.DYNAMIC: ("Study-4", "Study4"),
}

_DEVICE_BY_PARTICIPANT: Final[dict[int, Device]] = {
    1: Device.SAMSUNG,
    2: Device.XIAOMI,
    3: Device.GOOGLE,
    4: Device.SAMSUNG,
    5: Device.XIAOMI,
    6: Device.XIAOMI,
    7: Device.GOOGLE,
    8: Device.GOOGLE,
    9: Device.SAMSUNG,
    10: Device.SAMSUNG,
    11: Device.SAMSUNG,
    12: Device.XIAOMI,
    13: Device.GOOGLE,
    14: Device.SAMSUNG,
    15: Device.XIAOMI,
    16: Device.GOOGLE,
    17: Device.XIAOMI,
    18: Device.GOOGLE,
    19: Device.XIAOMI,
    20: Device.GOOGLE,
    21: Device.XIAOMI,
    22: Device.SAMSUNG,
    23: Device.GOOGLE,
    24: Device.SAMSUNG,
}
"""Study 1 participant to watch model, read off the published folder names.

Studies 3 and 4 name folders `sub{N}_video` with no device in the name, so
their device is unknown from the layout alone. This table is not extended to
cover them by guessing, because a wrong device label silently corrupts a
cross-device split, which is precisely the split it would be used for.
"""

_STUDY12_DIR = re.compile(
    r"^sub(?P<pid>\d+)_(?P<device>samsung|xiaomi|google)_(?P<hand>left|right)"
    r"(?:_(?P<condition>sit|arm_up|arm_down))?_video$"
)
_STUDY3_DIR = re.compile(r"^sub(?P<pid>\d+)_video$")
_STUDY4_DIR = re.compile(r"^sub(?P<pid>\d+)_(?P<condition>normal|fast|slow|free)$")


@dataclass(frozen=True, slots=True)
class GestureLabel:
    """One labelled interval from a `*_records.txt` line.

    Timestamps are absolute Unix seconds, which is what the synchronisation
    formula consumes, so they are kept as recorded rather than rebased.
    """

    label_id: int
    start_s: float
    end_s: float
    name: str

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(f"gesture {self.name!r} ends before it starts")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class SessionRef:
    """A session located but not yet read.

    Separating locating from reading is what lets a split be constructed, a
    manifest be checked, and a participant count be reported without touching
    the 62 MB of float32 behind each session.
    """

    study: Study
    participant: int
    session_index: int
    """Zero based. Session 0 is the file named audio001."""

    directory: PurePosixPath
    """Relative to the dataset root, so it is also the manifest key prefix."""

    audio_stem: str
    """For example `audio001`. Profiles are this plus a fixed suffix."""

    device: Device | None = None
    """None where the published layout does not record it. Not inferred."""

    hand: Hand | None = None
    condition: str = ""
    """Body posture in study 2, motion speed in study 4, noise condition
    label in study 3. Empty where the study has no such axis."""

    def __post_init__(self) -> None:
        if self.participant < 1:
            raise ValueError("participant ids are one based")
        if self.session_index < 0:
            raise ValueError("session_index is zero based and cannot be negative")

    @property
    def profile_path(self) -> str:
        return (self.directory / f"{self.audio_stem}{_PROFILE_SUFFIX}").as_posix()

    @property
    def diff_profile_path(self) -> str:
        return (self.directory / f"{self.audio_stem}{_DIFF_SUFFIX}").as_posix()

    @property
    def landmarks_path(self) -> str:
        """Sidecar you generate. Never shipped. See module docstring."""
        return (self.directory / f"{self.audio_stem}_landmarks.npy").as_posix()

    @property
    def identity(self) -> str:
        """Stable, sortable, human-readable identity for manifests and splits.

        Everything that changes what a number means goes in: study, person,
        device, worn hand, condition, session. Two sessions with the same
        identity string are the same experimental cell.
        """
        device = self.device.value if self.device is not None else "unknown-device"
        hand = self.hand.value if self.hand is not None else "unknown-hand"
        condition = self.condition or "none"
        return (
            f"study{self.study.value}/p{self.participant:02d}/{device}/"
            f"{hand}/{condition}/s{self.session_index:03d}"
        )

    def describe(self) -> str:
        return self.identity


@dataclass(frozen=True, slots=True)
class SessionData:
    """A session's arrays and labels, already integrity checked.

    Profiles are memory mapped by default. A single study 1 folder is 1.2 GB
    of float32 and a full sweep is 175 GB, so eager loading is not a
    performance question, it is the difference between running and not.
    """

    ref: SessionRef
    profiles: NDArray[np.float32]
    """Shape (600, n_frames). Row is range bin, column is time."""

    diff_profiles: NDArray[np.float32]
    """Shape (600, n_frames - k) for a small k, since differencing loses
    frames. Do not assume the two arrays are the same width."""

    gestures: tuple[GestureLabel, ...]
    frame_timestamps: NDArray[np.float64]
    """Absolute Unix seconds, one per ground truth video frame, 30 fps."""

    audio_sync_index: int
    """Sample index in the original recording of the synchronisation pose."""

    gt_sync_timestamp: float
    sample_rate: int
    frame_length: int

    @property
    def n_frames(self) -> int:
        return int(self.profiles.shape[1])

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.frame_length

    def profile_index_for(self, timestamp_s: float) -> int:
        """Ground truth timestamp to echo profile column.

        The published formula, unchanged. It is affine in the timestamp, so
        it extrapolates happily past both ends of the recording; callers are
        expected to bounds check, and `frames` below does.
        """
        return round(
            (
                (timestamp_s - self.gt_sync_timestamp) * self.sample_rate
                + self.audio_sync_index
            )
            / self.frame_length
        )

    def timestamp_for(self, profile_index: int) -> float:
        """Inverse of `profile_index_for`, for putting labels back on a plot."""
        return (
            profile_index * self.frame_length - self.audio_sync_index
        ) / self.sample_rate + self.gt_sync_timestamp

    def echo_profile(
        self,
        frame_index: int,
        *,
        differential: bool = False,
        crop_bins: int | None = MODEL_CROP_BINS,
        normalise: bool = True,
    ) -> EchoProfile:
        """One frame, as the project's own type.

        `normalise` defaults on because the shipped profiles carry an
        undocumented absolute scale (see module docstring). Dividing by the
        frame's own peak makes a dataset frame and a live frame commensurable
        without knowing that scale, at the cost of discarding absolute
        reflectivity, which nothing downstream uses.
        """
        source = self.diff_profiles if differential else self.profiles
        width = int(source.shape[1])
        if not -width <= frame_index < width:
            raise IndexError(
                f"frame {frame_index} out of range for {width} frames "
                f"in {self.ref.identity}"
            )
        column = np.asarray(source[:, frame_index], dtype=np.float32)
        if crop_bins is not None:
            if not 0 < crop_bins <= column.shape[0]:
                raise ValueError(
                    f"crop_bins must be in 1..{column.shape[0]}, got {crop_bins}"
                )
            column = column[:crop_bins]
        if normalise:
            column = _peak_normalise(column)
        return EchoProfile(
            samples=np.ascontiguousarray(column, dtype=np.float32),
            bin_metres=PROFILE_BIN_METRES,
            timestamp_s=self.timestamp_for(frame_index % width),
            differential=differential,
        )

    def frames(
        self,
        *,
        differential: bool = False,
        crop_bins: int | None = MODEL_CROP_BINS,
        normalise: bool = True,
    ) -> Iterator[EchoProfile]:
        source = self.diff_profiles if differential else self.profiles
        for index in range(int(source.shape[1])):
            yield self.echo_profile(
                index,
                differential=differential,
                crop_bins=crop_bins,
                normalise=normalise,
            )

    def windows(
        self,
        *,
        width: int = MODEL_WINDOW_FRAMES,
        stride: int = 1,
        crop_bins: int = MODEL_CROP_BINS,
        normalise: bool = True,
    ) -> Iterator[tuple[int, NDArray[np.float32]]]:
        """Two-channel model inputs, shape (2, crop_bins, width).

        Channel order is original then differential, matching the paper. The
        two arrays are trimmed to their common length first, because the
        differential is shorter and pairing them by raw index would drift.
        """
        if width < 1 or stride < 1:
            raise ValueError("width and stride must be positive")
        if not 0 < crop_bins <= N_RANGE_BINS:
            raise ValueError(f"crop_bins must be in 1..{N_RANGE_BINS}")
        common = min(int(self.profiles.shape[1]), int(self.diff_profiles.shape[1]))
        if common < width:
            return
        original = np.asarray(self.profiles[:crop_bins, :common], dtype=np.float32)
        diff = np.asarray(self.diff_profiles[:crop_bins, :common], dtype=np.float32)
        for start in range(0, common - width + 1, stride):
            stop = start + width
            stacked = np.stack((original[:, start:stop], diff[:, start:stop]))
            if normalise:
                stacked = _peak_normalise(stacked)
            yield start, np.ascontiguousarray(stacked, dtype=np.float32)

    def gesture_at(self, timestamp_s: float) -> GestureLabel | None:
        for label in self.gestures:
            if label.start_s <= timestamp_s < label.end_s:
                return label
        return None

    def hand_poses(
        self,
        landmarks: NDArray[np.float32] | None,
        *,
        handedness: str = "right",
    ) -> tuple[HandPose, ...]:
        """Wrap generated MediaPipe landmarks in the project's pose type.

        Takes the array rather than reading it, so that the only place that
        decides where your landmarks live is `WatchHandDataset`. Raises rather
        than defaulting when there are none, because a silently empty ground
        truth is how an evaluation ends up measuring nothing.
        """
        if landmarks is None:
            raise MissingLandmarksError(
                f"no hand landmarks for {self.ref.identity}. WatchHand ships the "
                f"study video and gesture labels but not the fitted pose; run "
                f"MediaPipe Hands over {self.ref.directory} and write "
                f"{self.ref.audio_stem}_landmarks.npy of shape "
                f"(n_frames, {N_JOINTS}, 3)"
            )
        if landmarks.ndim != 3 or landmarks.shape[1:] != (N_JOINTS, 3):
            raise WatchHandError(
                f"landmarks for {self.ref.identity} must have shape "
                f"(n_frames, {N_JOINTS}, 3), got {landmarks.shape}"
            )
        return tuple(
            HandPose(
                joints=np.ascontiguousarray(frame, dtype=np.float32),
                wrist_relative=True,
                handedness=handedness,
            )
            for frame in _wrist_relative(landmarks)
        )


class WatchHandDataset:
    """The dataset on disk, gated behind a manifest.

    Construction requires a manifest. There is no keyword to skip it, and no
    default that quietly permits an unverified read, because the whole point
    is that an evaluation cannot run against data nobody pinned. Build one
    with `build_watchhand_manifest` after downloading, commit it, and let it
    fail loudly when the tree stops matching.
    """

    __slots__ = ("_root", "_verifier")

    def __init__(self, root: Path, manifest: Manifest) -> None:
        root = Path(root)
        if not root.is_dir():
            raise WatchHandError(f"dataset root {root} does not exist")
        if manifest.dataset != DATASET_NAME:
            raise IntegrityError(
                f"manifest names dataset {manifest.dataset!r}, "
                f"not {DATASET_NAME!r}; refusing to cross-load"
            )
        self._root = root
        self._verifier = LazyVerifier(manifest, root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> Manifest:
        return self._verifier.manifest

    @property
    def version(self) -> str:
        return self._verifier.manifest.version

    def sessions(
        self, studies: Sequence[Study] | None = None
    ) -> tuple[SessionRef, ...]:
        """Every session found on disk, in a stable order.

        Enumeration is by directory scan rather than by manifest, so that a
        tree carrying extra sessions is discovered here and rejected at load
        rather than being invisible.
        """
        wanted = tuple(Study) if studies is None else tuple(studies)
        found: list[SessionRef] = []
        for study in wanted:
            study_dir = self._study_dir(study)
            if study_dir is None:
                continue
            found.extend(_scan_study(study, study_dir, self._root))
        found.sort(key=lambda ref: ref.identity)
        return tuple(found)

    def load(self, ref: SessionRef, *, mmap: bool = True) -> SessionData:
        """Read one session, verifying every file it touches."""
        config = self._read_config(ref)
        audio_config = config["audio"]["config"]
        sample_rate = int(audio_config["sampling_rate"])
        frame_length = int(audio_config["frame_length"])
        if sample_rate != WATCHHAND_CHIRP.sample_rate:
            raise WatchHandError(
                f"{ref.identity} was recorded at {sample_rate} Hz, not "
                f"{WATCHHAND_CHIRP.sample_rate} Hz; the range axis constant "
                "in this module would be wrong for it"
            )
        if frame_length != WATCHHAND_CHIRP.n_samples:
            raise WatchHandError(
                f"{ref.identity} has frame_length {frame_length}, not "
                f"{WATCHHAND_CHIRP.n_samples}; the chirp does not match"
            )

        profiles = self._load_array(ref.profile_path, mmap=mmap)
        diff_profiles = self._load_array(ref.diff_profile_path, mmap=mmap)
        for name, array in (("profiles", profiles), ("diff", diff_profiles)):
            if array.ndim != 2 or array.shape[0] != N_RANGE_BINS:
                raise WatchHandError(
                    f"{ref.identity} {name} has shape {array.shape}, expected "
                    f"({N_RANGE_BINS}, n_frames)"
                )

        gestures, frame_timestamps = self._read_ground_truth(ref, config)
        audio_sync, gt_sync = _sync_pair(config, ref)

        return SessionData(
            ref=ref,
            profiles=profiles,
            diff_profiles=diff_profiles,
            gestures=gestures,
            frame_timestamps=frame_timestamps,
            audio_sync_index=audio_sync,
            gt_sync_timestamp=gt_sync,
            sample_rate=sample_rate,
            frame_length=frame_length,
        )

    def landmarks(self, ref: SessionRef) -> NDArray[np.float32] | None:
        """Generated pose sidecar for a session, or None if you have not made one."""
        path = self._root / PurePosixPath(ref.landmarks_path)
        if not path.is_file():
            return None
        if ref.landmarks_path in self.manifest:
            self._verifier.check(ref.landmarks_path)
        return np.asarray(np.load(path), dtype=np.float32)

    def _study_dir(self, study: Study) -> Path | None:
        for alias in _STUDY_DIR_ALIASES[study]:
            candidate = self._root / alias
            if candidate.is_dir():
                return candidate
        return None

    def _load_array(self, relative: str, *, mmap: bool) -> NDArray[np.float32]:
        path = self._verifier.check(relative)
        loaded = np.load(path, mmap_mode="r" if mmap else None)
        array = cast(NDArray[Any], loaded)
        if array.dtype != np.float32:
            raise WatchHandError(
                f"{relative} has dtype {array.dtype}, expected float32; "
                "every shipped WatchHand profile is float32"
            )
        return cast(NDArray[np.float32], array)

    def _read_config(self, ref: SessionRef) -> dict[str, Any]:
        relative = (ref.directory / "config.json").as_posix()
        path = self._verifier.check(relative)
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    def _read_ground_truth(
        self, ref: SessionRef, config: dict[str, Any]
    ) -> tuple[tuple[GestureLabel, ...], NDArray[np.float64]]:
        files = [str(name) for name in config["ground_truth"]["files"]]
        if not 0 <= ref.session_index < len(files):
            raise WatchHandError(
                f"{ref.identity} has no ground truth entry at index "
                f"{ref.session_index} among {len(files)} records files"
            )
        records_name = files[ref.session_index]
        records_rel = (ref.directory / records_name).as_posix()
        frame_time_rel = (
            ref.directory / records_name.replace("_records.txt", "_frame_time.txt")
        ).as_posix()

        gestures = _parse_records(self._verifier.check(records_rel))
        frame_times = _parse_frame_times(self._verifier.check(frame_time_rel))
        return gestures, frame_times


def build_watchhand_manifest(
    root: Path,
    *,
    version: str,
    notes: str = "",
) -> Manifest:
    """Manifest over the files that determine an evaluation's result.

    PNG previews, .DS_Store and the study videos are excluded. The videos are
    excluded reluctantly: they are the source of the pose ground truth, but
    they are also 57 MB each and are not read by any training or evaluation
    path once landmarks exist. Pin the landmark sidecars instead, which this
    does, since those are what a model is actually scored against.
    """
    from wristsonar.data.manifest import build_manifest

    dataset = _BareDataset(Path(root))
    identities = tuple(ref.identity for ref in dataset.sessions())
    return build_manifest(
        Path(root),
        dataset=DATASET_NAME,
        version=version,
        identities=identities,
        include_suffixes=(".npy", ".json", ".txt", ".csv"),
        notes=notes,
    )


def watchhand_protocol(
    split: Split,
    *,
    subjects: int,
    version: str,
    calibration_minutes: float = 0.0,
    held_out_poses: bool = False,
    seed: int | None = None,
    notes: str = "",
) -> Protocol:
    """A `Protocol` with WatchHand's fixed facts already correct.

    The ground truth field is not a parameter. It is MediaPipe on video for
    every session in this release, and letting a caller pass OPTICAL_MOCAP by
    accident would let a number claim an accuracy floor the data cannot
    support.
    """
    return Protocol(
        split=split,
        dataset=DATASET_NAME,
        dataset_version=version,
        ground_truth=WATCHHAND_GROUND_TRUTH,
        subjects=subjects,
        calibration_minutes=calibration_minutes,
        held_out_poses=held_out_poses,
        seed=seed,
        notes=notes,
    )


def estimate_bin_zero_offset(
    profiles: NDArray[np.float32], *, search_bins: int = 40
) -> int:
    """Bin index of the direct speaker-to-microphone path.

    WatchHand does not document which correlation lag became bin zero, and a
    live capture pipeline that gets it wrong shifts every range by a constant,
    which a model reads as a different hand. The direct path is by far the
    largest static reflector at essentially zero range, so its bin is
    measurable: take the median over time to kill hand motion, then find the
    peak in the near field.
    """
    if profiles.ndim != 2:
        raise ValueError(f"expected a (bins, frames) array, got {profiles.shape}")
    limit = min(search_bins, int(profiles.shape[0]))
    if limit < 1:
        raise ValueError("search_bins must select at least one bin")
    static = np.median(np.abs(np.asarray(profiles[:limit], dtype=np.float64)), axis=1)
    return int(np.argmax(static))


class _BareDataset:
    """Session enumeration without a manifest, for building the first one.

    Exists only so that `build_watchhand_manifest` can list identities before
    a manifest exists. Deliberately private: it is the one code path that
    reads the tree unverified, and it must not become a supported way to load.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    def sessions(self) -> tuple[SessionRef, ...]:
        found: list[SessionRef] = []
        for study in Study:
            for alias in _STUDY_DIR_ALIASES[study]:
                study_dir = self._root / alias
                if study_dir.is_dir():
                    found.extend(_scan_study(study, study_dir, self._root))
                    break
        found.sort(key=lambda ref: ref.identity)
        return tuple(found)


def _scan_study(study: Study, study_dir: Path, root: Path) -> list[SessionRef]:
    refs: list[SessionRef] = []
    for entry in sorted(study_dir.iterdir()):
        if not entry.is_dir():
            continue
        if study is Study.DYNAMIC:
            # Study 4 nests speed conditions one level deeper, under sub{N}_raw.
            for inner in sorted(entry.iterdir()):
                if inner.is_dir():
                    refs.extend(_scan_session_dir(study, inner, root))
        else:
            refs.extend(_scan_session_dir(study, entry, root))
    return refs


def _scan_session_dir(study: Study, directory: Path, root: Path) -> list[SessionRef]:
    parsed = _parse_dir_name(study, directory.name)
    if parsed is None:
        return []
    participant, device, hand, condition = parsed
    relative = PurePosixPath(directory.relative_to(root).as_posix())

    refs: list[SessionRef] = []
    for profile in sorted(directory.glob(f"*{_PROFILE_SUFFIX}")):
        stem = profile.name[: -len(_PROFILE_SUFFIX)]
        index = _session_index(stem)
        if index is None:
            continue
        refs.append(
            SessionRef(
                study=study,
                participant=participant,
                session_index=index,
                directory=relative,
                audio_stem=stem,
                device=device,
                hand=hand,
                condition=condition,
            )
        )
    return refs


def _parse_dir_name(
    study: Study, name: str
) -> tuple[int, Device | None, Hand | None, str] | None:
    if study in (Study.MAIN, Study.POSTURE):
        match = _STUDY12_DIR.match(name)
        if match is None:
            return None
        participant = int(match["pid"])
        device = Device(match["device"])
        expected = _DEVICE_BY_PARTICIPANT.get(participant)
        if expected is not None and expected is not device:
            # A folder claiming a watch its participant never wore means the
            # tree has been renamed or merged, which would quietly poison a
            # cross-device split. Cheaper to catch here than in a result.
            raise WatchHandError(
                f"{name} says {device.value}, but participant {participant} "
                f"used a {expected.value} watch in the published release"
            )
        return participant, device, Hand(match["hand"]), match["condition"] or ""
    if study is Study.NOISE:
        match = _STUDY3_DIR.match(name)
        if match is None:
            return None
        # Device and worn hand are absent from study 3 folder names. Left as
        # None rather than borrowed from study 1: the participant numbering
        # is independent between studies, so sub5 here is not sub5 there.
        return int(match["pid"]), None, None, ""
    match = _STUDY4_DIR.match(name)
    if match is None:
        return None
    return int(match["pid"]), None, None, match["condition"]


def _session_index(stem: str) -> int | None:
    """`audio001` becomes 0.

    The published loading example reads the profile filename out of
    `config['audio']['files']`, but that list holds the original `audioNNN.raw`
    names and the .raw files were never released. Deriving the index from the
    filename on disk avoids inheriting that bug.
    """
    match = re.fullmatch(r"audio(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1)) - 1


def _sync_pair(config: dict[str, Any], ref: SessionRef) -> tuple[int, float]:
    audio_sync = [int(v) for v in config["audio"]["syncing_poses"]]
    gt_sync = [float(v) for v in config["ground_truth"]["syncing_poses"]]
    if ref.session_index >= min(len(audio_sync), len(gt_sync)):
        raise WatchHandError(
            f"{ref.identity} has no synchronisation entry at index "
            f"{ref.session_index}; config lists {len(audio_sync)} audio and "
            f"{len(gt_sync)} ground truth sync points"
        )
    return audio_sync[ref.session_index], gt_sync[ref.session_index]


def _parse_records(path: Path) -> tuple[GestureLabel, ...]:
    """Parse `label_id,start,end,name` lines.

    The name column is not always an integer or a word: study 1 uses digits
    for finger bends and words for ASL, so it is kept as text and never
    coerced.
    """
    labels: list[GestureLabel] = []
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            raise WatchHandError(
                f"{path.name} line {lineno} has {len(parts)} fields, expected 4"
            )
        try:
            labels.append(
                GestureLabel(
                    label_id=int(parts[0]),
                    start_s=float(parts[1]),
                    end_s=float(parts[2]),
                    name=",".join(parts[3:]).strip(),
                )
            )
        except ValueError as exc:
            raise WatchHandError(
                f"{path.name} line {lineno} is not a gesture record: {line!r}"
            ) from exc
    return tuple(labels)


def _parse_frame_times(path: Path) -> NDArray[np.float64]:
    values: list[float] = []
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            values.append(float(line.split(",")[0]))
        except ValueError as exc:
            raise WatchHandError(
                f"{path.name} line {lineno} is not a timestamp: {line!r}"
            ) from exc
    return np.asarray(values, dtype=np.float64)


def _peak_normalise(array: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale by peak absolute value, leaving an all-zero array alone.

    Guarding the zero case rather than adding an epsilon, because an epsilon
    turns a silent frame into amplified numerical noise that looks like signal.
    """
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak == 0.0:
        return np.asarray(array, dtype=np.float32)
    return np.asarray(array / peak, dtype=np.float32)


def _wrist_relative(landmarks: NDArray[np.float32]) -> NDArray[np.float32]:
    """Subtract the wrist landmark from every joint of every frame.

    `HandPose` defaults to wrist_relative=True and a wrist-mounted sensor has
    no access to world position, so re-centring here means the flag never
    lies. Landmark 0 is the wrist in the MediaPipe ordering that
    `JOINT_NAMES` follows.
    """
    array = np.asarray(landmarks, dtype=np.float32)
    return np.asarray(array - array[:, 0:1, :], dtype=np.float32)
