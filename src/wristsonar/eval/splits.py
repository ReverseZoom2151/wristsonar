"""Split construction, and loud failure when a split leaks.

A leaked split is the single most likely way this project produces a wrong
headline number. It does not look like a bug. It looks like a good result.

The leak modes that matter here, in the order they actually happen:

Participant leak. The same person appears in train and test of a cross user
split, usually because a participant recorded under two identifiers or because
a random shuffle was applied over frames rather than over people. Since these
systems fit a per person hand manifold, one leaked participant can move the
headline by several millimetres.

Frame adjacency leak. Consecutive frames at 80 frames per second are almost
the same pose. A random frame level shuffle inside a session puts near
duplicates on both sides, which is why WITHIN_SESSION is scored here by a
contiguous block rather than by a shuffle, and why the within session number
is treated as the weakest claim in the enum.

Session leak. Cross session means a different mounting of the device. If the
watch never came off, the geometry never changed and the split is a within
session split wearing a different label.

Every constructor here calls assert_no_leakage before returning, and that
function raises. It does not warn. A warning in a long log is not a control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.eval.metrics import PoseArray, as_pose_array
from wristsonar.protocol import Split
from wristsonar.types import HandPose

__all__ = [
    "Dataset",
    "LeakageError",
    "SampleMeta",
    "SplitIndices",
    "assert_no_leakage",
    "cross_device_split",
    "cross_session_split",
    "cross_user_folds",
    "within_session_split",
]

IndexArray = NDArray[np.intp]


class LeakageError(AssertionError):
    """Raised when a split places related data on both sides.

    Deliberately an AssertionError subclass so that it reads as a violated
    invariant rather than as a recoverable condition. Nothing in this package
    catches it.
    """


@dataclass(frozen=True, slots=True)
class SampleMeta:
    """Who, when and on what. The three axes the four splits are cut along.

    session identifies one continuous wearing. Two recordings separated by
    taking the watch off and putting it back on are two sessions even if they
    are ten seconds apart, because remounting is the thing being tested.
    """

    participant: str
    session: str
    device: str
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("participant", "session", "device"):
            if not getattr(self, field_name):
                raise ValueError(f"SampleMeta.{field_name} must be non empty")


@dataclass(frozen=True, slots=True)
class Dataset:
    """Poses with the metadata needed to cut an honest split.

    features is optional because the trivial baselines that matter most do not
    use it. A nearest neighbour baseline does, and will say so if it is
    missing rather than quietly falling back to something that peeks at the
    labels.
    """

    poses: PoseArray
    meta: tuple[SampleMeta, ...]
    features: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.poses.shape[0] != len(self.meta):
            raise ValueError(
                f"poses and meta disagree: {self.poses.shape[0]} poses against "
                f"{len(self.meta)} metadata rows"
            )
        if len(self.meta) == 0:
            raise ValueError("a dataset must contain at least one sample")
        if self.features is not None and self.features.shape[0] != len(self.meta):
            raise ValueError(
                f"features and meta disagree: {self.features.shape[0]} rows "
                f"against {len(self.meta)}"
            )

    @classmethod
    def build(
        cls,
        poses: Sequence[HandPose] | NDArray[np.floating],
        meta: Sequence[SampleMeta],
        features: NDArray[np.floating] | None = None,
    ) -> Dataset:
        return cls(
            poses=as_pose_array(poses),
            meta=tuple(meta),
            features=(
                None if features is None else np.asarray(features, dtype=np.float64)
            ),
        )

    @property
    def n_samples(self) -> int:
        return len(self.meta)

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({m.participant for m in self.meta}))

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted({m.session for m in self.meta}))

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(sorted({m.device for m in self.meta}))

    def participant_of(self, idx: IndexArray) -> tuple[str, ...]:
        return tuple(self.meta[int(i)].participant for i in idx)

    def session_of(self, idx: IndexArray) -> tuple[str, ...]:
        return tuple(self.meta[int(i)].session for i in idx)

    def device_of(self, idx: IndexArray) -> tuple[str, ...]:
        return tuple(self.meta[int(i)].device for i in idx)

    def indices_where(
        self,
        *,
        participant: str | None = None,
        session: str | None = None,
        device: str | None = None,
    ) -> IndexArray:
        keep = [
            i
            for i, m in enumerate(self.meta)
            if (participant is None or m.participant == participant)
            and (session is None or m.session == session)
            and (device is None or m.device == device)
        ]
        return np.asarray(keep, dtype=np.intp)

    def require_features(self) -> NDArray[np.float64]:
        if self.features is None:
            raise ValueError(
                "this dataset carries no input features, so any predictor that "
                "needs them cannot be run; refusing to substitute ground truth"
            )
        return self.features


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """One train and test partition, labelled with the claim it supports."""

    name: str
    split: Split
    train: IndexArray
    test: IndexArray

    def __post_init__(self) -> None:
        if self.train.size == 0:
            raise ValueError(f"split {self.name} has an empty training set")
        if self.test.size == 0:
            raise ValueError(f"split {self.name} has an empty test set")
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise LeakageError(
                f"split {self.name} places {overlap.size} sample indices in both "
                f"train and test, for example index {int(overlap[0])}"
            )

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)


def assert_no_leakage(dataset: Dataset, indices: SplitIndices) -> None:
    """Raise LeakageError if the partition violates its own split's promise.

    Checked per split kind, because each split promises something different:

    CROSS_USER promises that no participant appears on both sides. This is the
    check the whole module exists for.

    CROSS_SESSION promises that no session appears on both sides. It
    deliberately does not require disjoint participants, because the same
    person on a different mounting is exactly what is being measured.

    CROSS_DEVICE promises that no device model appears on both sides.

    WITHIN_SESSION promises nothing beyond disjoint sample indices, which is
    why it is the weakest claim in the enum and why the report layer will not
    let it stand alone.
    """
    if indices.split is Split.CROSS_USER:
        shared = set(dataset.participant_of(indices.train)) & set(
            dataset.participant_of(indices.test)
        )
        if shared:
            raise LeakageError(
                f"cross-user split {indices.name!r} leaks: participants "
                f"{sorted(shared)} appear in both training and test data. "
                f"A cross-user number computed from this split is wrong, and "
                f"wrong in the flattering direction."
            )
    elif indices.split is Split.CROSS_SESSION:
        shared_sessions = set(dataset.session_of(indices.train)) & set(
            dataset.session_of(indices.test)
        )
        if shared_sessions:
            raise LeakageError(
                f"cross-session split {indices.name!r} leaks: sessions "
                f"{sorted(shared_sessions)} appear on both sides. The device "
                f"was never remounted, so this is a within-session split "
                f"wearing a cross-session label."
            )
    elif indices.split is Split.CROSS_DEVICE:
        shared_devices = set(dataset.device_of(indices.train)) & set(
            dataset.device_of(indices.test)
        )
        if shared_devices:
            raise LeakageError(
                f"cross-device split {indices.name!r} leaks: devices "
                f"{sorted(shared_devices)} appear on both sides."
            )

    max_index = max(int(indices.train.max()), int(indices.test.max()))
    if max_index >= dataset.n_samples:
        raise LeakageError(
            f"split {indices.name!r} indexes sample {max_index} but the "
            f"dataset holds {dataset.n_samples}"
        )


def _checked(dataset: Dataset, indices: SplitIndices) -> SplitIndices:
    assert_no_leakage(dataset, indices)
    return indices


def within_session_split(
    dataset: Dataset,
    *,
    session: str | None = None,
    test_fraction: float = 0.2,
    contiguous: bool = True,
) -> SplitIndices:
    """The weakest split, cut inside one continuous wearing.

    Contiguous by default. A random frame level shuffle inside a session puts
    near identical neighbouring frames on both sides, which is how a within
    session number gets to be several times better than it has any right to
    be. RAM-Hand's 10.71 mm is a within session 80/20 split, the weakest
    protocol in the field, and this function exists mostly so that number can
    be reproduced honestly and then labelled as what it is.

    Pass contiguous=False only to demonstrate how much a shuffle buys.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    if session is None:
        session = dataset.sessions[0]
    idx = dataset.indices_where(session=session)
    if idx.size < 2:
        raise ValueError(f"session {session!r} has too few samples to split")
    order = np.argsort([dataset.meta[int(i)].timestamp_s for i in idx], kind="stable")
    idx = idx[order]
    n_test = max(1, round(idx.size * test_fraction))
    if n_test >= idx.size:
        n_test = idx.size - 1
    if contiguous:
        test = idx[-n_test:]
        train = idx[:-n_test]
    else:
        shuffled = idx.copy()
        np.random.default_rng(0).shuffle(shuffled)
        test = shuffled[:n_test]
        train = shuffled[n_test:]
    return _checked(
        dataset,
        SplitIndices(
            name=f"within-session[{session}]",
            split=Split.WITHIN_SESSION,
            train=np.sort(train),
            test=np.sort(test),
        ),
    )


def cross_session_split(
    dataset: Dataset,
    *,
    participant: str | None = None,
    held_out_session: str | None = None,
) -> SplitIndices:
    """Same person, different mounting of the device.

    The first honest number. Training uses every session of this participant
    except one, testing uses only the held out session. Other participants are
    excluded entirely so that the number is about remounting and not about
    population size.
    """
    if participant is None:
        participant = dataset.participants[0]
    own = dataset.indices_where(participant=participant)
    if own.size == 0:
        raise ValueError(f"no samples for participant {participant!r}")
    sessions = sorted({dataset.meta[int(i)].session for i in own})
    if len(sessions) < 2:
        raise ValueError(
            f"participant {participant!r} has only session {sessions[0]!r}; a "
            f"cross-session split needs a remount, and inventing one by "
            f"cutting a single session in half would be a within-session "
            f"split under a stronger label"
        )
    if held_out_session is None:
        held_out_session = sessions[-1]
    elif held_out_session not in sessions:
        raise ValueError(
            f"session {held_out_session!r} is not a session of {participant!r}"
        )
    test = np.asarray(
        [i for i in own if dataset.meta[int(i)].session == held_out_session],
        dtype=np.intp,
    )
    train = np.asarray(
        [i for i in own if dataset.meta[int(i)].session != held_out_session],
        dtype=np.intp,
    )
    return _checked(
        dataset,
        SplitIndices(
            name=f"cross-session[{participant}/{held_out_session}]",
            split=Split.CROSS_SESSION,
            train=train,
            test=test,
        ),
    )


def cross_user_folds(
    dataset: Dataset,
    *,
    participants: Sequence[str] | None = None,
) -> tuple[SplitIndices, ...]:
    """Leave one participant out, one fold per person.

    The number that decides whether the system is useful to anyone who did not
    help build it. Expect two to three times the cross session error:
    EchoWrist quotes 4.81 mm in its abstract and 12.2 mm cross user in section
    6.5, and WatchHand goes 6.02 mm within session, 7.87 mm cross session,
    14.88 mm cross user.

    Every fold is leakage checked before it is returned, so a mislabelled
    participant identifier fails here rather than surfacing as an unusually
    good headline.
    """
    people = tuple(participants) if participants is not None else dataset.participants
    if len(people) < 2:
        raise ValueError(
            "leave-one-user-out needs at least two participants; with one "
            "person there is no cross-user question to ask"
        )
    folds: list[SplitIndices] = []
    for person in people:
        test = dataset.indices_where(participant=person)
        if test.size == 0:
            raise ValueError(f"no samples for participant {person!r}")
        train = np.asarray(
            [
                i
                for i in range(dataset.n_samples)
                if dataset.meta[i].participant != person
            ],
            dtype=np.intp,
        )
        folds.append(
            _checked(
                dataset,
                SplitIndices(
                    name=f"cross-user[held-out={person}]",
                    split=Split.CROSS_USER,
                    train=train,
                    test=test,
                ),
            )
        )
    return tuple(folds)


def cross_device_split(
    dataset: Dataset,
    *,
    held_out_device: str | None = None,
) -> SplitIndices:
    """Test on a device model that contributed no training data.

    A distinct failure axis from cross user, because speaker and microphone
    response varies by tens of dB between models in the 18 to 22 kHz band.
    Usually not reported at all, which is why it is a first class constructor
    here rather than an option on another one.
    """
    devices = dataset.devices
    if len(devices) < 2:
        raise ValueError(
            "a cross-device split needs at least two device models; this "
            "dataset has "
            f"{devices!r}"
        )
    if held_out_device is None:
        held_out_device = devices[-1]
    elif held_out_device not in devices:
        raise ValueError(f"device {held_out_device!r} is not in the dataset")
    test = dataset.indices_where(device=held_out_device)
    train = np.asarray(
        [
            i
            for i in range(dataset.n_samples)
            if dataset.meta[i].device != held_out_device
        ],
        dtype=np.intp,
    )
    return _checked(
        dataset,
        SplitIndices(
            name=f"cross-device[held-out={held_out_device}]",
            split=Split.CROSS_DEVICE,
            train=train,
            test=test,
        ),
    )
