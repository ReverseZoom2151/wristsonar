from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np
import pytest
from numpy.typing import NDArray

from wristsonar.data.watchhand import (
    SessionData,
    SessionRef,
    Study,
    WatchHandDataset,
)
from wristsonar.model.corpus import SessionWindowSource, TrainingCorpus
from wristsonar.model.dataset import WindowExample
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    PreprocessingMismatchError,
)
from wristsonar.types import N_JOINTS


def _example(index: int) -> WindowExample:
    return WindowExample(
        features=np.full((2, 3, 4), index, dtype=np.float32),
        target=np.full((21, 3), index / 100.0, dtype=np.float32),
        participant="study1/sub1",
        session="study1/p01/s000",
        device="samsung",
        timestamp_s=float(index),
    )


def test_corpus_preserves_metadata_and_only_flattens_features_on_request() -> None:
    corpus = TrainingCorpus.from_examples([_example(1), _example(2)])

    compact = corpus.evaluation_dataset()
    with_features = corpus.evaluation_dataset(include_features=True)

    assert corpus.features.shape == (2, 2, 3, 4)
    assert compact.features is None
    assert with_features.features is not None
    assert with_features.features.shape == (2, 24)
    assert compact.meta[1].timestamp_s == 2.0


def test_corpus_rejects_empty_or_inconsistent_windows() -> None:
    with pytest.raises(ValueError, match="no examples"):
        TrainingCorpus.from_examples([])
    bad = WindowExample(
        features=np.zeros((2, 2, 4), dtype=np.float32),
        target=np.zeros((21, 3), dtype=np.float32),
        participant="study1/sub1",
        session="s1",
        device="samsung",
        timestamp_s=0,
    )
    with pytest.raises(ValueError, match="same shape"):
        TrainingCorpus.from_examples([_example(1), bad])


def test_a_corpus_carrying_a_contract_must_match_it() -> None:
    """A corpus that states its preprocessing cannot hold a different shape.

    The point is the checkpoint at the other end: it records the descriptor
    the corpus was built with, and that record is worth nothing if the two
    could disagree.
    """
    tiny = WATCHHAND_PREPROCESSING.with_window_frames(4)

    with pytest.raises(ValueError, match="preprocessing contract says"):
        TrainingCorpus.from_examples([_example(1)], descriptor=tiny)

    matching = WindowExample(
        features=np.zeros(tiny.window_shape, dtype=np.float32),
        target=np.zeros((21, 3), dtype=np.float32),
        participant="study1/sub1",
        session="study1/p01/s000",
        device="samsung",
        timestamp_s=0.0,
    )
    corpus = TrainingCorpus.from_examples([matching], descriptor=tiny)
    assert corpus.descriptor is tiny


def test_corpus_requires_study_qualified_participants_for_cross_user_splits() -> None:
    first = _example(1)
    second = WindowExample(
        features=first.features.copy(),
        target=first.target.copy(),
        participant="study3/sub1",
        session="study3/p01/s000",
        device="unknown-device",
        timestamp_s=2.0,
    )

    corpus = TrainingCorpus.from_examples([first, second])

    assert corpus.evaluation_dataset().participants == ("study1/sub1", "study3/sub1")


_TINY = WATCHHAND_PREPROCESSING.with_window_frames(4)
_REF = SessionRef(Study.MAIN, 1, 0, PurePosixPath("synthetic"), "audio001")


def _session() -> SessionData:
    """A session whose arrays state their own preprocessing.

    The direct path sits in bin 0 and the differential is trimmed to the lag
    the default descriptor declares, so a correct descriptor verifies and a
    wrong one has something to be wrong about.
    """
    rng = np.random.default_rng(5)
    profiles = np.abs(rng.normal(0.0, 0.05, size=(600, 40))).astype(np.float32)
    profiles[0, :] += 20.0
    lag = WATCHHAND_PREPROCESSING.differential_lag
    diff = np.ascontiguousarray(
        np.diff(profiles, axis=1)[:, lag - 1 :], dtype=np.float32
    )
    return SessionData(
        ref=_REF,
        profiles=profiles,
        diff_profiles=diff,
        gestures=(),
        frame_timestamps=np.arange(5, dtype=np.float64) / 30.0,
        audio_sync_index=0,
        gt_sync_timestamp=0.0,
        sample_rate=48_000,
        frame_length=600,
    )


class _OneSessionDataset(WatchHandDataset):
    """A dataset of exactly one in-memory session.

    A subclass rather than a stand-in object, so the corpus builder is
    exercised through the type it really takes. Every method that would touch
    a root directory or a manifest is overridden, which is also the list of
    methods the builder is allowed to call.
    """

    def __init__(self, session: SessionData) -> None:
        self._session = session

    def sessions(self, studies: object = None) -> tuple[SessionRef, ...]:
        return (self._session.ref,)

    def load(self, ref: SessionRef, *, mmap: bool = True) -> SessionData:
        return self._session

    def landmarks(self, ref: SessionRef) -> NDArray[np.float32] | None:
        return np.zeros((5, N_JOINTS, 3), dtype=np.float32)


def test_the_corpus_builder_measures_every_session_before_using_it() -> None:
    """The one path that can measure the undocumented parameters, doing it.

    `estimate_bin_zero_offset` existed for a release and was called from
    nowhere in `src`, which meant the training crop origin and the live crop
    origin were free to differ by a constant nobody would ever notice. It is
    called here, per session, before any window of that session is yielded.
    """
    source = SessionWindowSource(
        dataset=_OneSessionDataset(_session()),
        refs=(_REF,),
        descriptor=_TINY,
        stride=8,
    )

    corpus = source.materialize()

    assert corpus.descriptor is _TINY
    assert corpus.features.shape[1:] == _TINY.window_shape

    wrong_origin = SessionWindowSource(
        dataset=_OneSessionDataset(_session()),
        refs=(_REF,),
        descriptor=_TINY.with_bin_zero_offset(9),
        stride=8,
    )
    with pytest.raises(PreprocessingMismatchError, match="direct path at bin 0"):
        wrong_origin.materialize()
