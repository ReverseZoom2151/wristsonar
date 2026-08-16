from __future__ import annotations

import numpy as np
import pytest

from wristsonar.model.corpus import TrainingCorpus
from wristsonar.model.dataset import WindowExample


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
