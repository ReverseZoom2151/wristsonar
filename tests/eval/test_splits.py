"""Split construction, and the leakage assertions that must raise."""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset
from wristsonar.eval.splits import (
    Dataset,
    LeakageError,
    SampleMeta,
    SplitIndices,
    assert_no_leakage,
    cross_device_split,
    cross_session_split,
    cross_user_folds,
    within_session_split,
)
from wristsonar.protocol import Split


def test_cross_user_folds_hold_out_exactly_one_person() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    folds = cross_user_folds(data)

    assert len(folds) == 4
    for fold, person in zip(folds, data.participants, strict=True):
        assert fold.split is Split.CROSS_USER
        assert set(data.participant_of(fold.test)) == {person}
        assert person not in set(data.participant_of(fold.train))
        assert fold.n_train + fold.n_test == data.n_samples


def test_a_leaked_cross_user_split_raises_rather_than_warns() -> None:
    data = make_dataset(n_participants=3, n_sessions=1, n_frames=8)
    held_out = data.participants[0]
    test = data.indices_where(participant=held_out)
    # The classic accident: one participant recorded under two identifiers, so
    # a few of their frames end up on the training side.
    others = np.asarray(
        [i for i in range(data.n_samples) if i not in set(test.tolist())],
        dtype=np.intp,
    )
    indices = SplitIndices(
        name="leaky",
        split=Split.CROSS_USER,
        train=np.sort(np.concatenate([others, test[:3]])),
        test=test[3:],
    )

    with pytest.raises(LeakageError, match="leaks: participants"):
        assert_no_leakage(data, indices)


def test_index_overlap_is_caught_at_construction() -> None:
    with pytest.raises(LeakageError, match="both"):
        SplitIndices(
            name="overlapping",
            split=Split.WITHIN_SESSION,
            train=np.asarray([0, 1, 2], dtype=np.intp),
            test=np.asarray([2, 3], dtype=np.intp),
        )


def test_cross_session_split_rejects_a_shared_session() -> None:
    data = make_dataset(n_participants=2, n_sessions=2, n_frames=6)
    person = data.participants[0]
    own = data.indices_where(participant=person)
    indices = SplitIndices(
        name="fake-cross-session",
        split=Split.CROSS_SESSION,
        train=own[:3],
        test=own[3:6],
    )
    # Both halves come from the same session, so the device never came off.
    with pytest.raises(LeakageError, match="never remounted"):
        assert_no_leakage(data, indices)


def test_cross_session_needs_an_actual_remount() -> None:
    data = make_dataset(n_participants=2, n_sessions=1, n_frames=6)
    with pytest.raises(ValueError, match="needs a remount"):
        cross_session_split(data)

    good = make_dataset(n_participants=2, n_sessions=3, n_frames=6)
    fold = cross_session_split(good, participant="P00")
    assert fold.split is Split.CROSS_SESSION
    assert set(good.participant_of(fold.test)) == {"P00"}
    assert set(good.session_of(fold.test)).isdisjoint(set(good.session_of(fold.train)))


def test_within_session_is_contiguous_by_default() -> None:
    data = make_dataset(n_participants=1, n_sessions=1, n_frames=20)
    fold = within_session_split(data, test_fraction=0.25)

    assert fold.split is Split.WITHIN_SESSION
    assert fold.n_test == 5
    assert int(fold.train.max()) < int(fold.test.min())


def test_cross_device_split_holds_out_a_device_model() -> None:
    data = make_dataset(
        n_participants=4, n_sessions=2, n_frames=6, devices=("watch-a", "watch-b")
    )
    fold = cross_device_split(data, held_out_device="watch-b")

    assert fold.split is Split.CROSS_DEVICE
    assert set(data.device_of(fold.test)) == {"watch-b"}
    assert "watch-b" not in set(data.device_of(fold.train))

    single = make_dataset(n_participants=2, n_sessions=1, n_frames=4)
    with pytest.raises(ValueError, match="at least two device models"):
        cross_device_split(single)


def test_single_participant_cannot_be_asked_a_cross_user_question() -> None:
    data = make_dataset(n_participants=1, n_sessions=2, n_frames=5)
    with pytest.raises(ValueError, match="at least two participants"):
        cross_user_folds(data)


def test_dataset_rejects_mismatched_metadata() -> None:
    poses = np.zeros((3, 21, 3), dtype=np.float64)
    meta = [SampleMeta(participant="P0", session="S0", device="d")]
    with pytest.raises(ValueError, match="disagree"):
        Dataset(poses=poses, meta=tuple(meta))
