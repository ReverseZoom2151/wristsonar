"""Trivial predictors, shipped as first class citizens.

These are not sanity checks tucked into a test file. They are the comparison
that decides whether a model in this field has done anything, and they belong
in the same table as the model, in the same run, under the same protocol.

Why they are so strong here. Hands are not free to take arbitrary
configurations. Tendon coupling means twenty joints have far fewer than sixty
effective degrees of freedom, and the poses in any recorded dataset cluster
hard around a few resting and gesturing configurations. A predictor that
ignores the microphone entirely and always emits the average of that cluster
therefore lands within a few millimetres of the truth on most frames. That is
also the exact mechanism a real model uses, since range resolution is about
5.7 cm and the millimetre output comes from a learned prior over hand shape.
The model is trying to beat the mean pose using a signal that resolves nothing
finer than a whole hand.

Three baselines, in increasing order of how embarrassing they are:

Mean pose. One pose for every frame of every person. Scores embarrassingly
well on MPJPE because MPJPE averages over twenty one joints, fifteen of which
barely move relative to the wrist.

Nearest neighbour. Looks up the training frame whose input features are
closest and returns its pose. Needs features, and refuses to run without
them rather than peeking at the labels.

Per subject mean pose. One pose per person. The strongest trivial baseline and
the one most likely to embarrass a real model, because hand size and resting
posture are largely a per person constant and a within user protocol hands
that constant to the baseline for free. On a cross user split it necessarily
degrades to the global mean, and the size of that gap is a direct measurement
of how much of a within user result was person specific bias rather than
sensing.

The rule this module exists to enforce: any model that does not clearly beat
all three must be reported as not clearly beating them. Report.render_text
does that automatically and does not offer a way to turn it off.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from wristsonar.eval.metrics import (
    Alignment,
    MetricSet,
    PoseArray,
    evaluate,
)
from wristsonar.eval.splits import Dataset, IndexArray, SplitIndices
from wristsonar.protocol import Protocol
from wristsonar.types import N_JOINTS

__all__ = [
    "Baseline",
    "MeanPoseBaseline",
    "NearestNeighbourBaseline",
    "PerSubjectMeanPoseBaseline",
    "default_baselines",
    "evaluate_baselines",
]


class Baseline(ABC):
    """A predictor with no learned parameters worth the name.

    The interface is deliberately the same shape a real model would use, so a
    model can be dropped into evaluate_baselines and compared without any
    special casing that could quietly change the protocol.
    """

    name: str

    @abstractmethod
    def fit(self, dataset: Dataset, train: IndexArray) -> None:
        """Absorb the training half of a split. Must not look at test."""

    @abstractmethod
    def predict(self, dataset: Dataset, test: IndexArray) -> PoseArray:
        """Return poses of shape (len(test), N_JOINTS, 3) in metres."""

    def fit_predict(self, dataset: Dataset, indices: SplitIndices) -> PoseArray:
        self.fit(dataset, indices.train)
        return self.predict(dataset, indices.test)


@dataclass
class MeanPoseBaseline(Baseline):
    """The arithmetic mean training pose, emitted for every test frame.

    Ignores the sensor completely. Expect it to score in the same order of
    magnitude as published systems on root aligned MPJPE, and to be visibly
    beaten only on the fingertips, which is the whole argument for the per
    joint breakdown being mandatory.
    """

    name: str = "mean-pose"
    _mean: PoseArray | None = field(default=None, repr=False)

    def fit(self, dataset: Dataset, train: IndexArray) -> None:
        if train.size == 0:
            raise ValueError("mean-pose baseline needs at least one training pose")
        self._mean = dataset.poses[train].mean(axis=0)

    def predict(self, dataset: Dataset, test: IndexArray) -> PoseArray:
        if self._mean is None:
            raise RuntimeError("fit must be called before predict")
        return np.repeat(self._mean[None, ...], test.size, axis=0)


@dataclass
class NearestNeighbourBaseline(Baseline):
    """The training pose whose input features are closest to the test frame.

    A genuine predictor in the sense that it reads the microphone, and a
    non parametric upper bound on what pure memorisation of the training set
    buys. If a learned model does not beat this, the model has learned a
    lookup table with extra steps.

    Requires features. If the dataset has none it raises, because the only
    alternatives are to compare poses to poses, which is reading the answer,
    or to fall back to the mean, which would be a different baseline
    reported under this one's name.
    """

    name: str = "nearest-neighbour"
    _train_features: NDArray[np.float64] | None = field(default=None, repr=False)
    _train_poses: PoseArray | None = field(default=None, repr=False)

    def fit(self, dataset: Dataset, train: IndexArray) -> None:
        features = dataset.require_features()
        if train.size == 0:
            raise ValueError("nearest-neighbour baseline needs training samples")
        self._train_features = features[train]
        self._train_poses = dataset.poses[train]

    def predict(self, dataset: Dataset, test: IndexArray) -> PoseArray:
        if self._train_features is None or self._train_poses is None:
            raise RuntimeError("fit must be called before predict")
        query = dataset.require_features()[test]
        # Squared Euclidean distance, expanded so the whole thing stays in one
        # matrix product. Datasets in this field are small enough that this is
        # cheaper than any tree.
        d2 = (
            (query**2).sum(axis=1)[:, None]
            - 2.0 * query @ self._train_features.T
            + (self._train_features**2).sum(axis=1)[None, :]
        )
        nearest = np.argmin(d2, axis=1)
        out: PoseArray = self._train_poses[nearest]
        return out


@dataclass
class PerSubjectMeanPoseBaseline(Baseline):
    """One mean pose per participant. The strongest trivial baseline.

    On a within session or cross session protocol this is close to free and
    close to devastating, because hand size and habitual resting posture are
    largely per person constants and the split hands them over.

    On a cross user split the test participant has no training data by
    construction, so every prediction falls back to the global mean and this
    becomes MeanPoseBaseline exactly. That collapse is informative rather than
    a defect: unseen_fraction reports how often the fallback fired, and a
    cross user run where it is not 1.0 means the split leaked.

    Using participant identity at prediction time is legitimate for a personal
    device, which does know whose wrist it is on. It is not legitimate to
    quote the resulting number as a cross user result, and the split machinery
    is what prevents that.
    """

    name: str = "per-subject-mean-pose"
    _by_subject: dict[str, PoseArray] = field(default_factory=dict, repr=False)
    _global: PoseArray | None = field(default=None, repr=False)
    unseen_fraction: float = 0.0

    def fit(self, dataset: Dataset, train: IndexArray) -> None:
        if train.size == 0:
            raise ValueError("per-subject baseline needs training samples")
        self._global = dataset.poses[train].mean(axis=0)
        self._by_subject = {}
        people = {dataset.meta[int(i)].participant for i in train}
        for person in people:
            own = np.asarray(
                [i for i in train if dataset.meta[int(i)].participant == person],
                dtype=np.intp,
            )
            self._by_subject[person] = dataset.poses[own].mean(axis=0)

    def predict(self, dataset: Dataset, test: IndexArray) -> PoseArray:
        if self._global is None:
            raise RuntimeError("fit must be called before predict")
        out = np.empty((test.size, N_JOINTS, 3), dtype=np.float64)
        unseen = 0
        for row, i in enumerate(test):
            person = dataset.meta[int(i)].participant
            mean = self._by_subject.get(person)
            if mean is None:
                mean = self._global
                unseen += 1
            out[row] = mean
        self.unseen_fraction = float(unseen) / float(test.size)
        return out


def default_baselines(*, include_nearest_neighbour: bool = True) -> list[Baseline]:
    """The three trivial predictors, freshly constructed.

    Fresh instances every call because a Baseline holds fitted state and
    reusing one across folds is a leak of the quiet kind.
    """
    baselines: list[Baseline] = [MeanPoseBaseline(), PerSubjectMeanPoseBaseline()]
    if include_nearest_neighbour:
        baselines.append(NearestNeighbourBaseline())
    return baselines


def evaluate_baselines(
    dataset: Dataset,
    indices: SplitIndices,
    protocol: Protocol,
    *,
    baselines: list[Baseline] | None = None,
    alignment: Alignment = Alignment.ROOT,
) -> dict[str, MetricSet]:
    """Fit and score every baseline on one split, under one protocol.

    The protocol passed here must be the same object handed to the model, and
    the split must be the same split. Anything else produces a table whose
    rows are not comparable, which is worse than no table.

    A dataset without features silently drops the nearest neighbour baseline
    rather than failing the whole run, because the two mean pose baselines are
    the ones that carry the argument and a features free dataset is a normal
    state for a pose only sanity run. The dropped row simply does not appear,
    so nobody can mistake it for a score.
    """
    if baselines is None:
        baselines = default_baselines(
            include_nearest_neighbour=dataset.features is not None
        )
    truth = dataset.poses[indices.test]
    results: dict[str, MetricSet] = {}
    for baseline in baselines:
        pred = baseline.fit_predict(dataset, indices)
        results[baseline.name] = evaluate(pred, truth, protocol, alignment=alignment)
    return results
