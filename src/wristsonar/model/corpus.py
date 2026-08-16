"""Lazy WatchHand windows and explicitly materialized training corpora.

The public release is approximately 175 GB and overlapping causal windows can
expand it by orders of magnitude.  Loading every window just to inspect the
dataset is therefore a bug, not a convenience.  ``SessionWindowSource`` keeps
the source lazy; materialization is a conscious request with an optional hard
limit for small experiments and tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.data.watchhand import SessionRef, WatchHandDataset
from wristsonar.eval.splits import Dataset, SampleMeta
from wristsonar.model.dataset import PoseWindows, WindowExample
from wristsonar.preprocess import (
    WATCHHAND_PREPROCESSING,
    PreprocessingDescriptor,
)
from wristsonar.types import N_JOINTS

__all__ = ["SessionWindowSource", "TrainingCorpus"]


@dataclass(frozen=True, slots=True)
class TrainingCorpus:
    """Materialized windows, targets and split metadata with shape checks."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    meta: tuple[SampleMeta, ...]
    descriptor: PreprocessingDescriptor | None = None
    """The contract these windows were built under, when they came from one.

    Carried so that a checkpoint exported from this corpus records the same
    preprocessing the corpus was materialized with, rather than whatever the
    exporter's caller happened to type. None is for hand-assembled arrays in
    tests and baselines, which have no session behind them and therefore no
    contract to inherit; shape checking against the descriptor is skipped for
    those rather than being invented.
    """

    def __post_init__(self) -> None:
        if self.features.ndim != 4 or self.features.shape[0] < 1:
            raise ValueError(
                "features must be nonempty (n, channels, bins, frames), got "
                f"{self.features.shape}"
            )
        if self.features.shape[1] != 2:
            raise ValueError("training features must hold original and differential")
        if self.targets.shape != (self.features.shape[0], N_JOINTS, 3):
            raise ValueError(
                "targets must match feature count and the 21-joint pose shape, got "
                f"{self.targets.shape}"
            )
        if len(self.meta) != self.features.shape[0]:
            raise ValueError("one SampleMeta is required for every training example")
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise ValueError("training corpus contains non-finite values")
        if (
            self.descriptor is not None
            and tuple(self.features.shape[1:]) != self.descriptor.window_shape
        ):
            raise ValueError(
                f"features are {tuple(self.features.shape[1:])} per example but "
                f"the preprocessing contract says {self.descriptor.window_shape}"
            )

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[WindowExample],
        *,
        descriptor: PreprocessingDescriptor | None = None,
    ) -> TrainingCorpus:
        """Materialize a finite selection of examples without changing order."""
        items = tuple(examples)
        if not items:
            raise ValueError("cannot build a training corpus with no examples")
        first = items[0].features.shape
        if any(item.features.shape != first for item in items):
            raise ValueError("all training windows must have the same shape")
        return cls(
            features=np.stack([item.features for item in items]).astype(np.float32),
            targets=np.stack([item.target for item in items]).astype(np.float32),
            descriptor=descriptor,
            meta=tuple(
                SampleMeta(
                    participant=item.participant,
                    session=item.session,
                    device=item.device,
                    timestamp_s=item.timestamp_s,
                )
                for item in items
            ),
        )

    def evaluation_dataset(self, *, include_features: bool = False) -> Dataset:
        """Expose targets and metadata to the existing honest split harness.

        Flattening full echo windows is expensive and needed only for the
        nearest-neighbour baseline, so it is explicit.  The two mean-pose
        baselines and every split can run without copying that large matrix.
        """
        features: NDArray[np.float64] | None = None
        if include_features:
            features = self.features.reshape(self.features.shape[0], -1).astype(
                np.float64
            )
        return Dataset(
            poses=np.asarray(self.targets, dtype=np.float64),
            meta=self.meta,
            features=features,
        )


@dataclass(frozen=True, slots=True)
class SessionWindowSource:
    """A reproducible, lazy view over verified WatchHand sessions."""

    dataset: WatchHandDataset
    refs: tuple[SessionRef, ...]
    descriptor: PreprocessingDescriptor = WATCHHAND_PREPROCESSING
    stride: int = 1

    def __post_init__(self) -> None:
        if not self.refs:
            raise ValueError("a window source needs at least one session")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        identities = [ref.identity for ref in self.refs]
        if len(set(identities)) != len(identities):
            raise ValueError("a window source cannot repeat one session")

    @classmethod
    def all(
        cls,
        dataset: WatchHandDataset,
        *,
        descriptor: PreprocessingDescriptor = WATCHHAND_PREPROCESSING,
        stride: int = 1,
    ) -> SessionWindowSource:
        return cls(
            dataset=dataset,
            refs=dataset.sessions(),
            descriptor=descriptor,
            stride=stride,
        )

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"study{ref.study.value}/sub{ref.participant}"
                    for ref in self.refs
                }
            )
        )

    def iter_examples(self) -> Iterator[WindowExample]:
        """Load one verified session at a time, never the whole release.

        Every session is measured against the descriptor before any of its
        windows are yielded. This is the path that needs it: the two
        undocumented preprocessing parameters are recoverable from these
        arrays and from nowhere else, and a corpus built under the wrong one
        trains a model that a correct live pipeline can no longer feed. Doing
        it per session rather than once for the dataset is deliberate, since
        the offset is a property of a device and a recording rather than of
        the release.
        """
        for ref in self.refs:
            session = self.dataset.load(ref)
            session.verify_preprocessing(self.descriptor)
            landmarks = self.dataset.landmarks(ref)
            yield from PoseWindows.from_session(
                session,
                landmarks,
                descriptor=self.descriptor,
                stride=self.stride,
            )

    def materialize(self, *, limit: int | None = None) -> TrainingCorpus:
        """Collect a deliberate finite experiment from the lazy source.

        ``limit`` is a guardrail for development, not a sampling policy.  It
        preserves source order so a caller cannot accidentally turn an ordered
        within-session selection into a random split by materializing it.
        """
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when supplied")
        selected: list[WindowExample] = []
        for example in self.iter_examples():
            selected.append(example)
            if limit is not None and len(selected) >= limit:
                break
        return TrainingCorpus.from_examples(selected, descriptor=self.descriptor)
