"""Classification, and the case for merging Repp's eight classes down."""

from __future__ import annotations

import numpy as np
import pytest

from wristsonar.clap.configuration import (
    CONFUSABLE_PAIRS,
    DEFAULT_TAXONOMY,
    MERGE_MAP,
    MODE_PROTOTYPES,
    ClapFeatures,
    ClapMode,
    ConfigurationClassifier,
    ContactClass,
    Label,
    Taxonomy,
    accuracy,
    accuracy_measurement,
    merge_mode,
    prototype_classifier,
)
from wristsonar.protocol import GroundTruth, Protocol, Split

from .synth import synth_feature_set

_TRAIN_PER_MODE = 30
_TEST_PER_MODE = 30

_PROTOCOL = Protocol(
    split=Split.WITHIN_SESSION,
    dataset="synthetic-claps",
    dataset_version="0.0.1",
    ground_truth=GroundTruth.SYNTHETIC,
    subjects=1,
    notes="Peltola centre frequencies, second-order resonators, offline",
)


Dataset = tuple[list[ClapFeatures], list[Label]]


def _split(seed: int) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    train = synth_feature_set(rng, _TRAIN_PER_MODE)
    test = synth_feature_set(rng, _TEST_PER_MODE)
    return train, test


def _keep(dataset: Dataset, modes: set[ClapMode]) -> Dataset:
    features, labels = dataset
    kept = [
        (f, y)
        for f, y in zip(features, labels, strict=True)
        if isinstance(y, ClapMode) and y in modes
    ]
    return [f for f, _ in kept], [y for _, y in kept]


def test_the_default_taxonomy_is_the_merged_one() -> None:
    assert DEFAULT_TAXONOMY is Taxonomy.MERGED


def test_every_repp_mode_maps_into_the_merged_taxonomy() -> None:
    assert set(MERGE_MAP) == set(ClapMode)
    assert set(MERGE_MAP.values()) == set(ContactClass)


def test_the_merge_absorbs_every_reported_confusion_pair() -> None:
    """The merge is justified by the confusions, not by taste."""
    for left, right in CONFUSABLE_PAIRS:
        assert merge_mode(left) is merge_mode(right)


def test_prototype_frequencies_order_by_cavity_size() -> None:
    """Repp: smaller enclosed cavity, higher resonance."""
    ordered = [
        ClapMode.A1PLUS,
        ClapMode.A1,
        ClapMode.P2,
        ClapMode.A1MINUS,
        ClapMode.P1,
        ClapMode.A2,
        ClapMode.A3,
        ClapMode.P3,
    ]
    frequencies = [MODE_PROTOTYPES[mode][0] for mode in ordered]

    assert frequencies == sorted(frequencies)
    assert frequencies[0] == pytest.approx(500.0)
    assert frequencies[-1] == pytest.approx(1562.0)


def test_merged_classification_beats_chance_by_a_wide_margin() -> None:
    (train_x, train_y), (test_x, test_y) = _split(101)
    model = ConfigurationClassifier.fit(train_x, train_y)

    confusion = model.confusion_matrix(test_x, test_y)
    score = accuracy(confusion)

    assert model.chance_accuracy == pytest.approx(0.25)
    assert score > 0.6


def test_eight_class_classification_beats_chance_but_not_by_much() -> None:
    (train_x, train_y), (test_x, test_y) = _split(101)
    model = ConfigurationClassifier.fit(train_x, train_y, taxonomy=Taxonomy.EIGHT_MODE)

    score = accuracy(model.confusion_matrix(test_x, test_y))

    assert model.chance_accuracy == pytest.approx(0.125)
    assert score > 0.125 * 2.0
    assert score < 0.7


def test_merging_beats_eight_classes_on_the_confusable_pairs() -> None:
    """The core claim.

    Both classifiers see the same full training set. They are then asked
    only about claps drawn from the two pairs Jylha and Erkut reported as
    systematically confused. The eight-way answer on those is close to a coin
    toss; the merged answer is reliable, because the merge is drawn exactly
    where the acoustics stop separating.
    """
    train, test = _split(202)
    modes = {mode for pair in CONFUSABLE_PAIRS for mode in pair}
    hard_x, hard_y = _keep(test, modes)

    eight = ConfigurationClassifier.fit(
        train[0], train[1], taxonomy=Taxonomy.EIGHT_MODE
    )
    merged = ConfigurationClassifier.fit(train[0], train[1], taxonomy=Taxonomy.MERGED)

    eight_score = accuracy(eight.confusion_matrix(hard_x, hard_y))
    merged_score = accuracy(merged.confusion_matrix(hard_x, hard_y))

    assert merged_score > eight_score + 0.25
    assert merged_score > 0.8


def test_the_prototype_classifier_works_without_any_training_data() -> None:
    rng = np.random.default_rng(303)
    test_x, test_y = synth_feature_set(rng, 20)

    model = prototype_classifier()
    score = accuracy(model.confusion_matrix(test_x, test_y))

    assert score > model.chance_accuracy * 1.5


def test_features_are_two_dimensional() -> None:
    """A second-order fit has two free parameters after gain. No more."""
    assert ClapFeatures(1000.0, 200.0, energy=1.0).vector().shape == (2,)


def test_energy_is_carried_but_not_classified_on() -> None:
    quiet = ClapFeatures(1000.0, 200.0, energy=1e-6)
    loud = ClapFeatures(1000.0, 200.0, energy=1.0)
    model = prototype_classifier()

    assert model.predict(quiet) is model.predict(loud)


def test_posterior_is_a_distribution() -> None:
    model = prototype_classifier()
    posterior = model.predict_proba(ClapFeatures(900.0, 200.0))

    assert posterior.shape == (len(model.labels),)
    assert float(posterior.sum()) == pytest.approx(1.0)
    assert bool(np.all(posterior >= 0.0))


def test_a_merged_class_cannot_be_expanded_back_to_a_repp_mode() -> None:
    model = ConfigurationClassifier.fit(
        [ClapFeatures(*MODE_PROTOTYPES[m]) for m in ClapMode],
        list(ClapMode),
        taxonomy=Taxonomy.EIGHT_MODE,
    )

    with pytest.raises(ValueError, match="deliberately lossy"):
        model.confusion_matrix([ClapFeatures(900.0, 200.0)], [ContactClass.FLAT])


def test_accuracy_is_reported_with_its_protocol() -> None:
    (train_x, train_y), (test_x, test_y) = _split(404)
    model = ConfigurationClassifier.fit(train_x, train_y)
    confusion = model.confusion_matrix(test_x, test_y)

    measurement = accuracy_measurement(confusion, _PROTOCOL)

    assert measurement.unit == "fraction correct"
    assert measurement.samples == int(confusion.sum())
    assert not measurement.is_honest
    assert "synthetic" in str(measurement)
