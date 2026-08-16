"""The bit budget: what an impulse carries, and what a pose costs."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from wristsonar.clap.capacity import (
    ARM_DOF,
    OverclaimError,
    arm_pose_bits,
    assert_within_capacity,
    capacity_measurement,
    channel_capacity,
    mutual_information_bits,
    pose_gap,
)
from wristsonar.protocol import GroundTruth, Protocol, Split

_PROTOCOL = Protocol(
    split=Split.CROSS_USER,
    dataset="synthetic-claps",
    dataset_version="0.0.1",
    ground_truth=GroundTruth.SYNTHETIC,
    subjects=4,
)


def _random_confusion(n_classes: int, n_samples: int, seed: int) -> NDArray[np.int64]:
    """Predictions drawn independently of the truth."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, n_classes, n_samples)
    predicted = rng.integers(0, n_classes, n_samples)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(truth, predicted, strict=True):
        matrix[t, p] += 1
    return matrix


def test_a_random_classifier_carries_almost_no_information() -> None:
    confusion = _random_confusion(4, 20_000, seed=1)

    bits = mutual_information_bits(confusion)

    assert bits < 0.02


def test_a_perfect_classifier_carries_the_full_label_entropy() -> None:
    confusion = np.diag(np.full(4, 250, dtype=np.int64))

    assert mutual_information_bits(confusion) == pytest.approx(2.0)


def test_information_rises_with_classifier_quality() -> None:
    scores = []
    for correct in (25, 40, 60, 80, 97):
        off = (100 - correct) // 3
        confusion = np.full((4, 4), off, dtype=np.int64)
        np.fill_diagonal(confusion, correct)
        scores.append(mutual_information_bits(confusion))

    assert all(a < b for a, b in itertools.pairwise(scores))
    assert scores[0] < 0.05
    assert scores[-1] > 1.7


def test_accuracy_can_hide_what_bits_expose() -> None:
    """Same accuracy, different information, because errors have structure."""
    structured = np.array(
        [[60, 40, 0, 0], [40, 60, 0, 0], [0, 0, 60, 40], [0, 0, 40, 60]],
        dtype=np.int64,
    )
    diffuse = np.full((4, 4), 40, dtype=np.int64)
    np.fill_diagonal(diffuse, 60)

    assert np.trace(structured) == np.trace(diffuse)
    assert mutual_information_bits(structured) > mutual_information_bits(diffuse)


def test_channel_capacity_summarises_a_confusion_matrix() -> None:
    confusion = np.full((4, 4), 10, dtype=np.int64)
    np.fill_diagonal(confusion, 70)

    summary = channel_capacity(confusion)

    assert summary.max_bits == pytest.approx(2.0)
    assert summary.input_entropy_bits == pytest.approx(2.0)
    assert 0.0 < summary.efficiency < 1.0
    assert summary.accuracy == pytest.approx(0.7)
    assert summary.samples == 400
    assert summary.residual_entropy_bits > 0.0
    assert "bits" in summary.describe()


def test_an_arm_pose_costs_about_thirty_two_bits() -> None:
    bits = arm_pose_bits()

    assert bits == pytest.approx(32.1, abs=0.5)
    assert arm_pose_bits(dof=ARM_DOF, resolution_deg=10.0) < bits


def test_the_gap_between_a_clap_and_a_pose_is_structural() -> None:
    gap = pose_gap(1.8)

    assert not gap.is_feasible
    assert gap.deficit_bits > 28.0
    assert gap.events_required > 15.0
    assert gap.achievable_resolution_deg > 60.0
    assert "short by" in gap.describe()


def test_even_a_generous_bit_budget_does_not_close_the_gap() -> None:
    """Grant the whole impulse, all ten bits, to the arm. Still not close."""
    gap = pose_gap(10.0)

    assert not gap.is_feasible
    assert gap.deficit_bits > 20.0


def test_overclaiming_raises() -> None:
    with pytest.raises(OverclaimError, match="short by"):
        assert_within_capacity(arm_pose_bits(), 1.8, what="3D arm pose from one clap")


def test_a_claim_within_budget_passes() -> None:
    assert_within_capacity(1.5, 1.8, what="four-class contact configuration")


def test_bits_are_reported_with_their_protocol() -> None:
    confusion = np.full((4, 4), 10, dtype=np.int64)
    np.fill_diagonal(confusion, 70)

    measurement = capacity_measurement(confusion, _PROTOCOL)

    assert measurement.unit == "bits"
    assert measurement.samples == 400
    assert measurement.is_honest
    assert 0.0 < measurement.value < 2.0


def test_empty_and_malformed_matrices_are_rejected() -> None:
    with pytest.raises(ValueError):
        mutual_information_bits(np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        mutual_information_bits(np.zeros((2, 2, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        channel_capacity(np.ones((2, 3), dtype=np.int64))
