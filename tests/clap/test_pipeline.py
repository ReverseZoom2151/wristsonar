"""End to end: recording in, contact classes and a stated bit budget out."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from tests.clap.synth import SAMPLE_RATE, synth_feature_set, synth_recording

from wristsonar.clap.capacity import (
    arm_pose_bits,
    assert_within_capacity,
    capacity_measurement,
    channel_capacity,
)
from wristsonar.clap.configuration import (
    ClapMode,
    ConfigurationClassifier,
    Label,
    accuracy,
    accuracy_measurement,
    features_from_fit,
)
from wristsonar.clap.detect import detect_impulses
from wristsonar.clap.resonance import (
    cavity_volume_m3,
    fit_resonance,
)
from wristsonar.protocol import GroundTruth, Protocol, Split

_PROTOCOL = Protocol(
    split=Split.WITHIN_SESSION,
    dataset="synthetic-claps",
    dataset_version="0.0.1",
    ground_truth=GroundTruth.SYNTHETIC,
    subjects=1,
    notes="offline synthesis from Peltola centre frequencies",
)


def test_a_full_recording_becomes_labelled_contact_classes() -> None:
    rng = np.random.default_rng(808)
    train_x, train_y = synth_feature_set(rng, 30)
    model = ConfigurationClassifier.fit(train_x, train_y)

    modes = list(ClapMode) * 3
    rng.shuffle(modes)  # type: ignore[arg-type]
    signal, times, truth = synth_recording(modes, rng, gap_s=0.4)

    events = detect_impulses(signal, SAMPLE_RATE)
    assert len(events) == len(times)

    features = [
        features_from_fit(fit_resonance(event.response, SAMPLE_RATE))
        for event in events
    ]
    labels: list[Label] = list(truth)
    confusion = model.confusion_matrix(features, labels)

    measurement = accuracy_measurement(confusion, _PROTOCOL)
    assert measurement.value > model.chance_accuracy * 2.0
    assert measurement.samples == len(events)

    bits = capacity_measurement(confusion, _PROTOCOL)
    assert bits.unit == "bits"
    assert bits.value > 0.5


def test_the_pipeline_will_not_let_a_caller_claim_a_pose() -> None:
    """The whole reason capacity.py exists."""
    rng = np.random.default_rng(909)
    train_x, train_y = synth_feature_set(rng, 25)
    test_x, test_y = synth_feature_set(rng, 25)
    model = ConfigurationClassifier.fit(train_x, train_y)

    summary = channel_capacity(model.confusion_matrix(test_x, test_y))

    assert summary.mutual_information_bits < 2.0
    assert_within_capacity(
        summary.mutual_information_bits,
        summary.mutual_information_bits,
        what="contact class",
    )
    with pytest.raises(Exception, match="short by"):
        assert_within_capacity(
            arm_pose_bits(),
            summary.mutual_information_bits,
            what="7-DOF arm pose",
        )


def test_recovered_cavity_volume_orders_claps_by_configuration() -> None:
    """Volume is degenerate in the vent, but its ordering still survives."""
    rng = np.random.default_rng(1010)
    volumes: dict[ClapMode, float] = {}
    for mode in (ClapMode.A1PLUS, ClapMode.A1, ClapMode.A2, ClapMode.P3):
        samples = []
        for _ in range(15):
            signal, _, _ = synth_recording([mode], rng, lead_s=0.3, tail_s=0.3)
            event = detect_impulses(signal, SAMPLE_RATE)[0]
            fit = fit_resonance(event.response, SAMPLE_RATE)
            samples.append(cavity_volume_m3(fit.centre_frequency_hz))
        volumes[mode] = float(np.median(samples))

    ordered = [
        volumes[ClapMode.A1PLUS],
        volumes[ClapMode.A1],
        volumes[ClapMode.A2],
        volumes[ClapMode.P3],
    ]
    assert all(a > b for a, b in itertools.pairwise(ordered))


def test_accuracy_and_bits_agree_about_a_useless_classifier() -> None:
    rng = np.random.default_rng(1111)
    test_x, test_y = synth_feature_set(rng, 20)
    model = ConfigurationClassifier.fit(*synth_feature_set(rng, 20))

    shuffled: list[Label] = list(test_y)
    rng.shuffle(shuffled)  # type: ignore[arg-type]
    confusion = model.confusion_matrix(test_x, shuffled)

    assert accuracy(confusion) < 0.5
    assert channel_capacity(confusion).mutual_information_bits < 0.2
