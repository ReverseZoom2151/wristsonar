"""Onset detection: finds impulses, ignores everything that sustains."""

from __future__ import annotations

import numpy as np

from wristsonar.clap.configuration import ClapMode
from wristsonar.clap.detect import detect_impulses

from .synth import SAMPLE_RATE, synth_recording

_MODES = [
    ClapMode.A1PLUS,
    ClapMode.A1,
    ClapMode.P2,
    ClapMode.A1MINUS,
    ClapMode.P1,
    ClapMode.A2,
    ClapMode.A3,
    ClapMode.P3,
]


def test_finds_every_impulse_in_a_noisy_train() -> None:
    rng = np.random.default_rng(20250816)
    signal, times, _ = synth_recording(_MODES, rng)

    events = detect_impulses(signal, SAMPLE_RATE)

    assert len(events) == len(times)
    for event, expected in zip(events, times, strict=True):
        assert abs(event.time_s - expected) < 0.004


def test_no_spurious_events_in_pure_noise() -> None:
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, 5 * SAMPLE_RATE)

    assert detect_impulses(noise, SAMPLE_RATE) == []


def test_rejects_a_sustained_tone_with_a_sharp_attack() -> None:
    """A rising edge is not an impulse. The decay test is what separates them."""
    rng = np.random.default_rng(11)
    n = 4 * SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    signal = rng.normal(0.0, 0.01, n)
    start = SAMPLE_RATE
    stop = 3 * SAMPLE_RATE
    signal[start:stop] += 0.5 * np.sin(2.0 * np.pi * 800.0 * t[: stop - start])

    events = detect_impulses(signal, SAMPLE_RATE)

    assert events == []


def test_rejects_a_sustained_noise_burst() -> None:
    """A hand dryer switching on has an onset and no decay."""
    rng = np.random.default_rng(12)
    n = 4 * SAMPLE_RATE
    signal = rng.normal(0.0, 0.01, n)
    signal[SAMPLE_RATE : 3 * SAMPLE_RATE] += rng.normal(0.0, 0.4, 2 * SAMPLE_RATE)

    assert detect_impulses(signal, SAMPLE_RATE) == []


def test_survives_a_noise_floor_that_moves_forty_db() -> None:
    """A fixed threshold cannot do this, which is why the floor is adaptive."""
    rng = np.random.default_rng(3)
    signal, times, _ = synth_recording(_MODES, rng, noise_rms=0.002, noise_ramp_db=40.0)

    events = detect_impulses(signal, SAMPLE_RATE)

    assert len(events) == len(times)
    for event, expected in zip(events, times, strict=True):
        assert abs(event.time_s - expected) < 0.004


def test_window_starts_before_the_peak_and_response_starts_at_it() -> None:
    rng = np.random.default_rng(5)
    signal, _, _ = synth_recording([ClapMode.A1], rng)

    event = detect_impulses(signal, SAMPLE_RATE)[0]

    assert event.pre_samples > 0
    assert event.window.size > event.pre_samples
    assert event.response.size == event.window.size - event.pre_samples
    assert event.decay_db > 6.0
    assert event.rise_db > 6.0


def test_detection_is_scale_invariant() -> None:
    rng = np.random.default_rng(9)
    signal, times, _ = synth_recording(_MODES, rng)

    loud = detect_impulses(signal * 1000.0, SAMPLE_RATE)
    quiet = detect_impulses(signal * 0.001, SAMPLE_RATE)

    assert [e.index for e in loud] == [e.index for e in quiet]
    assert len(loud) == len(times)
