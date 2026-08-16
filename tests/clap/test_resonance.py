"""Resonator fitting and the Helmholtz reading of the result."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from tests.clap.synth import SAMPLE_RATE, resonator_impulse_response, synth_clap

from wristsonar.clap.configuration import MODE_PROTOTYPES, ClapMode
from wristsonar.clap.resonance import (
    ResonanceFitError,
    cavity_volume_m3,
    fit_resonance,
    helmholtz_frequency_hz,
    helmholtz_geometry_ratio,
)
from wristsonar.types import SPEED_OF_SOUND


@pytest.mark.parametrize("frequency", [500.0, 794.0, 1063.0, 1562.0])
@pytest.mark.parametrize("bandwidth", [110.0, 240.0])
def test_recovers_a_known_resonance(frequency: float, bandwidth: float) -> None:
    response = resonator_impulse_response(
        frequency, bandwidth, SAMPLE_RATE, int(0.06 * SAMPLE_RATE)
    )

    fit = fit_resonance(response, SAMPLE_RATE)

    assert fit.centre_frequency_hz == pytest.approx(frequency, rel=0.02)
    assert fit.bandwidth_hz == pytest.approx(bandwidth, rel=0.10)
    assert fit.decay_rate_s == pytest.approx(math.pi * bandwidth, rel=0.10)
    assert 0.0 < fit.pole_radius < 1.0
    assert fit.residual < 0.05


def test_recovers_a_known_decay_rate_in_t60() -> None:
    bandwidth = 150.0
    response = resonator_impulse_response(
        1000.0, bandwidth, SAMPLE_RATE, int(0.06 * SAMPLE_RATE)
    )

    fit = fit_resonance(response, SAMPLE_RATE)

    expected_t60 = math.log(1000.0) / (math.pi * bandwidth)
    assert fit.t60_s == pytest.approx(expected_t60, rel=0.10)


def test_survives_additive_noise() -> None:
    rng = np.random.default_rng(41)
    clap = synth_clap(ClapMode.A1, rng)
    noisy = clap.samples + rng.normal(0.0, 0.01, clap.samples.size)

    fit = fit_resonance(noisy, SAMPLE_RATE)

    assert fit.centre_frequency_hz == pytest.approx(clap.centre_frequency_hz, rel=0.06)


def test_higher_order_still_picks_the_dominant_mode() -> None:
    """Order 4 has spare poles. The residue-weighted pick must ignore them."""
    response = resonator_impulse_response(
        900.0, 160.0, SAMPLE_RATE, int(0.06 * SAMPLE_RATE)
    )

    fit = fit_resonance(response, SAMPLE_RATE, order=4)

    assert fit.centre_frequency_hz == pytest.approx(900.0, rel=0.05)


def test_rejects_a_resonance_outside_the_accepted_band() -> None:
    response = resonator_impulse_response(
        6000.0, 200.0, SAMPLE_RATE, int(0.06 * SAMPLE_RATE)
    )

    with pytest.raises(ResonanceFitError):
        fit_resonance(response, SAMPLE_RATE)


def test_rejects_a_silent_window() -> None:
    with pytest.raises(ResonanceFitError):
        fit_resonance(np.zeros(2000), SAMPLE_RATE)


def test_cavity_volume_is_monotone_decreasing_in_frequency() -> None:
    frequencies = sorted(f for f, _ in MODE_PROTOTYPES.values())
    volumes = [cavity_volume_m3(f) for f in frequencies]

    assert all(a > b for a, b in itertools.pairwise(volumes))
    assert volumes[0] > volumes[-1] * 5.0


def test_helmholtz_round_trips() -> None:
    volume = 6.0e-5
    area = 3.0e-4
    length = 0.010

    frequency = helmholtz_frequency_hz(volume, area, length)
    recovered = cavity_volume_m3(frequency, neck_area_m2=area, neck_length_m=length)

    assert recovered == pytest.approx(volume, rel=1e-9)


def test_the_geometry_ratio_is_what_the_resonance_identifies() -> None:
    assert helmholtz_geometry_ratio(1000.0) == pytest.approx(
        (2.0 * math.pi * 1000.0 / SPEED_OF_SOUND) ** 2
    )


def test_volume_is_degenerate_in_the_assumed_vent() -> None:
    """The documented alias, asserted so nobody quietly forgets it.

    A large cavity with a large vent and a small cavity with a small vent
    produce the same frequency. Doubling the assumed neck area doubles the
    reported volume with no change to the recording.
    """
    frequency = 900.0
    small_vent = cavity_volume_m3(
        frequency, neck_area_m2=1.5e-4, neck_length_m=0.010, neck_radius_m=0.0
    )
    large_vent = cavity_volume_m3(
        frequency, neck_area_m2=3.0e-4, neck_length_m=0.010, neck_radius_m=0.0
    )

    assert large_vent == pytest.approx(2.0 * small_vent, rel=1e-9)


def test_estimated_volume_is_physically_plausible_for_a_cupped_hand() -> None:
    """Sanity, not calibration: a very cupped clap should land in tens of mL."""
    volume = cavity_volume_m3(500.0)

    assert 1.0e-5 < volume < 5.0e-4
