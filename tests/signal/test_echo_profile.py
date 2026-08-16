"""Tests for the profile layer, including the range convention under cropping.

This module carried the differential claim and the crop helper with no tests
at all, which is how a crop that silently shifted every range survived. The
convention under test throughout: `bin_metres` is one-way reflector distance,
and a cropped profile records where its bin zero now sits so that a range
computed from it still means the same thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristsonar.signal.echo_profile import (
    Normalisation,
    differential_profiles,
    normalise,
    profiles_from_recording,
    segment_frames,
)
from wristsonar.signal.matched_filter import bin_metres_for
from wristsonar.signal.simulate import Reflector, simulate_recording
from wristsonar.types import ChirpConfig, EchoProfile

SAMPLE_RATE = 48_000
BIN_M = bin_metres_for(SAMPLE_RATE)


def _spike_profile(range_m: float, n_bins: int = 600) -> EchoProfile:
    samples = np.zeros(n_bins, dtype=np.float32)
    samples[round(range_m / BIN_M)] = 1.0
    return EchoProfile(samples=samples, bin_metres=BIN_M, timestamp_s=0.0)


class TestCropPreservesRange:
    """The defect this file exists for."""

    def test_a_cropped_peak_reports_the_same_distance(self) -> None:
        profile = _spike_profile(0.100)
        before = profile.range_of_bin(int(np.argmax(profile.samples)))

        cropped = profile.crop(0.02, 0.30)
        after = cropped.range_of_bin(int(np.argmax(cropped.samples)))

        assert before == pytest.approx(0.100, abs=BIN_M)
        # Without the recorded offset this read 0.082 m, a 1.8 cm near-bias
        # on every range in the system.
        assert after == pytest.approx(before, abs=1e-9)

    def test_bin_zero_of_a_cropped_profile_is_the_near_edge(self) -> None:
        cropped = _spike_profile(0.100).crop(0.02, 0.30)
        assert cropped.near_range == pytest.approx(0.02, abs=BIN_M)
        assert cropped.range_of_bin(0) == cropped.near_range

    def test_cropping_twice_accumulates_the_offset(self) -> None:
        profile = _spike_profile(0.100)
        once = profile.crop(0.02, 0.30)
        twice = once.crop(0.05, 0.20)

        peak = twice.range_of_bin(int(np.argmax(twice.samples)))
        assert peak == pytest.approx(0.100, abs=BIN_M)
        assert twice.near_range > once.near_range

    def test_max_range_stays_meaningful_after_cropping(self) -> None:
        cropped = _spike_profile(0.100).crop(0.02, 0.30)
        assert cropped.max_range == pytest.approx(
            cropped.near_range + cropped.n_bins * cropped.bin_metres
        )
        assert cropped.max_range <= 0.30 + cropped.bin_metres

    def test_an_uncropped_profile_has_no_offset(self) -> None:
        profile = _spike_profile(0.100)
        assert profile.range_offset_m == 0.0
        assert profile.range_of_bin(3) == pytest.approx(3 * BIN_M)

    def test_a_window_selecting_no_bins_is_refused(self) -> None:
        # Narrower than one bin, so both edges land on the same index.
        with pytest.raises(ValueError, match="selects no bins"):
            _spike_profile(0.100).crop(0.0001, 0.0002)

    def test_an_inverted_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            _spike_profile(0.100).crop(0.30, 0.02)

    def test_a_negative_offset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            EchoProfile(
                samples=np.zeros(4, dtype=np.float32),
                bin_metres=BIN_M,
                timestamp_s=0.0,
                range_offset_m=-0.01,
            )


class TestDifferential:
    """The module docstring's central claim, previously unpinned."""

    def test_a_static_reflector_is_suppressed_and_a_moving_one_survives(
        self,
    ) -> None:
        config = ChirpConfig(sample_rate=SAMPLE_RATE)
        recording = simulate_recording(
            config,
            [
                [Reflector(range_m=0.12, amplitude=1.0)],
                [Reflector(range_m=0.14, amplitude=1.0)],
            ],
            static=[Reflector(range_m=0.05, amplitude=1.0)],
        )
        frames = profiles_from_recording(recording, config)

        diffs = differential_profiles(frames)
        assert len(diffs) == 1
        diff = diffs[0]
        assert diff.differential is True

        static_bin = round(0.05 / BIN_M)
        moved_bin = round(0.14 / BIN_M)
        static_residual = float(np.abs(diff.samples[static_bin]))
        moved_energy = float(
            np.max(np.abs(diff.samples[moved_bin - 4 : moved_bin + 5]))
        )
        assert static_residual < 0.25 * moved_energy

    def test_the_difference_is_timestamped_with_the_later_frame(self) -> None:
        a = EchoProfile(
            samples=np.zeros(8, dtype=np.float32), bin_metres=BIN_M, timestamp_s=1.0
        )
        b = EchoProfile(
            samples=np.ones(8, dtype=np.float32), bin_metres=BIN_M, timestamp_s=2.0
        )
        assert differential_profiles([a, b])[0].timestamp_s == 2.0

    def test_a_single_frame_yields_no_difference(self) -> None:
        only = EchoProfile(
            samples=np.zeros(8, dtype=np.float32), bin_metres=BIN_M, timestamp_s=0.0
        )
        assert differential_profiles([only]) == []

    def test_the_crop_offset_survives_differencing(self) -> None:
        """Differencing must not undo what cropping recorded.

        The natural order is crop then difference, and a difference that built
        its output with the default offset of zero would put every range in
        the differenced sequence back to reading low by the near edge, which
        is exactly the defect `range_offset_m` was added to close.
        """
        config = ChirpConfig(sample_rate=SAMPLE_RATE)
        recording = simulate_recording(
            config,
            [
                [Reflector(range_m=0.12, amplitude=1.0)],
                [Reflector(range_m=0.16, amplitude=1.0)],
            ],
        )
        cropped = [
            p.crop(0.02, 0.30) for p in profiles_from_recording(recording, config)
        ]

        diff = differential_profiles(cropped)[0]
        assert diff.range_offset_m == cropped[0].range_offset_m
        assert diff.range_offset_m > 0.015

        arrived = int(np.argmax(diff.samples))
        # The apex of a signed difference sits between the bin the reflector
        # arrived in and the one it left, so this is deliberately a window
        # rather than a point. What is exact is the offset arithmetic below.
        assert 0.10 < diff.range_of_bin(arrived) < 0.20
        assert diff.range_of_bin(arrived) - arrived * BIN_M == pytest.approx(
            diff.range_offset_m
        )

    def test_profiles_cropped_differently_are_refused(self) -> None:
        """Bin k of two differently cropped profiles is two different distances.

        Subtracting them is not a difference in time, so it is refused rather
        than producing a plausible array with no meaning. Same length and same
        bin size, so only the offset separates them and only the offset check
        can catch it.
        """
        samples = np.zeros(64, dtype=np.float32)
        near = EchoProfile(
            samples=samples, bin_metres=BIN_M, timestamp_s=0.0, range_offset_m=0.02
        )
        far = EchoProfile(
            samples=samples, bin_metres=BIN_M, timestamp_s=1.0, range_offset_m=0.05
        )
        with pytest.raises(ValueError, match="different range calibrations"):
            differential_profiles([near, far])


class TestNormaliseAndSegment:
    def test_an_all_zero_frame_survives_normalisation(self) -> None:
        zeros = np.zeros(16, dtype=np.float64)
        for mode in Normalisation:
            assert np.array_equal(normalise(zeros, mode), zeros)

    def test_peak_normalisation_reaches_unit_magnitude(self) -> None:
        values = np.array([0.0, -4.0, 2.0], dtype=np.float64)
        assert float(np.max(np.abs(normalise(values, Normalisation.PEAK)))) == (
            pytest.approx(1.0)
        )

    def test_segmentation_drops_a_trailing_partial_frame(self) -> None:
        config = ChirpConfig(sample_rate=SAMPLE_RATE)
        n = config.n_samples
        recording = np.zeros(2 * n + n // 3, dtype=np.float64)
        assert segment_frames(recording, config).shape == (2, n)
