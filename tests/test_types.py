"""The invariants the containers promise, and the arithmetic in their properties.

These types are plain by design, so the only things worth testing are the two
that are not plain: the guards in `__post_init__`, which are the last line of
defence against an array of the wrong shape reaching the physics, and the
derived properties, which are the place a factor of two would hide. Every
number a docstring in this module claims is checked here rather than trusted,
because a docstring that drifts from its code in this file drifts through the
whole package.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristsonar.types import (
    JOINT_NAMES,
    N_JOINTS,
    SPEED_OF_SOUND,
    ChirpConfig,
    EchoProfile,
    HandPose,
)


class TestChirpConfigGuards:
    def test_a_downward_sweep_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sweep upward"):
            ChirpConfig(f_start=21_000.0, f_end=18_000.0)

    def test_a_zero_width_sweep_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sweep upward"):
            ChirpConfig(f_start=19_000.0, f_end=19_000.0)

    def test_a_non_positive_duration_is_refused(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            ChirpConfig(duration_s=0.0)

    def test_a_non_positive_sample_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sample rate must be positive"):
            ChirpConfig(sample_rate=0)

    def test_a_sweep_past_nyquist_is_refused(self) -> None:
        """The one guard that catches a plausible configuration rather than a typo.

        A 23 kHz top at 44.1 kHz sampling is a sweep somebody would actually
        write down, and it aliases. The reflected image lands back in band and
        the matched filter correlates against it happily, so nothing
        downstream would complain.
        """
        with pytest.raises(ValueError, match="Nyquist"):
            ChirpConfig(f_start=18_000.0, f_end=23_000.0, sample_rate=44_100)

    def test_a_sweep_exactly_at_nyquist_is_allowed(self) -> None:
        """The boundary is inclusive, and that is the right side to be on.

        A component exactly at Nyquist does not fold to a different frequency,
        it merely samples degenerately, and refusing it would reject the
        24 kHz top of a 48 kHz configuration for no acoustic reason.
        """
        assert ChirpConfig(f_end=24_000.0, sample_rate=48_000).f_end == 24_000.0


class TestChirpConfigArithmetic:
    def test_the_defaults_are_the_documented_ones(self) -> None:
        config = ChirpConfig()
        assert config.bandwidth == 3_000.0
        assert config.n_samples == 600
        assert config.frame_rate == pytest.approx(80.0)

    def test_range_resolution_is_c_over_twice_the_bandwidth(self) -> None:
        config = ChirpConfig()
        assert config.range_resolution == pytest.approx(
            SPEED_OF_SOUND / (2.0 * config.bandwidth)
        )
        assert config.range_resolution == pytest.approx(0.0572, abs=1e-4)

    def test_max_unambiguous_range_is_one_way(self) -> None:
        """The convention that the whole package hangs on, checked as a number.

        Half the distance sound covers in one frame, because the path is out
        and back. A round-trip reading here would double every range in the
        system and nothing would raise.
        """
        config = ChirpConfig()
        travelled = SPEED_OF_SOUND * config.duration_s
        assert config.max_unambiguous_range == pytest.approx(travelled / 2.0)
        assert config.max_unambiguous_range == pytest.approx(2.144, abs=1e-3)

    def test_the_bin_grid_spans_exactly_the_unambiguous_range(self) -> None:
        """Two derivations of the same quantity, one from fs and one from T.

        `n_samples * c / (2 fs)` and `c T / 2` are the same number only if the
        factor of two is applied once in each. Pinned here as well as in the
        signal layer, since this is the side that would be edited by someone
        who never opened `matched_filter`.
        """
        for config in (
            ChirpConfig(),
            ChirpConfig(duration_s=0.02, sample_rate=44_100, f_end=20_000.0),
            ChirpConfig(duration_s=0.005, sample_rate=96_000),
        ):
            bin_metres = SPEED_OF_SOUND / (2.0 * config.sample_rate)
            assert config.n_samples * bin_metres == pytest.approx(
                config.max_unambiguous_range, rel=1e-9
            )

    def test_wavelength_is_taken_at_the_band_centre(self) -> None:
        config = ChirpConfig()
        assert config.wavelength_at_centre == pytest.approx(
            SPEED_OF_SOUND / 19_500.0
        )
        assert config.wavelength_at_centre == pytest.approx(0.0176, abs=1e-4)

    def test_resolution_is_far_coarser_than_the_bin_grid(self) -> None:
        """The distinction the project exists to keep visible, as an inequality."""
        config = ChirpConfig()
        bin_metres = SPEED_OF_SOUND / (2.0 * config.sample_rate)
        assert config.range_resolution > 10.0 * bin_metres


class TestEchoProfileGuards:
    def test_a_two_dimensional_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one dimensional"):
            EchoProfile(
                samples=np.zeros((4, 4), dtype=np.float32),
                bin_metres=0.0036,
                timestamp_s=0.0,
            )

    def test_a_non_positive_bin_size_is_refused(self) -> None:
        for bad in (0.0, -0.0036):
            with pytest.raises(ValueError, match="bin_metres must be positive"):
                EchoProfile(
                    samples=np.zeros(4, dtype=np.float32),
                    bin_metres=bad,
                    timestamp_s=0.0,
                )

    def test_a_negative_offset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            EchoProfile(
                samples=np.zeros(4, dtype=np.float32),
                bin_metres=0.0036,
                timestamp_s=0.0,
                range_offset_m=-1e-9,
            )

    def test_an_empty_profile_is_allowed(self) -> None:
        """Legitimate rather than a bug: a recording shorter than one frame.

        Refusing it here would push a special case into every caller that
        segments a stream.
        """
        empty = EchoProfile(
            samples=np.zeros(0, dtype=np.float32),
            bin_metres=0.0036,
            timestamp_s=0.0,
        )
        assert empty.n_bins == 0
        assert empty.max_range == 0.0


class TestEchoProfileRanges:
    def _profile(self, offset: float = 0.0) -> EchoProfile:
        return EchoProfile(
            samples=np.zeros(10, dtype=np.float32),
            bin_metres=0.01,
            timestamp_s=0.0,
            range_offset_m=offset,
        )

    def test_range_of_bin_accepts_a_sub_bin_index(self) -> None:
        """Sub-bin interpolation returns a float index, so this must not round."""
        profile = self._profile(0.02)
        assert profile.range_of_bin(3.5) == pytest.approx(0.055)

    def test_near_and_max_range_bracket_every_bin(self) -> None:
        profile = self._profile(0.02)
        assert profile.near_range == profile.range_of_bin(0)
        assert profile.max_range == profile.range_of_bin(profile.n_bins)
        assert profile.max_range > profile.range_of_bin(profile.n_bins - 1)

    def test_cropping_shifts_the_window_without_moving_a_reflector(self) -> None:
        samples = np.zeros(100, dtype=np.float32)
        samples[70] = 1.0
        profile = EchoProfile(samples=samples, bin_metres=0.01, timestamp_s=0.0)
        truth = profile.range_of_bin(70)

        cropped = profile.crop(0.20, 0.90)
        assert cropped.range_of_bin(int(np.argmax(cropped.samples))) == pytest.approx(
            truth
        )
        assert cropped.near_range == pytest.approx(0.20)

    def test_a_crop_wider_than_the_profile_is_a_no_op(self) -> None:
        profile = self._profile()
        widened = profile.crop(-1.0, 100.0)
        assert widened.n_bins == profile.n_bins
        assert widened.range_offset_m == 0.0


class TestHandPose:
    def test_the_joint_names_are_the_mediapipe_twenty_one(self) -> None:
        assert N_JOINTS == 21
        assert JOINT_NAMES[0] == "wrist"
        assert len(set(JOINT_NAMES)) == N_JOINTS

    def test_a_wrong_shaped_joint_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="joint array"):
            HandPose(joints=np.zeros((20, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="joint array"):
            HandPose(joints=np.zeros((N_JOINTS, 2), dtype=np.float32))

    def test_an_unknown_handedness_is_refused(self) -> None:
        with pytest.raises(ValueError, match="handedness"):
            HandPose(
                joints=np.zeros((N_JOINTS, 3), dtype=np.float32), handedness="either"
            )

    def test_joints_are_looked_up_by_name(self) -> None:
        joints = np.arange(N_JOINTS * 3, dtype=np.float32).reshape(N_JOINTS, 3)
        pose = HandPose(joints=joints)
        np.testing.assert_array_equal(pose.joint("wrist"), joints[0])
        np.testing.assert_array_equal(
            pose.joint("pinky_tip"), joints[N_JOINTS - 1]
        )

    def test_an_unknown_joint_name_raises_key_error(self) -> None:
        pose = HandPose(joints=np.zeros((N_JOINTS, 3), dtype=np.float32))
        with pytest.raises(KeyError, match="unknown joint"):
            pose.joint("elbow")
