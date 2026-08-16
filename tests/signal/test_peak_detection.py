"""Peak detection on a cropped profile, which is the shape it actually meets.

Every other test in this layer feeds the detector a full 600 bin profile, and
on a full profile the noise floor question is uninteresting because a hand
occupies a few tens of bins out of six hundred. The recommended pipeline crops
to 2 cm to 30 cm before anything else looks at the profile, and that window is
78 bins, about five range-resolution cells, most of which a hand fills. The
estimator has to survive that, and this module is where it is measured.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from wristsonar.signal import (
    Reflector,
    bin_metres_for,
    matched_filter,
    noise_floor,
    peak_ranges,
    peak_ranges_of,
    simulate_frame,
    strongest_peak,
    strongest_peak_of,
)
from wristsonar.signal.echo_profile import profile_from_frame
from wristsonar.types import ChirpConfig, EchoProfile

HAND_NEAR_M = 0.02
HAND_FAR_M = 0.30

HAND_SCENE = (
    Reflector(0.05, 1.0),
    Reflector(0.11, 0.7),
    Reflector(0.18, 0.5),
)
"""Palm, knuckles and fingertips, spread across the recommended window.

Three point reflectors is the crudest thing that is still hand-shaped, and it
is enough: their mainlobes and skirts cover most of the crop, which is the
condition a whole-profile median estimator cannot survive.
"""


def _hand_profile(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> EchoProfile:
    frame = simulate_frame(
        config, list(HAND_SCENE), snr_db=20.0, rng=np.random.default_rng(3)
    )
    return profile_from_frame(frame, reference, config.sample_rate, 0.0)


class TestNoiseFloorSurvivesTheRecommendedCrop:
    """The defect this module exists for.

    Before the fix the floor was the median of whatever it was handed. On the
    hand scene above that was 0.63 against a peak of 1.05, so peak over floor
    was 1.7, the height threshold sat above the peak, `peak_ranges` returned
    nothing and `strongest_peak` returned None. On the uncropped profile of
    the same frame the median was 0.0054 and peak over floor was 193, so the
    detector worked perfectly right up until the crop the package recommends.
    """

    def test_the_median_still_collapses_on_the_crop(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """The defect itself, stated as arithmetic rather than as history.

        This is not testing the fix. It is pinning the reason the fix was
        needed, so that anyone tempted to go back to a plain median can see
        the number they would be choosing.
        """
        profile = _hand_profile(config, reference)
        cropped = profile.crop(HAND_NEAR_M, HAND_FAR_M)

        full_median = float(np.median(profile.samples))
        crop_median = float(np.median(cropped.samples))
        crop_peak = float(np.max(cropped.samples))

        assert crop_median > 40.0 * full_median
        assert crop_peak / crop_median < 2.0
        assert crop_peak / full_median > 100.0

    def test_a_hand_in_the_cropped_window_is_still_detected(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """The regression. Silence here is the bug."""
        cropped = _hand_profile(config, reference).crop(HAND_NEAR_M, HAND_FAR_M)

        floor = noise_floor(cropped.samples.astype(np.float64))
        peak = float(np.max(cropped.samples))
        assert peak / floor > 8.0

        assert len(peak_ranges_of(cropped)) >= 1
        assert strongest_peak_of(cropped) is not None

    def test_the_cropped_and_uncropped_profiles_agree_on_the_range(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """Cropping is a change of window, not a change of scene.

        Both the floor estimate and the offset arithmetic have to be right for
        this to hold, and either being wrong moves the answer by centimetres.
        """
        profile = _hand_profile(config, reference)
        cropped = profile.crop(HAND_NEAR_M, HAND_FAR_M)

        full_peak = strongest_peak_of(profile)
        crop_peak = strongest_peak_of(cropped)
        assert full_peak is not None
        assert crop_peak is not None
        assert crop_peak.range_m == pytest.approx(
            full_peak.range_m, abs=profile.bin_metres
        )
        # Near the strongest reflector rather than on it: the 5 cm and 11 cm
        # returns are inside one 5.7 cm resolution cell of each other, so they
        # interfere and the joint apex sits between them. That is physics, not
        # an estimator error, and the tolerance says so.
        assert crop_peak.range_m == pytest.approx(0.05, abs=0.015)


def _noise_false_positive_rates(
    config: ChirpConfig,
    reference: NDArray[np.complex128],
    *,
    crop: bool,
    trials: int = 200,
) -> tuple[int, int]:
    """False positives under the old median floor and under the current one.

    Reported as a pair because the number that matters is comparative. A floor
    estimator that reads too low turns the largest noise bin into a confident
    reflector, which is worse than silence because it is plausible, and the
    only way to know whether a new estimator has traded one failure for the
    other is to score both on the same frames.
    """
    rng = np.random.default_rng(31)
    bm = bin_metres_for(config.sample_rate)
    lo, hi = round(HAND_NEAR_M / bm), round(HAND_FAR_M / bm)
    by_median = 0
    by_floor = 0
    for _ in range(trials):
        profile = matched_filter(rng.normal(size=config.n_samples), reference)
        if crop:
            profile = profile[lo:hi]
        peak = float(np.max(profile))
        by_median += peak / float(np.median(profile)) >= 4.0
        by_floor += peak / noise_floor(profile) >= 4.0
    return by_median, by_floor


class TestNoiseFloorOnNoise:
    def test_a_full_noise_profile_is_reported_as_a_hand_less_often_than_before(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """The trap the first attempt at this fix fell into.

        A plain low quantile over the whole profile lands in the frame
        truncation taper, reads far too low, and takes false positives on
        noise-only frames from 27 percent to 92 percent. Blocking the estimate
        is what avoids that, and this is the measurement that says so. It also
        comes out ahead of the median it replaces, which is not the bar but is
        worth knowing.
        """
        by_median, by_floor = _noise_false_positive_rates(
            config, reference, crop=False
        )
        assert by_median == 53
        assert by_floor <= by_median
        # Pinned exactly, not bounded loosely. `noise_floor` quotes 10 percent
        # of these 200 frames in its docstring as the argument for blocking the
        # estimate, and a bound of 40 would let the real rate double while the
        # docstring kept claiming 20.
        assert by_floor == 20

    def test_a_pure_noise_crop_is_rarely_reported_as_a_hand(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """The crop must not become trigger happy in exchange for seeing hands.

        A handful of false positives survive because `min_peak_to_noise` of 4
        is a marginal bar on a 78 bin window, not because the floor is wrong;
        the median scores in the same range on the same frames.
        """
        by_median, by_floor = _noise_false_positive_rates(config, reference, crop=True)
        assert by_floor <= 5
        assert by_floor <= by_median + 5

    def test_the_floor_never_exceeds_the_profile_maximum(self) -> None:
        """A window with no quiet bins at all still gets a usable answer.

        The Rayleigh rescaling assumes the low bins are noise. Where they are
        not it overshoots, and uncapped it can put the floor above everything
        in the profile, which is not a floor. Capping keeps `peak_to_noise` at
        or above one, so a saturated window reports that nothing stands out
        rather than reporting a negative-looking ratio.
        """
        for samples in (
            np.ones(3, dtype=np.float64),
            np.array([1.0, 5.0, 1.0]),
            np.arange(10.0),
            np.full(200, 2.5),
        ):
            assert noise_floor(samples) <= float(np.max(samples)) + 1e-12

    def test_the_estimator_tracks_the_scale_of_the_noise(self) -> None:
        """Scale invariance, which is what makes a ratio threshold meaningful.

        Doubling every sample must double the floor exactly, or
        `min_peak_to_noise` would mean different things at different volumes.
        """
        rng = np.random.default_rng(5)
        samples = np.abs(rng.normal(size=400) + 1j * rng.normal(size=400))
        assert noise_floor(3.7 * samples) == pytest.approx(3.7 * noise_floor(samples))

    def test_a_silent_profile_has_a_zero_floor_and_no_peak(
        self, bin_metres: float
    ) -> None:
        zeros = np.zeros(128, dtype=np.float64)
        assert noise_floor(zeros) == 0.0
        assert strongest_peak(zeros, bin_metres) is None
        assert peak_ranges(zeros, bin_metres) == []


class TestCropOffsetReachesThePeak:
    """A range computed as index times bin_metres loses the crop's near edge.

    `EchoProfile.range_offset_m` exists to carry it, and a helper that takes
    bare `samples` plus `bin_metres` has no way to see it. These pin that the
    profile-taking entry points do.
    """

    def test_a_bare_samples_call_reads_low_by_the_near_edge(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """The trap, measured. Kept as a test so the size of it is on record."""
        profile = _hand_profile(config, reference)
        cropped = profile.crop(HAND_NEAR_M, HAND_FAR_M)

        naive = strongest_peak(cropped.samples.astype(np.float64), cropped.bin_metres)
        correct = strongest_peak_of(cropped)
        assert naive is not None
        assert correct is not None
        assert correct.range_m - naive.range_m == pytest.approx(
            cropped.range_offset_m, abs=1e-9
        )
        assert cropped.range_offset_m > 0.015

    def test_peak_ranges_of_matches_range_of_bin(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """Two independent routes from a bin to a distance must agree."""
        cropped = _hand_profile(config, reference).crop(HAND_NEAR_M, HAND_FAR_M)
        for peak in peak_ranges_of(cropped):
            assert peak.range_m == pytest.approx(
                cropped.range_of_bin(peak.bin_index), abs=cropped.bin_metres
            )

    def test_an_uncropped_profile_needs_no_offset(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        profile = _hand_profile(config, reference)
        assert profile.range_offset_m == 0.0
        bare = strongest_peak(profile.samples.astype(np.float64), profile.bin_metres)
        via_profile = strongest_peak_of(profile)
        assert bare is not None
        assert via_profile is not None
        assert bare.range_m == pytest.approx(via_profile.range_m, abs=1e-12)

    def test_a_negative_offset_is_refused(self, bin_metres: float) -> None:
        samples = np.zeros(16, dtype=np.float64)
        with pytest.raises(ValueError, match="cannot be negative"):
            peak_ranges(samples, bin_metres, range_offset_m=-0.01)
        with pytest.raises(ValueError, match="cannot be negative"):
            strongest_peak(samples, bin_metres, range_offset_m=-0.01)


class TestSingleReflectorAcrossTheCroppedWindow:
    def test_every_range_in_the_window_survives_the_crop(
        self, config: ChirpConfig, reference: NDArray[np.complex128]
    ) -> None:
        """The end-to-end claim, restated for the cropped path.

        `test_range_recovery.py` makes this claim on full profiles. Repeating
        it on the crop is not redundant: the crop is what the pipeline
        actually hands the detector, and it is the case that broke.
        """
        for truth in np.linspace(0.04, 0.26, 12):
            frame = simulate_frame(config, [Reflector(float(truth), 1.0)])
            profile = profile_from_frame(frame, reference, config.sample_rate, 0.0)
            cropped = profile.crop(HAND_NEAR_M, HAND_FAR_M)

            peak = strongest_peak_of(cropped)
            assert peak is not None, f"lost the reflector at {truth:.3f} m"
            assert peak.range_m == pytest.approx(float(truth), abs=cropped.bin_metres)
