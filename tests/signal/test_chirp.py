"""The sweep itself: frequency content, the window, and the reference.

The measurements at the bottom of this file are the evidence for the windowing
argument in `chirp.py`. They are asserted rather than merely printed so that a
future change to the window silently costing 40 percent of the range
resolution is a failing test and not a paragraph that has quietly become
false.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from wristsonar.signal import (
    analytic_reference,
    bin_metres_for,
    chirp_window,
    linear_chirp,
    matched_filter,
    sample_chirp_at,
    windowed_chirp,
)
from wristsonar.signal.chirp import DEFAULT_TUKEY_ALPHA
from wristsonar.types import SPEED_OF_SOUND, ChirpConfig


def test_chirp_length_matches_config(config: ChirpConfig) -> None:
    assert linear_chirp(config).shape == (config.n_samples,)
    assert windowed_chirp(config).shape == (config.n_samples,)
    assert analytic_reference(config).shape == (config.n_samples,)


def test_instantaneous_frequency_sweeps_the_requested_band(
    config: ChirpConfig,
) -> None:
    """The sweep must start at f_start and end at f_end, or every range is wrong.

    Measured from the analytic phase rather than from an FFT, because an FFT
    of the whole sweep says only that the band is occupied and not that it is
    traversed linearly, and it is the linearity that the matched filter
    depends on.
    """
    trim = 50
    analytic = scipy.signal.hilbert(linear_chirp(config))
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase) / (2.0 * np.pi) * config.sample_rate
    # Trim the ends, where the Hilbert transform of a finite record rings.
    freq = freq[trim:-trim]
    rate = config.bandwidth / config.duration_s
    t = (np.arange(freq.size, dtype=np.float64) + trim + 0.5) / config.sample_rate
    expected = config.f_start + rate * t
    # Compared pointwise against the closed form, so this checks the band ends
    # and the linearity in between at once. A quadratic sweep would also rise
    # from 18 to 21 kHz and would fail here.
    assert float(np.max(np.abs(freq - expected))) < 100.0
    assert expected[0] == pytest.approx(config.f_start, abs=300.0)
    assert expected[-1] == pytest.approx(config.f_end, abs=300.0)


def test_window_tapers_the_ends_and_leaves_the_middle_alone(
    config: ChirpConfig,
) -> None:
    w = chirp_window(config, DEFAULT_TUKEY_ALPHA)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert float(np.max(w)) == pytest.approx(1.0)
    middle = w[config.n_samples // 2]
    assert middle == pytest.approx(1.0)
    fraction_untouched = float(np.mean(w > 0.999))
    assert fraction_untouched == pytest.approx(1.0 - DEFAULT_TUKEY_ALPHA, abs=0.02)


def test_zero_alpha_is_no_window(config: ChirpConfig) -> None:
    np.testing.assert_allclose(chirp_window(config, 0.0), 1.0)
    np.testing.assert_allclose(windowed_chirp(config, 0.0), linear_chirp(config))


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_invalid_alpha_rejected(config: ChirpConfig, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        chirp_window(config, alpha)


def test_continuous_sampler_agrees_with_the_sampled_chirp(
    config: ChirpConfig,
) -> None:
    """The simulator and the analysis must be evaluating the same waveform.

    If these two ever drift apart, every range in the test suite shifts by a
    constant and every test still passes, which is precisely the failure this
    layer is built to prevent.
    """
    t = np.arange(config.n_samples, dtype=np.float64) / config.sample_rate
    np.testing.assert_allclose(
        sample_chirp_at(config, t), windowed_chirp(config), atol=1e-12
    )


def test_sampler_is_zero_outside_the_sweep(config: ChirpConfig) -> None:
    t = np.array([-1e-3, -1e-9, config.duration_s, config.duration_s + 1e-3])
    np.testing.assert_allclose(sample_chirp_at(config, t), 0.0)


def test_analytic_reference_real_part_is_the_transmitted_signal(
    config: ChirpConfig,
) -> None:
    ref = analytic_reference(config)
    np.testing.assert_allclose(np.real(ref), windowed_chirp(config), atol=1e-12)


def _mainlobe_width_bins(profile: NDArray[np.float64]) -> int:
    """Width of the correlation peak at half power, in bins."""
    k = int(np.argmax(profile))
    half = float(profile[k]) / np.sqrt(2.0)
    lo = k
    while lo > 0 and profile[lo] > half:
        lo -= 1
    hi = k
    while hi < profile.size - 1 and profile[hi] > half:
        hi += 1
    return hi - lo


def _sidelobe_db(profile: NDArray[np.float64], skip_widths: int) -> float:
    k = int(np.argmax(profile))
    w = _mainlobe_width_bins(profile) * skip_widths
    mask = np.ones(profile.size, dtype=bool)
    mask[max(0, k - w) : k + w] = False
    return float(20.0 * np.log10(float(np.max(profile[mask])) / float(profile[k])))


def _autocorrelation(config: ChirpConfig, alpha: float) -> NDArray[np.float64]:
    from wristsonar.signal import Reflector, simulate_frame

    frame = simulate_frame(config, [Reflector(0.15, 1.0)], alpha=alpha)
    return matched_filter(frame, analytic_reference(config, alpha))


def test_mainlobe_width_is_the_theoretical_resolution_within_a_small_margin(
    config: ChirpConfig,
) -> None:
    """The half-power width of the compressed pulse should be about c/2B.

    This is the measurement that turns `ChirpConfig.range_resolution` from a
    docstring into a claim about the code.
    """
    bm = bin_metres_for(config.sample_rate)
    width_m = _mainlobe_width_bins(_autocorrelation(config, 0.0)) * bm
    assert width_m == pytest.approx(config.range_resolution, rel=0.05)


def test_tukey_costs_far_less_resolution_than_hann(config: ChirpConfig) -> None:
    """The reason the window is Tukey at 0.25 and not a full Hann.

    Hann broadens the compressed pulse to nearly twice the c/2B limit. On a
    system whose entire difficulty is that 5.7 cm is already wider than a
    hand, that is not an affordable price for sidelobes that are not the
    binding problem (see the next test).
    """
    bm = bin_metres_for(config.sample_rate)
    rect = _mainlobe_width_bins(_autocorrelation(config, 0.0)) * bm
    tukey = _mainlobe_width_bins(_autocorrelation(config, DEFAULT_TUKEY_ALPHA)) * bm
    hann = _mainlobe_width_bins(_autocorrelation(config, 1.0)) * bm
    assert tukey / rect < 1.20
    assert hann / rect > 1.70


def test_first_sidelobe_is_about_minus_13_db_whatever_the_window(
    config: ChirpConfig,
) -> None:
    """The measurement behind the peak-detection prominence threshold.

    A time-domain taper on an LFM sweep only becomes a spectral taper when the
    time-bandwidth product is large. Here BT is 37.5, the sweep spectrum is
    Fresnel-rippled rather than rectangular, and the first sidelobe stays near
    the classic -13.3 dB no matter how the ends are tapered. Peak detection
    therefore has to tolerate it, which is why the default prominence ratio
    downstream is 0.3 and not something smaller.
    """
    assert config.bandwidth * config.duration_s == pytest.approx(37.5)
    for alpha in (0.0, DEFAULT_TUKEY_ALPHA, 0.5, 1.0):
        first = _sidelobe_db(_autocorrelation(config, alpha), skip_widths=1)
        assert -16.0 < first < -11.0


def test_tapering_does_suppress_the_far_sidelobes(config: ChirpConfig) -> None:
    """What the window actually buys, as opposed to what it is assumed to buy."""
    rect = _sidelobe_db(_autocorrelation(config, 0.0), skip_widths=4)
    tukey = _sidelobe_db(_autocorrelation(config, DEFAULT_TUKEY_ALPHA), skip_widths=4)
    strong = _sidelobe_db(_autocorrelation(config, 0.5), skip_widths=4)
    assert tukey < rect
    assert strong < tukey - 10.0


@given(
    f_start=st.floats(15_000.0, 19_000.0),
    band=st.floats(500.0, 4_000.0),
    duration_ms=st.floats(5.0, 25.0),
)
def test_chirp_is_bounded_and_finite_for_any_plausible_config(
    f_start: float, band: float, duration_ms: float
) -> None:
    """Property: no configuration in the plausible envelope produces a NaN.

    A NaN here would propagate silently into a training set, so the range of
    configurations is swept rather than spot-checked.
    """
    cfg = ChirpConfig(
        f_start=f_start, f_end=f_start + band, duration_s=duration_ms / 1000.0
    )
    for signal in (linear_chirp(cfg), windowed_chirp(cfg)):
        assert np.all(np.isfinite(signal))
        assert float(np.max(np.abs(signal))) <= 1.0 + 1e-9
    assert np.all(np.isfinite(analytic_reference(cfg)))


def test_speed_of_sound_is_not_silently_redefined() -> None:
    """Guards the constant every range in this layer is scaled by."""
    assert pytest.approx(343.0) == SPEED_OF_SOUND
