"""The matched filter, tested at the level of its four failure modes.

Circular wraparound, the lag-to-range mapping, the envelope, and linearity.
Each of these fails silently rather than loudly, so each gets an explicit test
rather than being covered incidentally by an end-to-end range check.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from wristsonar.signal import (
    Reflector,
    analytic_reference,
    bin_metres_for,
    correlate_fft,
    envelope,
    matched_filter,
    parabolic_offset,
    range_axis,
    simulate_frame,
    strongest_peak,
)
from wristsonar.types import SPEED_OF_SOUND, ChirpConfig


def test_bin_metres_is_c_over_twice_the_sample_rate(config: ChirpConfig) -> None:
    """The factor of two is the round trip, and it belongs here exactly once."""
    bm = bin_metres_for(config.sample_rate)
    assert bm == pytest.approx(SPEED_OF_SOUND / (2.0 * config.sample_rate))
    assert bm == pytest.approx(0.003573, abs=1e-6)


def test_bin_spacing_and_range_resolution_are_different_quantities(
    config: ChirpConfig,
) -> None:
    """The distinction this whole project turns on.

    bin_metres is how finely the answer is written down: c/2fs, 3.6 mm.
    range_resolution is how finely two answers can be told apart: c/2B, 57 mm.
    Their ratio is fs/B, here 16. Any claim of millimetre sensing that rests
    on the first number while ignoring the second is the error the protocol
    module exists to make hard.
    """
    bm = bin_metres_for(config.sample_rate)
    assert config.range_resolution / bm == pytest.approx(
        config.sample_rate / config.bandwidth
    )
    assert config.range_resolution / bm == pytest.approx(16.0)


def test_a_full_frame_of_bins_spans_the_unambiguous_range(
    config: ChirpConfig,
) -> None:
    """n_bins * bin_metres must equal ChirpConfig.max_unambiguous_range.

    These are two independent derivations of the same quantity, one from the
    sample rate and one from the frame duration. Pinning their agreement is
    what stops a factor of two migrating between this layer and the types.
    """
    bm = bin_metres_for(config.sample_rate)
    assert config.n_samples * bm == pytest.approx(config.max_unambiguous_range)


def test_range_axis_matches_bin_metres(config: ChirpConfig) -> None:
    axis = range_axis(10, config.sample_rate)
    np.testing.assert_allclose(axis, np.arange(10) * bin_metres_for(config.sample_rate))


def test_lag_of_the_peak_equals_the_round_trip_in_samples(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """The single most important line in the layer, checked in raw samples.

    A reflector at r has an acoustic path of 2r, a flight time of 2r/c and a
    lag of round(2r*fs/c) samples. Checked in samples rather than in metres so
    that a failure points at the lag mapping and not at some downstream
    conversion.
    """
    for range_m in (0.02, 0.05, 0.083, 0.2, 0.5, 1.0):
        expected_lag = 2.0 * range_m * config.sample_rate / SPEED_OF_SOUND
        profile = matched_filter(
            simulate_frame(config, [Reflector(range_m, 1.0)]), reference
        )
        assert int(np.argmax(profile)) == pytest.approx(expected_lag, abs=1.0)


def test_no_circular_wraparound(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """Without zero padding, a far reflector appears on the skin.

    This is the bug that would be hardest to notice in a trained model: the
    ghost is stable, plausible and at short range, so it looks like the hand.
    Constructed here by hand rather than through the simulator so the test
    does not depend on the simulator being right either.
    """
    n = config.n_samples
    tx = np.real(reference)
    # An echo whose energy sits entirely in the last tenth of the frame. Under
    # circular correlation its correlation tail folds back onto lag 0.
    frame = np.zeros(n, dtype=np.float64)
    late = int(0.9 * n)
    frame[late:] = tx[: n - late]
    profile = matched_filter(frame, reference)
    near = profile[: n // 8]
    far = profile[n // 2 :]
    assert float(np.max(near)) < 0.2 * float(np.max(far))


def test_analytic_reference_path_equals_hilbert_of_the_real_correlation(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """Pins the identity that lets the envelope come out of one FFT.

    corr(x, r + iH(r)) = C + iH(C). If a library update changes a Hilbert sign
    convention, the two paths diverge here rather than producing a subtly
    biased range estimate everywhere.

    Compared on the full correlation rather than on the truncated pipeline
    output, because the Hilbert transform is not local: truncating first and
    transforming second is a different operation from transforming first and
    truncating second. The pipeline does the latter, which is the correct
    order, and this test compares like with like.

    The agreement is to about 1e-5 relative rather than to machine precision.
    That residual is not the correlation, it is the reference: an analytic
    signal built by Hilbert transforming a finite record is only approximately
    one-sided near its ends.
    """
    frame = simulate_frame(config, [Reflector(0.11, 0.8), Reflector(0.3, 0.4)])
    real_ref = np.real(reference)
    via_analytic = np.abs(scipy.signal.correlate(frame, reference, mode="full"))
    via_hilbert = envelope(scipy.signal.correlate(frame, real_ref, mode="full"))
    scale = float(np.max(via_hilbert))
    assert float(np.max(np.abs(via_analytic - via_hilbert))) < 1e-4 * scale
    # And the pipeline output is a slice of the analytic route, not of the other.
    ours = matched_filter(frame, reference)
    lag0 = real_ref.size - 1
    np.testing.assert_allclose(
        ours,
        via_analytic[lag0 : lag0 + ours.size] / float(np.sum(real_ref**2)),
        atol=1e-9,
    )


def test_envelope_does_not_oscillate_at_the_carrier(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """Why the envelope rather than the raw correlation.

    The raw correlation crosses zero every half carrier period, so its samples
    near the peak encode phase, not range. The envelope near the peak is
    smooth and single-signed, which is what makes a three-point interpolation
    of it meaningful.
    """
    frame = simulate_frame(config, [Reflector(0.12, 1.0)])
    raw = np.real(correlate_fft(frame, np.real(reference).astype(np.complex128)))
    env = matched_filter(frame, reference)
    k = int(np.argmax(env))
    window = slice(k - 6, k + 7)
    assert np.count_nonzero(np.diff(np.sign(raw[window]))) >= 3
    assert np.count_nonzero(np.diff(np.sign(env[window]))) == 0


@given(amplitude=st.floats(0.01, 10.0), extra=st.floats(0.01, 10.0))
def test_correlation_is_linear_in_reflector_amplitude(
    amplitude: float, extra: float
) -> None:
    """Superposition, checked as an absolute number and not as a ratio.

    The correlation is normalised by reference energy, so a reflector of
    amplitude a produces a peak of exactly a. That makes this test able to
    catch a shared scale error, which a ratio-based linearity test cannot.
    """
    config = ChirpConfig()
    reference = analytic_reference(config)
    one = matched_filter(
        simulate_frame(config, [Reflector(0.09, amplitude)]), reference
    )
    two = matched_filter(
        simulate_frame(config, [Reflector(0.09, amplitude + extra)]), reference
    )
    assert float(np.max(one)) == pytest.approx(amplitude, rel=0.02)
    assert float(np.max(two)) == pytest.approx(amplitude + extra, rel=0.02)


def test_superposition_of_two_separated_reflectors(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """The channel is linear, so two frames added is one frame with two targets.

    Held apart far enough that the two mainlobes do not interfere, since
    coherent interference is real physics and would make an exact-equality
    check here fail for the right reasons rather than the wrong ones.
    """
    a = simulate_frame(config, [Reflector(0.08, 1.0)])
    b = simulate_frame(config, [Reflector(0.60, 0.6)])
    both = simulate_frame(config, [Reflector(0.08, 1.0), Reflector(0.60, 0.6)])
    np.testing.assert_allclose(both, a + b, atol=1e-12)


@given(
    values=st.lists(st.floats(0.0, 1e3, allow_nan=False), min_size=3, max_size=64),
    index=st.integers(0, 63),
)
def test_parabolic_offset_never_leaves_the_bin(values: list[float], index: int) -> None:
    """Property: interpolation refines a bin, it never relocates the peak.

    An unclamped log-parabolic fit on a near-flat or noisy triple can produce
    an arbitrarily large offset. Allowing that would let one noisy sample move
    a reported reflector by centimetres, so the estimator is clamped and the
    clamp is a property rather than a comment.
    """
    samples = np.asarray(values, dtype=np.float64)
    offset = parabolic_offset(samples, min(index, samples.size - 1))
    assert -0.5 <= offset <= 0.5


def test_parabolic_offset_is_zero_at_the_array_edges() -> None:
    samples = np.array([5.0, 1.0, 0.5, 0.2])
    assert parabolic_offset(samples, 0) == 0.0
    assert parabolic_offset(samples, samples.size - 1) == 0.0


def test_parabolic_offset_recovers_a_known_gaussian_apex() -> None:
    """A Gaussian is exactly a parabola in the log domain, which is the model."""
    for true_offset in (-0.4, -0.1, 0.0, 0.25, 0.45):
        k = np.arange(21, dtype=np.float64)
        samples = np.exp(-(((k - (10.0 + true_offset)) / 3.0) ** 2))
        assert parabolic_offset(samples, 10) == pytest.approx(true_offset, abs=1e-9)


def test_silent_profile_reports_nothing(config: ChirpConfig) -> None:
    """A matched filter always has a global maximum. Silence must not be a hand."""
    bm = bin_metres_for(config.sample_rate)
    assert strongest_peak(np.zeros(64), bm) is None


def test_rejects_malformed_input(reference: NDArray[np.complex128]) -> None:
    with pytest.raises(ValueError, match="empty"):
        correlate_fft(np.zeros(0), reference)
    with pytest.raises(ValueError, match="one dimensional"):
        correlate_fft(np.zeros((4, 4)), reference)
    with pytest.raises(ValueError, match="sample rate"):
        bin_metres_for(0)


def test_scipy_correlate_agrees_with_the_fft_route(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """An independent implementation of the same convention, at small scale.

    Cheap insurance against the padding, slicing and conjugation in
    correlate_fft agreeing with themselves but not with correlation.
    """
    rng = np.random.default_rng(7)
    frame = rng.normal(size=200)
    ref = reference[:64]
    direct = scipy.signal.correlate(frame, ref, mode="full")
    # scipy's full output places lag 0 at index len(ref) - 1.
    direct = direct[ref.size - 1 :][: frame.size]
    ours = correlate_fft(frame, ref) * float(np.sum(np.real(ref) ** 2))
    np.testing.assert_allclose(ours, direct, atol=1e-9)
