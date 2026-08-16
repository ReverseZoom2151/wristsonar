"""The end-to-end claim: a reflector put at r comes back at r.

Everything else in this layer is a means to this. The test is deliberately
run across many ranges rather than at one convenient distance, because the
failure modes that matter here are proportional to range or to the fractional
part of a delay, and both hide perfectly at a single spot check.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from wristsonar.signal import (
    Reflector,
    analytic_reference,
    bin_metres_for,
    matched_filter,
    peak_ranges,
    simulate_frame,
    strongest_peak,
)
from wristsonar.types import ChirpConfig

HAND_NEAR_M = 0.02
"""Closer than this and a reflector is inside the watch case, not on a hand."""

HAND_FAR_M = 0.30
"""Roughly the reach of a hand from a wrist-worn transducer, with margin."""


def _recover(
    config: ChirpConfig,
    reference: NDArray[np.complex128],
    range_m: float,
    amplitude: float = 1.0,
) -> float:
    profile = matched_filter(
        simulate_frame(config, [Reflector(range_m, amplitude)]), reference
    )
    peak = strongest_peak(profile, bin_metres_for(config.sample_rate))
    assert peak is not None
    return peak.range_m


@given(range_m=st.floats(HAND_NEAR_M, HAND_FAR_M))
def test_single_reflector_is_recovered_within_one_bin(range_m: float) -> None:
    """Property, over the whole plausible hand envelope, at continuous ranges.

    Hypothesis is generating genuinely continuous ranges, so most examples put
    the true delay between two samples. That is the case a bin-quantised
    estimator would fail and a correct one would not, which is why the
    strategy is floats and not a grid.
    """
    config = ChirpConfig()
    reference = analytic_reference(config)
    recovered = _recover(config, reference, range_m)
    assert abs(recovered - range_m) < bin_metres_for(config.sample_rate)


def test_range_error_across_the_hand_envelope_is_sub_millimetre(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """The headline accuracy figure for this layer, stated as a bound.

    Noiseless and against point reflectors, so it is a statement about the
    arithmetic and not about a hand. What it rules out is a gross systematic
    bias: a delay convention off by half a sample would show up here as a
    0.9 mm offset while every within-one-bin test still passed.
    """
    bm = bin_metres_for(config.sample_rate)
    truths = np.linspace(HAND_NEAR_M, HAND_FAR_M, 97)
    errors = np.array([_recover(config, reference, float(r)) - r for r in truths])
    assert float(np.max(np.abs(errors))) < 0.7e-3
    assert float(np.max(np.abs(errors))) < 0.2 * bm
    near = errors[truths < 0.10]
    assert float(np.max(np.abs(near))) < 0.1e-3


def test_the_residual_bias_is_the_frame_truncation_and_it_pulls_inward(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """The error left over is not random, and it is worth naming rather than
    absorbing into a tolerance.

    An echo at delay d overlaps its own frame for only the remaining N - d
    samples, so the matched filter sees a progressively truncated copy of the
    sweep. Truncation removes the late, high-frequency end of the echo, which
    weights the correlation towards earlier lags, so every estimate is biased
    slightly near and the bias grows with range: about 0.05 mm at 10 cm,
    0.5 mm at 30 cm, 2.4 mm at 1 m and 16 mm at 1.5 m. The sign never flips.

    A real device fixes this with a guard interval between chirps. This layer
    cannot, since it is handed whatever the device recorded, so it measures
    the effect instead.
    """
    ranges = [0.10, 0.20, 0.30, 0.50, 1.00, 1.50]
    errors = [_recover(config, reference, r) - r for r in ranges]
    assert all(e < 0.0 for e in errors)
    magnitudes = [abs(e) for e in errors]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[0] < 0.1e-3
    assert magnitudes[-1] < 20e-3


def test_recovery_holds_out_to_the_unambiguous_range(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """The reflector is still found beyond the hand, just less precisely.

    Tolerances widen with range because the truncation above is real. Stating
    them as a table rather than as one generous bound keeps the degradation
    visible.
    """
    bm = bin_metres_for(config.sample_rate)
    for range_m, tolerance_bins in ((0.5, 1.0), (1.0, 1.0), (1.5, 6.0)):
        error = abs(_recover(config, reference, range_m) - range_m)
        assert error < tolerance_bins * bm


@given(
    range_m=st.floats(HAND_NEAR_M, HAND_FAR_M),
    amplitude=st.floats(0.05, 5.0),
)
def test_recovered_range_does_not_depend_on_amplitude(
    range_m: float, amplitude: float
) -> None:
    """Property: range and amplitude are independent outputs.

    They come from the same peak, so a bug that lets the interpolation pick up
    a scale factor would couple them, and a system that reports a nearer hand
    when the volume goes up is worse than useless.
    """
    config = ChirpConfig()
    reference = analytic_reference(config)
    loud = _recover(config, reference, range_m, amplitude)
    quiet = _recover(config, reference, range_m, 1.0)
    assert loud == pytest.approx(quiet, abs=1e-9)


def test_a_lone_reflector_produces_exactly_one_reported_peak(
    config: ChirpConfig, reference: NDArray[np.complex128]
) -> None:
    """One reflector must not become three via its own sidelobes.

    With the first sidelobe stuck near -13 dB (see test_chirp.py) this is a
    real risk, and it is what the prominence threshold in peak_ranges is set
    to prevent.
    """
    bm = bin_metres_for(config.sample_rate)
    for range_m in np.linspace(0.03, 0.25, 23):
        profile = matched_filter(
            simulate_frame(config, [Reflector(float(range_m), 1.0)]), reference
        )
        peaks = peak_ranges(profile, bm)
        assert len(peaks) == 1
        assert peaks[0].range_m == pytest.approx(float(range_m), abs=bm)
