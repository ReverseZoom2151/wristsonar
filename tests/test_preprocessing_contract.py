"""The cross-path test: one PCM stream, two window builders, one answer.

This is the test the rest of the preprocessing work exists for. Everything
else in this area is a mechanism; this is the measurement. Identical synthetic
PCM goes into the training corpus builder and into the live capture assembler,
and the windows they produce must be the same tensor to float32 tolerance.

Why one test rather than three. The three defects it replaces were not three
bugs, they were one absence showing up three times: the preprocessing contract
lived in `SessionData.windows` and in `EchoWindowAssembler` and was written
down in neither the checkpoint nor anywhere shared. Any drift between two
copies of a contract is invisible downstream, because a wrongly scaled,
wrongly cropped or wrongly aligned window still has the right shape and still
produces well formed pose JSON. So the assertion has to be equality of the
tensors themselves:

  * scale, which is what a live window normalised nowhere and a training
    window normalised to peak 1.0 disagreed about by three orders of magnitude;
  * origin, which is what the bin zero offset decides, and which a model reads
    as hand size rather than as an offset;
  * alignment, which is what pairing a shipped differential with an original
    by raw index got wrong, in the direction that lets a frame recorded after
    the predicted pose into a causal window.

WatchHand ships no PCM, so the training half here reconstructs the arrays the
release would have shipped for this recording, using the documented
preprocessing and nothing else. `_shipped_arrays` is the only place in the
suite that plays the part of the dataset's producer, and it is deliberately
literal about it.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np
import pytest
from numpy.typing import NDArray

from wristsonar.capture.processor import EchoWindowAssembler
from wristsonar.data.watchhand import (
    SessionData,
    SessionRef,
    Study,
    estimate_bin_zero_offset,
    estimate_differential_lag,
)
from wristsonar.preprocess import (
    WATCHHAND_CHIRP,
    WATCHHAND_PREPROCESSING,
    PreprocessingDescriptor,
    PreprocessingMismatchError,
)
from wristsonar.runtime.frames import CaptureFrame
from wristsonar.signal.chirp import analytic_reference
from wristsonar.signal.echo_profile import profile_from_frame
from wristsonar.signal.simulate import Reflector, simulate_recording

N_FRAMES = 14
FRAME_PERIOD_S = 1.0 / WATCHHAND_CHIRP.frame_rate


def _descriptor(*, lag: int, offset: int = 0) -> PreprocessingDescriptor:
    """A small window, so a fourteen frame recording produces several."""
    return (
        WATCHHAND_PREPROCESSING.with_window_frames(4)
        .with_differential_lag(lag)
        .with_bin_zero_offset(offset)
    )


def _pcm() -> NDArray[np.float64]:
    """One recording of back-to-back chirps off a hand that moves.

    Static reflectors stand for the watch body and the wrist, and the moving
    one for a fingertip. The scene has to move: a differential of a frozen
    scene is zero, and zero is equal to zero under every alignment, so a still
    hand would let the alignment defect pass this test unnoticed.
    """
    moving = [
        [Reflector(range_m=0.06 + 0.004 * index, amplitude=0.4)]
        for index in range(N_FRAMES)
    ]
    return simulate_recording(
        WATCHHAND_CHIRP,
        moving,
        # Range zero is the speaker to microphone direct path, the large
        # static reflector `estimate_bin_zero_offset` looks for.
        static=(
            Reflector(range_m=0.0, amplitude=1.0),
            Reflector(range_m=0.02, amplitude=0.3),
        ),
        snr_db=25.0,
        rng=np.random.default_rng(3),
    )


def _frames(pcm: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    n = WATCHHAND_CHIRP.n_samples
    return [np.asarray(pcm[i * n : (i + 1) * n]) for i in range(N_FRAMES)]


def _shipped_arrays(
    frames: list[NDArray[np.float64]], descriptor: PreprocessingDescriptor
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Play the part of WatchHand's producer for one recording.

    Matched filter every frame into a 600 bin profile, then difference
    consecutive profiles, then trim the leading columns that the declared lag
    implies. The trim is the only part that is a choice rather than a
    transcription, and it is exactly the choice the release does not document
    and `estimate_differential_lag` measures.
    """
    reference = analytic_reference(WATCHHAND_CHIRP)
    profiles = np.stack(
        [
            profile_from_frame(
                frame,
                reference,
                WATCHHAND_CHIRP.sample_rate,
                index * FRAME_PERIOD_S,
                n_bins=WATCHHAND_CHIRP.n_samples,
            ).samples
            for index, frame in enumerate(frames)
        ],
        axis=1,
    ).astype(np.float32)
    diff = np.diff(profiles, axis=1)[:, descriptor.differential_lag - 1 :]
    return profiles, np.ascontiguousarray(diff, dtype=np.float32)


def _session(
    profiles: NDArray[np.float32], diff: NDArray[np.float32]
) -> SessionData:
    return SessionData(
        ref=SessionRef(Study.MAIN, 1, 0, PurePosixPath("synthetic"), "audio001"),
        profiles=profiles,
        diff_profiles=diff,
        gestures=(),
        frame_timestamps=np.zeros(0, dtype=np.float64),
        audio_sync_index=0,
        gt_sync_timestamp=0.0,
        sample_rate=WATCHHAND_CHIRP.sample_rate,
        frame_length=WATCHHAND_CHIRP.n_samples,
    )


def _live_windows(
    frames: list[NDArray[np.float64]], descriptor: PreprocessingDescriptor
) -> list[CaptureFrame]:
    assembler = EchoWindowAssembler(descriptor=descriptor)
    produced: list[CaptureFrame] = []
    for index, frame in enumerate(frames):
        window = assembler.push(
            frame,
            timestamp_s=index * FRAME_PERIOD_S,
            sample_rate=WATCHHAND_CHIRP.sample_rate,
            continuous=True,
        )
        if window is not None:
            produced.append(window)
    return produced


@pytest.mark.parametrize(
    ("lag", "offset"),
    [(1, 0), (2, 0), (2, 4)],
)
def test_training_and_live_paths_build_the_same_window(lag: int, offset: int) -> None:
    """One PCM stream, two builders, identical tensors.

    Parametrised over both undocumented parameters, because a test pinned to
    one value of each would pass for a pipeline that ignores them and hard
    codes the same defaults twice, which is the situation this replaces.
    """
    descriptor = _descriptor(lag=lag, offset=offset)
    frames = _frames(_pcm())
    profiles, diff = _shipped_arrays(frames, descriptor)
    session = _session(profiles, diff)

    trained = [window for _, window in session.windows(descriptor=descriptor)]
    live = _live_windows(frames, descriptor)

    assert trained, "the synthetic recording must be long enough to window"
    # The live path is ready `lag - 1` windows sooner, because the corpus
    # builder skips the leading originals that the shipped array has no column
    # for while the live path has no shipped array to be short of. Compare
    # from the first moment both paths describe.
    assert len(live) == len(trained) + descriptor.differential_lag - 1
    aligned = live[descriptor.differential_lag - 1 :]
    for index, (expected, actual) in enumerate(zip(trained, aligned, strict=True)):
        assert actual.samples.shape == descriptor.window_shape
        np.testing.assert_allclose(
            actual.samples,
            expected,
            rtol=0.0,
            atol=1e-6,
            err_msg=f"window {index} differs between the training and live paths",
        )


def test_the_two_paths_agree_on_when_a_window_is_ready() -> None:
    """Alignment includes the clock, not only the tensor.

    The k-th window of each path has to describe the k-th moment. If one path
    warmed up faster than the other the tensors could still match pairwise
    while every pose came out shifted in time.
    """
    descriptor = _descriptor(lag=2)
    frames = _frames(_pcm())
    profiles, diff = _shipped_arrays(frames, descriptor)

    session = _session(profiles, diff)
    starts = [start for start, _ in session.windows(descriptor=descriptor)]
    live = _live_windows(frames, descriptor)
    aligned = live[descriptor.differential_lag - 1 :]

    assert starts[0] == descriptor.differential_lag
    last_frames = [start + descriptor.window_frames - 1 for start in starts]
    assert [window.timestamp_s for window in aligned] == pytest.approx(
        [frame * FRAME_PERIOD_S for frame in last_frames]
    )


def test_the_cross_path_test_has_teeth() -> None:
    """A wrongly declared alignment must break the equality, and be caught.

    Without this, an equality test between two paths that both do nothing
    interesting would pass forever. Here the session ships a differential
    aligned at lag 1 while the descriptor declares 2, so the training path
    reads a difference one frame stale and the live path, which builds the
    immediate difference, cannot reproduce it. The corpus builder is supposed
    to refuse before that ever reaches a model, and does.
    """
    frames = _frames(_pcm())
    shipped_at_one = _descriptor(lag=1)
    declared_two = _descriptor(lag=2)
    profiles, diff = _shipped_arrays(frames, shipped_at_one)
    session = _session(profiles, diff)

    trained = [window for _, window in session.windows(descriptor=declared_two)]
    live = _live_windows(frames, declared_two)
    aligned = live[declared_two.differential_lag - 1 :]

    assert not np.allclose(aligned[0].samples, trained[0], rtol=0.0, atol=1e-6)
    honest_origin = declared_two.with_bin_zero_offset(
        estimate_bin_zero_offset(profiles)
    )
    with pytest.raises(PreprocessingMismatchError, match=r"with_differential_lag\(1\)"):
        session.verify_preprocessing(honest_origin)


def test_the_synthetic_recording_measures_the_parameters_it_was_built_with() -> None:
    """The two estimators, run against a recording of known construction.

    The lag is recoverable exactly, because these arrays really are a
    difference of these originals, which is the case the released arrays may
    not be.

    The origin is recoverable only to within a bin or two, and that is a
    property of the physics rather than of the estimator. A matched filter
    puts range zero at lag zero, but the mainlobe of a 3 kHz sweep is tens of
    bins wide, so the argmax of a near-field peak sits wherever the skirts of
    the neighbouring reflectors leave it. Asserting a bound rather than an
    exact bin is the honest version, and the bound is what matters: two bins
    is 7 mm, against the 21.4 cm the crop spans.
    """
    frames = _frames(_pcm())
    for lag in (1, 2, 3):
        profiles, diff = _shipped_arrays(frames, _descriptor(lag=lag))
        assert estimate_differential_lag(profiles, diff) == lag
        assert estimate_bin_zero_offset(profiles) <= 2
