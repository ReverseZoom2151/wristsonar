"""The single preprocessing contract training and live capture must share.

This module exists because the contract used to exist twice: once inside
`SessionData.windows`, once inside `EchoWindowAssembler`, and written down in
neither the checkpoint nor anywhere both halves could read. Two copies of one
convention drift, and every way they drift is silent. A live window whose peak
is 6553 where training taught the model to expect roughly 1.0 still has the
right shape, the right dtype and produces well formed pose JSON. It is simply
wrong, and nothing in the pipeline can tell.

So the contract is one frozen object, and it travels: into the checkpoint, so
a weight file records the preprocessing it was trained under, and into both
window builders, so neither can quietly invent its own defaults.

Placement. This sits above `wristsonar.types` and `wristsonar.signal` and below
`wristsonar.data` and `wristsonar.capture`, which is what lets both of those
import it without importing each other.

Two of the fields describe things nobody documented and that therefore have to
be measured rather than assumed. `bin_zero_offset` is the array index that
represents range zero, and `differential_lag` is how a shipped differential
array lines up with the original array beside it. Both have measuring
functions in `wristsonar.data.watchhand`, and the corpus builder runs them and
refuses to build a corpus whose measurement contradicts the descriptor a
checkpoint will carry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from wristsonar.signal.echo_profile import Normalisation
from wristsonar.types import ChirpConfig

__all__ = [
    "PREPROCESSING_SCHEMA",
    "WATCHHAND_CHIRP",
    "WATCHHAND_PREPROCESSING",
    "DifferentialPhase",
    "Normalisation",
    "PreprocessingDescriptor",
    "PreprocessingMismatchError",
]

PREPROCESSING_SCHEMA: Final = "wristsonar.preprocessing/1"


class PreprocessingMismatchError(ValueError):
    """Two halves of the pipeline disagree about how a window is built.

    Its own class because this is never a corrupt-file problem and never a
    user input problem. It means a model is about to be fed something other
    than what it was trained on, which is the one failure this package cannot
    detect downstream.
    """


class DifferentialPhase(StrEnum):
    """Which frame of a differenced pair the difference is attributed to.

    A frame-to-frame difference is built from two originals, at indices i and
    i+1. Attributing it to i+1 is causal: a window ending at frame i+1 then
    contains only data recorded at or before i+1. Attributing it to i is not,
    because the window would carry evidence from i+1, recorded after the pose
    it is supposed to predict, and the resulting offline accuracy is
    unreachable online.

    Both names exist so a checkpoint can state which convention produced it,
    including one this release refuses to execute. `PreprocessingDescriptor`
    rejects `EARLIER` outright, so a bundle claiming it fails at load rather
    than being run and quietly believed.
    """

    LATER = "later"
    EARLIER = "earlier"


WATCHHAND_CHIRP: Final = ChirpConfig(
    f_start=18_000.0,
    f_end=21_000.0,
    duration_s=600 / 48_000,
    sample_rate=48_000,
)
"""The sweep the watches actually transmitted.

Read off the transmit waveform filename in every config.json,
`fmcw18000_b3000_l600_s48k.wav`: start 18000 Hz, bandwidth 3000 Hz, length
600 samples, 48 kHz. A live capture app that emits anything else is not
producing this dataset's input, however similar the numbers look.
"""


@dataclass(frozen=True, slots=True)
class PreprocessingDescriptor:
    """Everything a training window and a live window must agree about.

    Frozen and comparable, so that "do these two pipelines agree" is one
    equality test rather than a checklist that a future field silently falls
    off the end of.
    """

    chirp: ChirpConfig
    """Sweep and sample rate. A different chirp is a different matched filter,
    a different bin size and therefore a different range axis."""

    crop_bins: int
    """Bins kept from `bin_zero_offset` onwards. Everything past this is
    room, and a model handed the room learns the room."""

    window_frames: int
    """Frames per model input."""

    bin_zero_offset: int = 0
    """Array index that represents range zero.

    Zero is the matched filter's own origin, which is what the live path
    produces by construction: lag zero is range zero. WatchHand does not
    document the origin of its shipped arrays, so cropping both paths from
    index zero is an assumption until it is measured.
    `wristsonar.data.watchhand.estimate_bin_zero_offset` measures it from the
    static direct path, and `SessionData.verify_preprocessing` refuses a
    corpus build whose measurement disagrees with this field, naming the
    measured value. A constant offset between the two origins is not a subtle
    error: the model reads it as a differently sized hand.
    """

    differential_lag: int = 2
    """Original column that differential column zero of a shipped array holds.

    This is an index offset into somebody else's array, not a delay in the
    signal. Under the later-frame convention the differential belonging with
    original i is always the difference ending at i; the lag says only where
    that quantity sits in the array WatchHand shipped, so the training path
    reads column i - lag and the first `lag` originals have no differential
    and are skipped. The live path builds the difference itself and therefore
    has no array to index and no lag to apply, which is exactly why the two
    paths can be identical.

    Two is the conservative choice rather than a measured one, and the reason
    is in `estimate_differential_lag`: the shipped differential is two frames
    shorter than the original (27,198 against 27,196 for one documented
    session), so a plain frame-to-frame difference has additionally been
    trimmed by one column at an end nobody recorded. Trimmed at the front the
    lag is 2, trimmed at the back it is 1.

    The asymmetry is what decides the default. Declaring 1 when the truth is 2
    reads column i - 1, which holds the difference ending at i + 1, and puts a
    frame recorded after the predicted pose inside a causal window. Declaring
    2 when the truth is 1 reads a difference ending at i - 1: one frame stale,
    causal, and costing a little signal. Neither is free, and staleness is the
    one that cannot inflate a reported number. `estimate_differential_lag`
    measures the real value wherever the shipped differential is reproducible
    from the shipped original, and the corpus builder refuses to proceed when
    the measurement contradicts this field, so the fallback applies only where
    nothing measurable exists.
    """

    differential_phase: DifferentialPhase = DifferentialPhase.LATER
    """See `DifferentialPhase`. Recorded so a checkpoint states it."""

    window_normalisation: Normalisation = Normalisation.PEAK
    """Applied once to the assembled (2, bins, frames) window, never per frame.

    Peak, because that is what the training corpus was built with and what the
    feature normalizer was fitted on. Per-frame normalisation is deliberately
    absent from this contract and fixed at none on both paths: normalising
    before differencing rescales each frame independently and leaves a residue
    of the static background proportional to how much the scale moved, so the
    only correct place for it is after the two channels are stacked.
    """

    schema: str = PREPROCESSING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREPROCESSING_SCHEMA:
            raise PreprocessingMismatchError(
                f"unsupported preprocessing schema {self.schema!r}"
            )
        limit = self.chirp.n_samples
        if not 0 < self.crop_bins <= limit:
            raise ValueError(f"crop_bins must be in 1..{limit}, got {self.crop_bins}")
        if self.window_frames < 1:
            raise ValueError("window_frames must be positive")
        if self.bin_zero_offset < 0:
            raise ValueError("bin_zero_offset cannot be negative")
        if self.bin_zero_offset + self.crop_bins > limit:
            raise ValueError(
                f"crop of {self.crop_bins} bins from bin {self.bin_zero_offset} "
                f"runs past the {limit} bins one matched-filter frame holds"
            )
        if self.differential_phase is not DifferentialPhase.LATER:
            raise PreprocessingMismatchError(
                "this release builds causal windows only, so a differential "
                "must be attributed to the later frame of its pair; "
                f"{self.differential_phase.value!r} would place a frame "
                "recorded after the predicted pose inside the window"
            )
        if self.differential_lag < 1:
            raise PreprocessingMismatchError(
                "differential_lag must be at least 1: a differential "
                "attributed to the later frame of its pair cannot align with "
                "an original recorded before the difference existed"
            )

    @property
    def sample_rate(self) -> int:
        return self.chirp.sample_rate

    @property
    def n_bins(self) -> int:
        """Bins one matched-filter frame produces, before cropping."""
        return self.chirp.n_samples

    @property
    def crop_slice(self) -> slice:
        """Rows kept from a full profile, origin included."""
        return slice(self.bin_zero_offset, self.bin_zero_offset + self.crop_bins)

    @property
    def window_shape(self) -> tuple[int, int, int]:
        """The tensor both paths must emit: original and differential."""
        return (2, self.crop_bins, self.window_frames)

    def differential_column(self, original_column: int) -> int:
        """Differential index aligned with an original index."""
        return original_column - self.differential_lag

    def crop_rows(self, array: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the crop along the bin axis, which is always axis zero."""
        return np.asarray(array[self.crop_slice], dtype=np.float32)

    def normalise_window(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalise one assembled window, identically on both paths.

        Both callers go through this method rather than through their own
        arithmetic, so the two cannot round differently: the same scalar
        divides the same float32 array in the same order.

        An all-zero window is returned untouched rather than divided by an
        epsilon, because an epsilon turns a silent window into amplified
        numerical noise that looks exactly like signal.
        """
        values = np.ascontiguousarray(window, dtype=np.float32)
        if self.window_normalisation is Normalisation.NONE or values.size == 0:
            return values
        if self.window_normalisation is Normalisation.PEAK:
            scale = float(np.max(np.abs(values)))
        else:
            scale = float(np.linalg.norm(values.reshape(-1)))
        if scale <= 0.0 or not np.isfinite(scale):
            return values
        return np.ascontiguousarray(values / np.float32(scale), dtype=np.float32)

    def with_bin_zero_offset(self, offset: int) -> PreprocessingDescriptor:
        """The same contract, re-origined after a measurement."""
        return replace(self, bin_zero_offset=offset)

    def with_differential_lag(self, lag: int) -> PreprocessingDescriptor:
        """The same contract, re-aligned after a measurement."""
        return replace(self, differential_lag=lag)

    def with_window_frames(self, frames: int) -> PreprocessingDescriptor:
        """The same contract over a shorter or longer window.

        Exists for tests and short experiments, which need windows a synthetic
        session can actually fill. Changing this changes what a checkpoint is,
        which is why it produces a new descriptor rather than mutating one.
        """
        return replace(self, window_frames=frames)

    def describe(self) -> str:
        """One line, for an error message that has to be actionable."""
        return (
            f"chirp {self.chirp.f_start:.0f}-{self.chirp.f_end:.0f} Hz at "
            f"{self.chirp.sample_rate} Hz, {self.chirp.n_samples} samples; "
            f"crop {self.crop_bins} bins from bin {self.bin_zero_offset}; "
            f"window {self.window_frames} frames; differential lag "
            f"{self.differential_lag} attributed to the "
            f"{self.differential_phase.value} frame; window normalisation "
            f"{self.window_normalisation.value}"
        )

    def require_match(
        self,
        other: PreprocessingDescriptor,
        *,
        ours: str = "this pipeline",
        theirs: str = "the checkpoint",
    ) -> None:
        """Raise unless two descriptors are the same contract.

        Field by field rather than a bare inequality, because the whole value
        of catching this is being told which convention moved.
        """
        if self == other:
            return
        differences = [
            f"{name}: {ours} says {getattr(self, name)!r}, "
            f"{theirs} says {getattr(other, name)!r}"
            for name in (
                "chirp",
                "crop_bins",
                "window_frames",
                "bin_zero_offset",
                "differential_lag",
                "differential_phase",
                "window_normalisation",
            )
            if getattr(self, name) != getattr(other, name)
        ]
        raise PreprocessingMismatchError(
            "preprocessing contracts disagree, so the model would be fed "
            "something other than what it was trained on: "
            + "; ".join(differences)
        )

    def to_jsonable(self) -> dict[str, object]:
        """JSON-safe form, flat enough to read in a diff."""
        return {
            "schema": self.schema,
            "f_start_hz": self.chirp.f_start,
            "f_end_hz": self.chirp.f_end,
            "duration_s": self.chirp.duration_s,
            "sample_rate": self.chirp.sample_rate,
            "crop_bins": self.crop_bins,
            "window_frames": self.window_frames,
            "bin_zero_offset": self.bin_zero_offset,
            "differential_lag": self.differential_lag,
            "differential_phase": self.differential_phase.value,
            "window_normalisation": self.window_normalisation.value,
        }

    @classmethod
    def from_jsonable(cls, value: dict[str, Any]) -> PreprocessingDescriptor:
        if value.get("schema") != PREPROCESSING_SCHEMA:
            raise PreprocessingMismatchError(
                f"unsupported preprocessing schema {value.get('schema')!r}"
            )
        try:
            return cls(
                chirp=ChirpConfig(
                    f_start=float(value["f_start_hz"]),
                    f_end=float(value["f_end_hz"]),
                    duration_s=float(value["duration_s"]),
                    sample_rate=int(value["sample_rate"]),
                ),
                crop_bins=int(value["crop_bins"]),
                window_frames=int(value["window_frames"]),
                bin_zero_offset=int(value["bin_zero_offset"]),
                differential_lag=int(value["differential_lag"]),
                differential_phase=DifferentialPhase(value["differential_phase"]),
                window_normalisation=Normalisation(value["window_normalisation"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PreprocessingMismatchError(
                f"invalid preprocessing descriptor: {error}"
            ) from error


WATCHHAND_PREPROCESSING: Final = PreprocessingDescriptor(
    chirp=WATCHHAND_CHIRP,
    crop_bins=60,
    window_frames=96,
)
"""The contract this project trains and captures under.

Sixty bins is about 21.4 cm, wrist to middle fingertip, from the paper.
Ninety-six frames is 1.2 s at 80 frames per second, also from the paper. The
two undocumented fields keep their conservative defaults until something
measures them; see the fields for what measures them and what happens when the
measurement disagrees.
"""
