"""A forward model: point reflectors in, a recording out.

This module exists so the rest of the layer can be checked against a known
answer. Put a reflector at 8.3 cm, run the whole pipeline, and assert 8.3 cm
comes back. Without that, every function here is only checkable against
itself, and a consistent sign error in the delay convention would survive
indefinitely because the analysis code and the test code would share it.

The model is deliberately thin, and its omissions are the honest boundary of
what any test in this layer can claim:

  - Reflectors are points. A hand is not, and a real fingertip returns a
    smeared response from a surface comparable in size to the 1.76 cm
    wavelength, plus diffraction rather than reflection from anything smaller.
  - Propagation is ideal. No 1/r spreading loss, no frequency-dependent air
    absorption (which is far from negligible at 20 kHz), no transducer
    response, no multipath, no room.
  - The channel is linear and time-invariant within a frame. A hand moving at
    1 m/s shifts a 20 kHz carrier by about 120 Hz, which this ignores.

So a test built on this model can prove the pipeline recovers the range it was
given. It cannot prove the pipeline works on a wrist, and nothing in this
package should be read as claiming otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.signal.chirp import DEFAULT_TUKEY_ALPHA, sample_chirp_at
from wristsonar.types import SPEED_OF_SOUND, ChirpConfig

__all__ = ["Reflector", "simulate_frame", "simulate_recording"]


@dataclass(frozen=True, slots=True)
class Reflector:
    """A point scatterer at a known reflector range with a known amplitude.

    `amplitude` is dimensionless and is the factor the transmitted waveform is
    multiplied by, so it lands on the same scale that `correlate_fft` is
    calibrated to report. A reflector of amplitude 0.4 produces an envelope
    peak of 0.4.
    """

    range_m: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.range_m < 0.0:
            raise ValueError("a reflector cannot be at negative range")
        if not np.isfinite(self.amplitude):
            raise ValueError("reflector amplitude must be finite")

    @property
    def delay_s(self) -> float:
        """Round-trip flight time. The factor of two is the there and back."""
        return 2.0 * self.range_m / SPEED_OF_SOUND


def _frame_signal(
    config: ChirpConfig,
    reflectors: Sequence[Reflector],
    alpha: float,
    alias_beyond_range: bool,
) -> NDArray[np.float64]:
    t = np.arange(config.n_samples, dtype=np.float64) / config.sample_rate
    out = np.zeros(config.n_samples, dtype=np.float64)
    for r in reflectors:
        out += r.amplitude * sample_chirp_at(config, t - r.delay_s, alpha)
        if alias_beyond_range:
            # Steady-state contribution of the previous chirp in the train.
            # This is what makes a reflector past max_unambiguous_range show
            # up as a ghost at range - max_unambiguous_range, which is a real
            # property of back-to-back FMCW and not a simulator artefact.
            out += r.amplitude * sample_chirp_at(
                config, t - r.delay_s + config.duration_s, alpha
            )
    return out


def _add_noise(
    signal: NDArray[np.float64],
    snr_db: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Additive white Gaussian noise at a stated SNR.

    SNR is defined against the mean square of the whole noiseless recording,
    not against the peak and not per frame. That choice matters for reading
    the low-SNR tests: because the echo occupies only the fraction of each
    frame after its delay, and because matched filtering concentrates the
    sweep's time-bandwidth product into one peak, a nominal SNR of -10 dB here
    still leaves a usable peak. The processing gain is the point of FMCW.
    """
    power = float(np.mean(signal**2))
    if power <= 0.0:
        raise ValueError("cannot set an SNR against a silent signal")
    sigma = float(np.sqrt(power / (10.0 ** (snr_db / 10.0))))
    return np.asarray(signal + rng.normal(0.0, sigma, signal.shape), dtype=np.float64)


def simulate_frame(
    config: ChirpConfig,
    reflectors: Sequence[Reflector],
    *,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    alpha: float = DEFAULT_TUKEY_ALPHA,
    alias_beyond_range: bool = False,
) -> NDArray[np.float64]:
    """One chirp period of received signal from a static set of reflectors."""
    signal = _frame_signal(config, reflectors, alpha, alias_beyond_range)
    if snr_db is None:
        return signal
    generator = rng if rng is not None else np.random.default_rng()
    return _add_noise(signal, snr_db, generator)


def simulate_recording(
    config: ChirpConfig,
    frames: Sequence[Sequence[Reflector]],
    *,
    static: Sequence[Reflector] = (),
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    alpha: float = DEFAULT_TUKEY_ALPHA,
    alias_beyond_range: bool = False,
) -> NDArray[np.float64]:
    """A continuous recording of back-to-back chirps.

    `frames` gives the moving scene, one list of reflectors per chirp period.
    `static` is added identically to every frame and stands for the device
    body, the strap and the wrist: the reflections that dominate the raw
    profile and that the differential path exists to remove. Keeping it a
    separate argument rather than asking callers to repeat it in every frame
    is what makes "identical in every frame" true by construction, which is
    the condition the differential test is actually about.

    Noise is drawn once over the whole recording, so it is independent
    frame to frame and therefore does not cancel under differencing. That is
    the realistic case and the one worth testing: a differential profile
    suppresses static structure while roughly doubling noise variance.
    """
    if not frames:
        raise ValueError("a recording needs at least one frame")
    blocks = [
        _frame_signal(config, [*scene, *static], alpha, alias_beyond_range)
        for scene in frames
    ]
    recording = np.concatenate(blocks)
    if snr_db is None:
        return recording
    return _add_noise(
        recording, snr_db, rng if rng is not None else np.random.default_rng()
    )
