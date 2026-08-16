"""Synthetic claps, built from the physics the pipeline claims to invert.

Everything here is offline and seeded. A clap is modelled as a single impulse
exciting a damped second-order resonator, plus a short broadband contact
click, which is the structure Fu et al. (2025) and Peltola et al. (2007)
describe: an excitation at contact, a Helmholtz mode, and heavy soft-tissue
damping.

Per-clap variation is applied in log space to the centre frequency and the
bandwidth, because within-class scatter is multiplicative rather than
additive. The bandwidth scatter is deliberately large relative to the gap
between neighbouring modes, which is what reproduces the confusion structure
Jylha and Erkut reported and is the whole point of the merged taxonomy tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wristsonar.clap.configuration import (
    MODE_PROTOTYPES,
    ClapFeatures,
    ClapMode,
    Label,
    features_from_fit,
)
from wristsonar.clap.resonance import fit_resonance

SAMPLE_RATE = 48_000

FREQUENCY_JITTER = 0.045
"""Standard deviation of log10 centre frequency within a class, about 11
percent, which keeps the two reported confusion pairs genuinely confusable
without smearing the whole taxonomy."""

BANDWIDTH_JITTER = 0.14
"""Standard deviation of log10 bandwidth within a class, about 38 percent."""


@dataclass(frozen=True, slots=True)
class SynthClap:
    """One synthetic clap and the parameters it was drawn with."""

    samples: NDArray[np.float64]
    mode: ClapMode
    centre_frequency_hz: float
    bandwidth_hz: float
    sample_rate: int

    @property
    def decay_rate_s(self) -> float:
        return math.pi * self.bandwidth_hz


def resonator_impulse_response(
    centre_frequency_hz: float,
    bandwidth_hz: float,
    sample_rate: int,
    n_samples: int,
) -> NDArray[np.float64]:
    """Impulse response of one damped second-order mode.

    Written out directly rather than filtered, so the test fixture does not
    share an implementation with the code under test.
    """
    alpha = math.pi * bandwidth_hz
    index = np.arange(n_samples, dtype=np.float64)
    t = index / sample_rate
    return np.exp(-alpha * t) * np.sin(2.0 * math.pi * centre_frequency_hz * t)


def synth_clap(
    mode: ClapMode,
    rng: np.random.Generator,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_s: float = 0.060,
    frequency_jitter: float = FREQUENCY_JITTER,
    bandwidth_jitter: float = BANDWIDTH_JITTER,
    click_level: float = 0.35,
    click_tau_s: float = 0.0004,
    amplitude: float = 1.0,
) -> SynthClap:
    """One clap in the given Repp mode, with within-class variation."""
    nominal_f, nominal_bw = MODE_PROTOTYPES[mode]
    centre = float(nominal_f * 10.0 ** rng.normal(0.0, frequency_jitter))
    bandwidth = float(nominal_bw * 10.0 ** rng.normal(0.0, bandwidth_jitter))

    n = round(duration_s * sample_rate)
    ring = resonator_impulse_response(centre, bandwidth, sample_rate, n)
    peak = float(np.max(np.abs(ring)))
    if peak > 0.0:
        ring = ring / peak

    t = np.arange(n, dtype=np.float64) / sample_rate
    click = rng.normal(0.0, 1.0, n) * np.exp(-t / click_tau_s)
    click_peak = float(np.max(np.abs(click)))
    if click_peak > 0.0:
        click = click / click_peak

    samples = amplitude * (ring + click_level * click)
    return SynthClap(
        samples=samples,
        mode=mode,
        centre_frequency_hz=centre,
        bandwidth_hz=bandwidth,
        sample_rate=sample_rate,
    )


def synth_feature_set(
    rng: np.random.Generator,
    n_per_mode: int,
    *,
    modes: tuple[ClapMode, ...] | None = None,
    sample_rate: int = SAMPLE_RATE,
    noise_rms: float = 0.004,
) -> tuple[list[ClapFeatures], list[Label]]:
    """Fit features for ``n_per_mode`` claps of every requested mode."""
    chosen = tuple(ClapMode) if modes is None else modes
    features: list[ClapFeatures] = []
    labels: list[Label] = []
    for mode in chosen:
        for _ in range(n_per_mode):
            clap = synth_clap(mode, rng, sample_rate=sample_rate)
            noisy = clap.samples + rng.normal(0.0, noise_rms, clap.samples.size)
            fit = fit_resonance(noisy, sample_rate)
            features.append(features_from_fit(fit))
            labels.append(mode)
    return features, labels


def synth_recording(
    modes: list[ClapMode],
    rng: np.random.Generator,
    *,
    sample_rate: int = SAMPLE_RATE,
    gap_s: float = 0.5,
    lead_s: float = 1.0,
    tail_s: float = 0.5,
    noise_rms: float = 0.01,
    noise_ramp_db: float = 0.0,
    snr_db: float = 34.0,
) -> tuple[NDArray[np.float64], list[float], list[ClapMode]]:
    """A recording containing one clap per entry in ``modes``.

    ``noise_ramp_db`` sweeps the background level linearly in dB across the
    whole recording, so the detector meets a moving floor. Clap amplitude
    tracks the local floor at ``snr_db``, which is the realistic case: a
    person in a louder room is not quieter.

    Returns the signal, the true clap times in seconds, and the modes.
    """
    lead = round(lead_s * sample_rate)
    gap = round(gap_s * sample_rate)
    tail = round(tail_s * sample_rate)
    total = lead + gap * len(modes) + tail

    ramp_db = np.linspace(0.0, noise_ramp_db, total)
    floor = noise_rms * 10.0 ** (ramp_db / 20.0)
    signal = rng.normal(0.0, 1.0, total) * floor

    times: list[float] = []
    for index, mode in enumerate(modes):
        start = lead + index * gap
        local_floor = float(floor[start])
        amplitude = local_floor * 10.0 ** (snr_db / 20.0)
        clap = synth_clap(mode, rng, sample_rate=sample_rate, amplitude=amplitude)
        stop = min(total, start + clap.samples.size)
        signal[start:stop] += clap.samples[: stop - start]
        times.append(start / sample_rate)

    return signal, times, modes
