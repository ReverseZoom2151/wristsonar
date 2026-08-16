# Architecture

Five layers, in the order data flows through them. The offline and host-side
layers are implemented; Wear OS compilation and hardware validation, plus a
trained public checkpoint, remain external validation work. This document distinguishes those states so
the architecture is never mistaken for a live device result.

## The shape of the thing

```
smartwatch (stock)
  -> capture: 18-21 kHz FMCW chirps, 12.5 ms, 48 kHz duplex
  -> echo profiles: cross-correlation against the known chirp, plus differential
  -> model: small ViT or CNN over the echo-profile image
  -> 21 joints x 3D, plus a calibration state
  -> sinks: OpenXR hand joints, Blender, WebSocket, MIDI
```

Cutting across all five is `wristsonar.protocol`, which is not a layer but a
constraint: anything that produces a reportable number carries the conditions it
was produced under. See EVALUATION.md.

## Status

| Layer | Module | Status |
|---|---|---|
| Core types | `wristsonar.types` | Written |
| Protocol | `wristsonar.protocol` | Written |
| Signal | `wristsonar.signal` | Written and synthetic-verified |
| Evaluation | `wristsonar.eval` | Written |
| Data ingest | `wristsonar.data` | Written, manifest-gated |
| Model input | `wristsonar.model` | Written; training needs public data |
| Calibration | `wristsonar.eval.calibration` | Written |
| Capture health and transport | `wristsonar.capture` | Written; needs Wear OS build and hardware validation |
| Host inference and sinks | `wristsonar.runtime` | Capture wire format, strict inference, JSONL/TCP and Blender written; OpenXR pending |

Everything below marked as intent describes a design decision, not a
description of running code. The repository tree is the authority on what
exists; this table is a snapshot and will drift.

## Core types

`wristsonar.types` holds the shapes every other module agrees on: `ChirpConfig`,
`EchoProfile`, `HandPose`, the joint-name tuple, and the speed of sound. They
are deliberately plain, because the interesting constraints live in the physics
rather than in the containers.

Three of the decisions there are load-bearing rather than incidental.
`ChirpConfig` defaults to an 18 to 21 kHz sweep at 48 kHz in 12.5 ms frames, and
it refuses a sweep whose upper edge crosses Nyquist, because that sweep would
alias rather than fail loudly at runtime. It exposes `range_resolution`,
`wavelength_at_centre` and `max_unambiguous_range` as properties, which puts the
5.7 cm and 17.6 mm figures from PHYSICS.md inside the code rather than in a
comment somewhere. And `HandPose` is wrist-relative by default, because a
wrist-mounted sensor has no idea where the wrist is in the world, and claiming
global position from it would be inventing information.

The joint list is twenty-one landmarks, the wrist plus four per digit, ordered
to match the MediaPipe hand convention so that comparison against camera-based
systems needs no remapping table.

## Capture layer

The device-independent capture contract is written. `DuplexValidator` rejects
streams that silently resample 48 kHz audio, change the 600-sample chirp frame,
or contain discontinuities. `PcmSynchronizer` turns arbitrary callback
boundaries into periodic 600-sample chirp frames using the direct acoustic path
only for timing. A discontinuity discards partial history and forces a new
lock. `RawPcmWireFrame` supplies a versioned, length-delimited raw PCM transport
between a watch and this host layer. The Wear OS sender is in `wear/`; its
Android build and hardware validation remain pending.

A Wear OS application that plays and records simultaneously. The hard parts are
duplex timing, disabling automatic gain control and noise suppression where the
platform allows it, and detecting when the operating system has silently
resampled or applied processing anyway. That last one matters more than it
sounds: the differential echo profile is a difference of amplitudes, so a
time-varying gain manufactures motion that did not happen.

LLAP (MIT licence, github.com/Samsonsjarkal/LLAP) is the reference
implementation for the phase-tracking half of this on iOS and is worth reading
before writing any of it.

This layer is where platform reality bites, and it is deliberately late in the
build order. See ROADMAP.md, phase 3.

## Signal layer

Written and property-tested against synthetic echoes with known ranges.

Pure functions, no device required, fully testable offline: chirp synthesis,
matched filtering, echo-profile construction, differential profiles, and
cropping to the range window that can plausibly contain a hand. The declared
surface covers chirp generation and windowing, FFT correlation and envelope
extraction with sub-bin peak refinement by parabolic interpolation, frame
segmentation, profile construction, normalisation and differencing.

Two conventions in this layer exist because getting them wrong is silent rather
than loud. A range always means the one-way reflector distance, with the factor
of two for the out-and-back path already absorbed into the metres-per-bin
conversion, so that bin count times metres per bin equals the chirp's maximum
unambiguous range. And a forward simulator is a first-class part of the layer
rather than test scaffolding, because a sign error in a delay or an off-by-one
in a lag mapping produces plausible-looking numbers rather than a crash, and the
only defence against that is pinning the pipeline against synthetic echoes with
known delay and amplitude.

## Model layer

Written as a causal 2 by 60 by 96 window builder plus a compact optional Torch
CNN baseline. The public dataset must still be downloaded and its landmark
sidecars generated before any training claim can be made.

A small vision transformer or CNN over the echo-profile image, starting by
reproducing the published FastViT-T12 setup on the WatchHand data. The rule for
this layer is that it does not innovate first. The contribution is that it runs
at all, with released weights, before it is that it is better than anything.
Nothing in this field currently ships weights. See PRIOR_ART.md.

Output is twenty-one joints in three dimensions, wrist-relative, as a `HandPose`.

## Calibration layer

Written in `wristsonar.eval.calibration` as a reported axis, not a hidden
fine-tuning setting.

The field norm is roughly two minutes of per-user fine-tuning, and it buys a
lot: EchoWrist goes from 12.2 mm cold to 6.92 mm after one minute, plateauing at
twenty; WatchHand goes from 9.96 mm zero-shot on an unseen posture to 7.65 mm
after one two-minute session.

Because it buys that much, calibration is a first-class, measurable, reportable
thing rather than a hidden preprocessing step. Architecturally that means the
calibration state is an explicit object with an explicit cost in minutes, the
protocol carries that cost as a required field, and the evaluation harness
sweeps it as an axis rather than picking a value. The user-facing flow in the
application and the swept axis in the harness are the same mechanism, which is
the point: what gets demonstrated is what gets measured.

## Sink layer

The portable `PoseFrame`, versioned capture wire format, strict host inference,
JSON Lines/TCP sinks and Blender adapter are written. OpenXR remains pending.

OpenXR hand-joint output makes the system immediately useful inside existing XR
software without a bespoke integration. Blender is the demo, because a hand
moving on screen is the only convincing artifact. A WebSocket firehose makes it
hackable by anyone who does not want to link against either. MIDI is listed in
the plan's diagram and is the cheapest way to make the thing an instrument.

## Evaluation layer

Written, and the actual product.

`wristsonar.eval` covers split construction with leakage assertions, metrics
including MPJPE, PA-MPJPE, PCK and per-joint breakdowns with fingertips called
out, the three trivial baselines, the calibration sweep, the shortcut-learning
guards, and the report writer. Every reportable value carries a `Protocol`.

It is documented in full in EVALUATION.md rather than here, because it is the
argument of the project rather than a component of it.

## Data

Dataset ingest is written and manifest-gated. DATA.md covers the WatchHand
release, its format, and what must be verified before anything downstream can
be trusted. The single largest risk remains: the release is echo profiles rather
than raw audio, and if those profiles were produced by processing that cannot be
reproduced from a live device, the capture layer and model layer will never meet.
See ROADMAP.md.
