# Architecture

Five layers, in the order data flows through them. The offline and host-side
layers are implemented; Wear OS hardware validation and a trained public
checkpoint remain external validation work. A debug APK was assembled against
Android SDK 35 at one point, but the Wear sources have been rewritten since and
have not been compiled again, so treat the device half as unbuilt. This document
distinguishes those states so the architecture is never mistaken for a live
device result.

## The shape of the thing

```
smartwatch (stock)
  -> capture: 18-21 kHz FMCW chirps, 12.5 ms, 48 kHz duplex
  -> echo profiles: cross-correlation against the known chirp, plus differential
  -> model: small ViT or CNN over the echo-profile image
  -> 21 joints x 3D, plus a calibration state
  -> sinks: OpenXR hand joints, Blender, WebSocket, MIDI
```

Two things cut across all five rather than sitting in one of them.
`wristsonar.protocol` is the rule that anything producing a reportable number
carries the conditions it was produced under; see EVALUATION.md.
`wristsonar.preprocess` is the rule that the training pipeline and the live
pipeline build the same tensor; see below.

## Status

| Layer | Module | Status |
|---|---|---|
| Core types | `wristsonar.types` | Written |
| Protocol | `wristsonar.protocol` | Written |
| Preprocessing contract | `wristsonar.preprocess` | Written; two of its fields are declared conservatively rather than measured on the real release |
| Signal | `wristsonar.signal` | Written and synthetic-verified |
| Evaluation | `wristsonar.eval` | Written |
| Data ingest | `wristsonar.data` | Written, manifest-gated |
| Model input | `wristsonar.model` | Written; training needs public data |
| Calibration | `wristsonar.eval.calibration` | Written |
| Capture health and transport | `wristsonar.capture` | Host side written; the Wear sender has not been compiled since it was last changed, and has never run on a watch |
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

## The preprocessing contract

This is the piece that stops a trained model and a live watch disagreeing about
what their input is, and it is worth understanding before anything downstream
makes sense.

A model is a function of its weights and of the preprocessing those weights were
fitted under. For most of this project's life the second argument was written
down nowhere. The convention lived twice, once inside `SessionData.windows` on
the training side and once inside `EchoWindowAssembler` on the capture side, and
the two were kept in step by nothing except having been written on the same
afternoon. Every way two copies of one convention can drift is silent. A live
window whose peak is 6553 where training taught the model to expect roughly 1.0
still has the right shape, the right dtype, and still produces well formed pose
JSON. Nothing downstream can tell that the answer is meaningless.

So the contract is one frozen object, `PreprocessingDescriptor` in
`wristsonar.preprocess`, and it travels. It states the chirp, the number of bins
kept and the array index they are kept from, the window length in frames, which
column of a shipped differential array belongs with which original, which frame
of a differenced pair the difference is attributed to, and what normalisation is
applied to the assembled window. Both window builders take one and neither has
geometry defaults of its own any more. `CheckpointMetadata` carries one, so a
weight file records the preprocessing it was trained under, and
`load_torch_checkpoint` compares the two before it touches the weights rather
than after. The checkpoint and bundle schemas moved to
`wristsonar.checkpoint/2` and `checkpoint-bundle/2` for this, and a version 1
sidecar is refused rather than read with the missing fields defaulted, because
defaulting them is exactly the confident wrong answer the object exists to
prevent.

It sits above `wristsonar.types` and `wristsonar.signal` and below
`wristsonar.data` and `wristsonar.capture`, which is what lets the training path
and the capture path both import it without importing each other.

Two of its fields describe things nobody documented, so they are measured rather
than assumed. `bin_zero_offset` is the array index that represents range zero,
and `differential_lag` is where in WatchHand's shipped differential array the
difference belonging with a given original sits. `estimate_bin_zero_offset` and
`estimate_differential_lag` measure both from the shipped arrays, and
`SessionData.verify_preprocessing` runs them per session so a corpus build stops
when the measurement contradicts the descriptor a checkpoint will carry. The lag
is the weaker of the two: the released differential is not reproducible from the
released original, the measurement therefore returns nothing, and the declared
value stands on an argument rather than on evidence. See DATA.md.

The claim that the two paths agree is a measurement, not an assertion.
`tests/test_preprocessing_contract.py` pushes identical synthetic PCM through
the training corpus builder and through the live assembler and requires the two
tensors to match, which is the only form of the claim that can fail.

## Capture layer

The device-independent capture contract is written. `PcmSynchronizer` turns
arbitrary callback boundaries into whole chirp frames, and the way it does that
changed: alignment is now carried by counting samples forward from a single
correlation lock, and the lock is re-scored every `relock_frames`, one second by
default. Timestamps are not the alignment authority and cannot be, because a
callback stamp is taken after `AudioRecord.read` returns and carries
milliseconds of scheduling jitter, while one range bin is one sample, 20.8
microseconds. They keep two coarse jobs: a jump past `gap_tolerance_s`, 30 ms,
ends the segment and forces a fresh lock, and samples delivered divided by
elapsed monotonic time over a two second baseline gives `observed_sample_rate`.
That division is the only honest resampling detector on the host side. A stream
that cannot lock says so rather than waiting quietly: `locked` is never true
while nothing is being emitted, `status` carries a printable reason, and
`LiveCaptureProcessor` raises once it has consumed `lock_timeout_s` of unlocked
audio.

`DuplexValidator` no longer claims to detect resampling, because it never could.
It reads the rate the watch declares, and a resampled stream declares the same
number. What it checks is what it can see: that the declared rate does not change
mid-session, that the frame size matches the chirp, and that the segment is
continuous. `RawPcmWireFrame` supplies a versioned, length-delimited raw PCM
transport between a watch and this host layer.

The Wear OS sender in `wear/` plays and records simultaneously. It opens
`UNPROCESSED` where the device advertises
`PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED` and falls back to
`VOICE_RECOGNITION`, which is a weaker claim and is documented as one. It
pre-rolls the output ring before `play()` so that the guaranteed startup
underrun stops counting against the session, tolerates later underruns up to a
budget instead of dying on the first, recovers the effective sample rate from
`AudioTimestamp`, and hands the TCP write to a bounded queue on its own thread
so that a stalled socket drops frames rather than overflowing the record ring.
None of that has been compiled since it was written, and none of it has run on a
watch. The hard parts remain duplex timing and gain control: automatic gain
control across 18 to 21 kHz destroys the amplitude relationships the
differential echo profile is made of, `AutomaticGainControl.create()` returning
null usually means the gain control is in the HAL where it cannot be observed,
and no part of this stack can detect that it happened.

LLAP (MIT licence, github.com/Samsonsjarkal/LLAP) is the reference
implementation for the phase-tracking half of this on iOS and is worth reading
before trusting anything here on a real device.

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

A cropped profile carries the one-way distance of its own bin zero as
`range_offset_m`, and the peak helpers take it, because a crop otherwise reads
every range low by the near edge. `peak_ranges_of` and `strongest_peak_of` take
the profile itself and so cannot be called wrongly; the two-argument forms are
still there for bare arrays. `differential_profiles` carries the offset through
and refuses inputs that disagree about it, since bin k of two profiles cropped
to different near edges is two different distances and subtracting them is not a
difference in time.

The noise floor is the other thing worth knowing about. It used to be a global
median, which works while most bins are noise and fails on exactly the window
this project recommends: on a hand-sized crop the median sits on the signal, the
threshold rises above the peak, and the detector goes silent on the one window
it is aimed at. It is now a low quantile per block of 96 bins, blocks combined
by median, rescaled by the Rayleigh quantile-to-median factor so that thresholds
calibrated against the old median still mean what they meant, and capped at the
profile maximum. The blocking is not decoration. A quantile taken over a whole
profile lands in the correlation taper, where noise decays as sqrt(1 - k/N)
because only N minus k samples still overlap at lag k, and reads far too low,
which is a worse failure than the one it fixes.

## Model layer

Written as a causal window builder plus a compact optional Torch CNN baseline.
The window shape is 2 by 60 by 96 because that is what the shared descriptor
says, not because this layer decided it. The public dataset must still be
downloaded and its landmark sidecars generated before any training claim can be
made.

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
guards, and the report writer. Every reportable value carries a `Protocol`. One
`Report` holds one split, since rows in a table have to be comparable to be a
comparison at all, and `MultiSplitReport` holds one `Report` per split and is
what a complete result looks like.

It is documented in full in EVALUATION.md rather than here, because it is the
argument of the project rather than a component of it.

## Data

Dataset ingest is written and manifest-gated. DATA.md covers the WatchHand
release, its format, and what must be verified before anything downstream can
be trusted. The single largest risk is unchanged in kind and narrower in scope:
the release is echo profiles rather than raw audio, and if those profiles were
produced by processing that cannot be reproduced from a live device, the capture
layer and the model layer will never meet. The shared descriptor above is what
turns that from a risk nobody can act on into two named parameters, one of which
is measurable against the shipped arrays and one of which, so far, is not. See
ROADMAP.md.
