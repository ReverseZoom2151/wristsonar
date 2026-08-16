# Data

## Verdict on the risk

**Can a live capture app reproduce the WatchHand dataset's input
representation? Yes. Confidence: high, roughly 85 percent.**

The premise of the risk was correct: WatchHand ships **no raw audio**. Its
README says so plainly under Known Issues. Every session is a precomputed
C-FMCW echo profile. The conclusion drawn from that premise does not follow,
though, because the preprocessing is specified tightly enough to reimplement,
and the specification is self-consistent in a way that cannot be luck.

What pins it:

| Parameter | Value | Where it is stated |
|---|---|---|
| Transmit waveform | `fmcw18000_b3000_l600_s48k.wav` | `config.json`, every session |
| Sweep | linear 18 to 21 kHz | filename, README, paper |
| Sweep length | 600 samples, 12.5 ms | filename, `frame_length` in config |
| Sample rate | 48 kHz, 16-bit PCM | config, README, paper |
| Bandpass | 5th-order Butterworth, 18 to 21 kHz | paper; `bandpass_range` in config |
| Correlation | `np.correlate` of one tx sweep against filtered rx | paper, Grab-n-Go README |
| Range bin | one correlation lag, `c / (2 * fs)` = 3.5729 mm | paper quotes 3.57 mm |
| Profile shape | `(600, n_frames)`, float32 | verified from file sizes |
| Differential | frame-to-frame difference of correlation magnitude | paper |
| Model crop | nearest 60 bins, 21.4 cm | paper |
| Model window | 96 frames, 1.2 s at 80 fps | paper |
| Model input | 2 channels, original and differential | paper |

The strongest single piece of evidence is arithmetic that nobody wrote down
and that only works if the pipeline is what it is described to be. WatchHand
sends 600 bins at 48 kHz; `343 / (2 * 48000)` is 3.573 mm, and the paper
quotes 3.57 mm. The sibling Grab-n-Go release sends 600 bins at 50 kHz and
quotes a 2.06 m range; `600 * 343 / (2 * 50000)` is 2.058 m. Two independent
releases, two different sample rates, both consistent with "one range bin per
correlation lag, no decimation, no interpolation, no windowing". A pipeline
doing anything else would not land on both numbers.

File sizes confirm the container. `audio001_fmcw_16bit_profiles.npy` for
`sub1_samsung_left_video` is 65,275,328 bytes. Subtract the 128-byte npy
header, divide by 4 and by 600, and you get exactly 27,198 frames. The
differential for the same session is 65,270,528 bytes, exactly 27,196 frames.
Integer on both, so: float32, C-order, `(600, n_frames)`, and the differential
is genuinely shorter than the original.

### The 15 percent

Four details are not documented anywhere. None of them is fatal on its own, and
each has a named home in `wristsonar.preprocess.PreprocessingDescriptor` rather
than living as a default inside whichever function happened to need it:

1. **Absolute amplitude scale.** Whether the 16-bit PCM was correlated as
   int16, as float in [-1, 1], or something else is not stated, so the shipped
   profiles carry an unknown constant factor. Any per-profile normalisation
   cancels it. `SessionData.echo_profile` normalises by default for exactly
   this reason, and `SessionData.windows` normalises the assembled window
   according to `window_normalisation`, which is peak. The paper also
   normalises before feeding the model, so nothing is lost.
2. **Which correlation lag is bin zero.** `np.correlate` mode and the crop
   origin are unstated, so a live pipeline could be offset by a constant number
   of bins, which a model reads as a differently sized hand. This one is
   measurable rather than guessable: the speaker-to-microphone direct path is
   a large static peak at essentially zero range, so `estimate_bin_zero_offset`
   takes the median over time to kill hand motion and finds the near-field
   peak. It is no longer advisory. `SessionData.verify_preprocessing` runs it
   on every session the corpus builder touches and refuses to build when the
   measurement disagrees with `bin_zero_offset`, naming the value it measured.
3. **How the shipped differential lines up with the original beside it.** See
   below; this is the one the shapes cannot settle.
4. **Whether profiles are signed correlation or magnitude.** The paper
   describes the differential as a subtraction of magnitudes, which implies
   magnitude. Not stated outright.

Three of the four are resolvable empirically against the shipped profiles,
because the profiles are shipped. That is what makes this recoverable rather
than speculative. The honest close-out is a one-off validation: capture on a
Samsung Galaxy Watch 7, run the reimplemented pipeline, and check that the
static near-field structure and bin scaling match a WatchHand session. Do that
before trusting a cross-device number.

### The differential alignment, and why the declared lag is 2

The shipped differential is two frames shorter than the original, not the one
frame a plain frame-to-frame difference loses. For `sub1_samsung_left_video`
that is 27,196 columns against 27,198. So a difference was taken and then one
further column was trimmed off an end nobody recorded, and the shapes alone
cannot say which end.

Note carefully what the lag is. It is an index offset into somebody else's
array, not a delay in the signal. Under the causal convention this project
uses, the differential belonging with original column i is always the
difference ending at i. The lag says only where that quantity sits in the array
WatchHand shipped, so the training path reads differential column i minus lag,
and the first few originals have no differential aligned with them and are
skipped rather than paired with something else. The live path builds the
difference itself, has no shipped array to index and therefore no lag to apply,
which is precisely why the two paths can produce the same tensor.

Trimmed at the front the lag is 2, trimmed at the back it is 1, and the
descriptor declares 2. The asymmetry is what decides it rather than a
preference for the larger number. Declaring 1 when the truth is 2 reads a
column holding the difference ending at i plus 1, which puts a frame recorded
after the predicted pose inside a window documented as causal, and inflates
offline accuracy by an amount that can never be reproduced online. Declaring 2
when the truth is 1 reads a difference ending at i minus 1: one frame stale,
still causal, and costing a little signal. Only one of those two errors can
flatter a reported number.

`estimate_differential_lag` measures the real value wherever the shipped
differential can be reconstructed from the shipped original, and
`verify_preprocessing` refuses a corpus build when the measurement contradicts
the declared value. On the released arrays that reconstruction is expected to
fail, because something was trimmed or filtered that the loader never sees, and
the function returns nothing rather than picking whichever alignment scored
best. So the declared 2 currently rests on the argument above and not on a
measurement of the real data. That is a stated weakness, not a settled
question; see ROADMAP.md.

## The risk the brief did not anticipate

**The pose ground truth is not in the release.**

`config.json` lists `"tasks": ["classification", "hand_landmarks"]`, but the
only ground truth files shipped are:

* `record_*.mp4`, the study video (57 MB per session, faces anonymised),
* `record_*_frame_time.txt`, one absolute Unix timestamp per video frame,
* `record_*_records.txt`, lines of `label_id,start_ts,end_ts,label_name`.

That is gesture **class** labels, not joint positions. The 21 MediaPipe
landmarks the paper regresses were produced by the authors running MediaPipe
Hands over those videos, and were not published. The README confirms the
machine learning code is not included either.

Consequences:

* Reproducing the paper's 7.87 mm figure requires regenerating the ground
  truth yourself, and your MediaPipe version will not be theirs.
* Gesture classification against these 18 classes works out of the box.
  Continuous pose regression does not.
* `SessionData.hand_poses` raises `MissingLandmarksError` with instructions
  rather than returning anything. Generate `audio{SSS}_landmarks.npy` of shape
  `(n_frames, 21, 3)` next to each profile and the loader will read it.

## Ground truth quality

* **Method:** MediaPipe Hands on a MacBook Air webcam at 30 fps, positioned
  facing the palm to minimise occlusion.
* **Convention:** 21 landmarks, MediaPipe ordering, which is what
  `wristsonar.types.JOINT_NAMES` already follows. The paper reports over the
  20 non-wrist joints.
* **Frame:** wrist-relative. The paper subtracts the wrist coordinate, rotates
  to align the palm plane with a reference plane, and scales so the
  wrist-to-pinky-MCP distance matches a measured physical length. Wrist
  rotation is deliberately left unnormalised.
* **Units:** metres after the scaling step. `HandPose` expects metres.
* **Its own error: unstated.** The release gives no bound on MediaPipe's
  accuracy and justifies the choice only as "widely adopted by prior work".

This is `GroundTruth.VIDEO_FITTED`, not `OPTICAL_MOCAP`. Every WatchHand
number is error against another estimator, and inherits that estimator's
error. `watchhand_protocol` hard-codes this and does not accept it as an
argument, so no code path can accidentally claim mocap.

## What is in the release

**40 participants, 35.6 hours of hand pose variations, about 175 GB
uncompressed.** Four studies:

| Study | Participants | Layout | Sessions | Notes |
|---|---|---|---|---|
| 1, main | 24 (sub1-sub24) | `sub{N}_{watch}_{hand}_video` | 10 per folder, 2 folders per person | 93.58 GB. One watch model per person, 8 per model |
| 2, posture | 6 returning (4,5,6,7,8,21) | `sub{N}_{watch}_{hand}_{posture}_video` | sit 2, arm_up 10, arm_down 10 | 22.65 GB |
| 3, noise | 8 | `sub{N}_video` | 18 (audio001-014, 017-020) | 28.82 GB. No IMU. Watch model and worn hand are not in the folder names |
| 4, dynamic | 8 | `sub{N}_raw/sub{N}_{speed}` | normal 20, fast/slow/free 1 each | 29.60 GB |

Study 3 session conditions: 1-10 baseline, 11-12 music, 13-14 nearby human
movement, 15-16 walking, 17-18 altered hand poses.

**Devices:** Samsung Galaxy Watch 7, Xiaomi Watch 2 Pro, Google Pixel Watch 3,
all WearOS. Study 1 participant-to-watch mapping is readable from the folder
names and is recorded in `_DEVICE_BY_PARTICIPANT`, which the loader uses as a
consistency check. It is deliberately not extended to studies 3 and 4, whose
participant numbering is independent and whose folder names omit the device.
Guessing there would silently corrupt the one split it would be used for.

**Poses, 18 classes:** 5 single-finger bends, 10 ASL digits (0-9), 3 wrist
rotations (flexion, extension, ulnar deviation).

**Per session:** original and differential `.npy` profiles, matching `.png`
previews, the video, frame timestamps, gesture records, `config.json`. Studies
1, 2 and 4 also carry `imu_config.json` and a `{PPPP}_{MMDD}_{HHMMSS}.csv` of
accelerometer, gyroscope and magnetometer. The paper did not use the IMU.

**Known gaps:** sessions 5, 6, 8, 11, 12, 13 and 14 of `sub5` in Study 3 were
removed for anonymisation, so a full-participant loop over Study 3 must not
assume 18 sessions everywhere.

## Availability and licence

**Downloadable now, no form, no login, no embargo.** Two routes:

1. **Cornell eCommons**, DOI `10.7298/qf1v-j805`, handle `1813/121443`. Twelve
   zip archives totalling about 136 GB compressed, plus the README. Direct
   HTTP, checked live.
2. **GitHub**, `witlab-kaist/WatchHand`. This is not documentation only, as the
   README implies. The tree contains the real `config.json`, `*_records.txt`,
   `*_frame_time.txt` and IMU CSVs inline, with the `.npy` and `.mp4` as Git
   LFS pointers. Expect GitHub LFS bandwidth limits at 175 GB. Prefer
   eCommons for bulk, GitHub for inspecting metadata without downloading
   anything.

**Licence, for the data specifically: CC BY 4.0.** Stated in the dataset
README and confirmed in the eCommons item metadata. Commercial use permitted
with attribution. This is the licence on the data, distinct from the ACM
copyright on the CHI 2026 paper (`10.1145/3772318.3790932`, arXiv
`2602.21610`).

**No code accompanies the release.** No training code, no preprocessing code,
no MediaPipe pipeline. The repository is a README, a hand pose figure, and the
data tree.

## Grab-n-Go, the sibling release

`cjlisalee/Grab-n-Go_Data`, DOI `10.7298/7kbd-vv75`, **CC0 1.0**, note the
different licence. Same lab, same C-FMCW technique, different hardware: a
custom wristband with two speaker-microphone pairs, 18-21 kHz and 21.5-24.5
kHz, 12 ms sweeps, 50 kHz sampling.

It does document the pipeline end to end, and it filled the two gaps
WatchHand's README leaves: that filtering is `scipy.signal.butter` and that
correlation is `np.correlate`. What it does **not** give is the filter order,
the correlation mode, or the normalisation, all of which it explicitly leaves
undocumented. The filter order came from the WatchHand paper instead
(5th-order Butterworth).

Its real contribution to this project is the arithmetic cross-check described
above: 600 bins at 50 kHz mapping to 2.06 m independently confirms the
one-bin-per-lag reading of WatchHand's 3.57 mm.

Its data is also differently shaped, per-instance 2 second clips rather than
continuous sessions, and it is gesture classification with no pose ground
truth at all. Useful as a preprocessing reference, not as a second training
set.

## Using the loader

Integrity is not optional. `WatchHandDataset` takes a `Manifest` and there is
no flag to skip it, because the failure this guards against is an evaluation
that completes successfully against the wrong bytes.

```python
from pathlib import Path
from wristsonar.data import (
    Manifest,
    Study,
    WatchHandDataset,
    build_watchhand_manifest,
    watchhand_protocol,
)
from wristsonar.protocol import Split

root = Path("/data/watchhand")

# Once, after downloading. Hashes every .npy, .json, .txt and .csv.
manifest = build_watchhand_manifest(root, version="ecommons-2026-07-07")
manifest.write(Path("manifests/watchhand.json"))

# Thereafter.
dataset = WatchHandDataset(root, Manifest.read(Path("manifests/watchhand.json")))
for ref in dataset.sessions(studies=[Study.MAIN]):
    session = dataset.load(ref)  # verifies each file it opens
    session.verify_preprocessing()  # measures what the descriptor asserts
    for start, window in session.windows():  # (2, 60, 96) float32
        ...

protocol = watchhand_protocol(Split.CROSS_USER, subjects=24, version=dataset.version)
```

`windows` takes the same `PreprocessingDescriptor` the live capture path takes
and the checkpoint records, and has no geometry defaults of its own. The
yielded integer is the index of the window's first original column, so windows
begin at `differential_lag` rather than at zero: the leading originals have no
differential aligned with them. `SessionWindowSource.iter_examples` calls
`verify_preprocessing` per session before it yields anything, so a corpus
cannot be built on top of a measurement that contradicts the contract a
checkpoint will carry. The call is per session rather than once per release
because the bin zero offset is a property of a device and a recording.

Landmark sidecars are verified against the manifest whenever they exist. There
used to be an exemption for sidecars the manifest did not mention, which was
exactly backwards: landmarks are generated after the manifest is built, so
being unpinned is their normal state, and the ground truth every reported
number is measured against was the one file that skipped its check. A sidecar
that is absent still reads as absent, and `hand_poses` raises from there. One
that is present and unaccounted for is refused, and the fix is to rebuild the
manifest now that the sidecars exist.

Two verification modes, because 175 GB. `Manifest.verify(root, deep=True)`
walks and hashes everything, and is what you run after downloading and again
before publishing a number. The loader's `LazyVerifier` hashes each file the
first time it is opened, which costs one hash per session touched and catches
the case that actually happens: the file being read right now is not the file
that was recorded. `deep=False` compares presence and size only. That is not
integrity, and the docstring says so, but it runs in seconds and therefore
runs in CI.

`build_watchhand_manifest` excludes `.png` previews, `.DS_Store` and the
`.mp4` videos. The videos are excluded reluctantly, since they are the source
of the pose ground truth, but they are 57 MB each and no training or
evaluation path reads them once landmarks exist. Pin the landmark sidecars
instead, which the manifest does.

## Tests

`tests/data/` runs with nothing downloaded. `conftest.py` generates a
miniature tree that mirrors the published layout key for key: the same
directory name grammar, the same `config.json` structure, the same records
format, the same `(600, n_frames)` float32 arrays with a differential two frames
shorter, trimmed at the front, so that the lag the loader has to recover is a
property of the fixture rather than an assumption baked into both halves of the
test. Generating rather than checking in a sample avoids redistributing participant
data, and keeps the schema written down in exactly one place, so a mismatch
with the real dataset surfaces as a failing parse instead of a test passing
against a stale copy.

The suite also asserts the physical constants directly, including that the
range resolution `c / 2B` is about 5.7 cm while a bin is 3.57 mm. Any code
that starts treating a bin as a measurement of geometry rather than as a
sample of a learned prior will trip that test.

## One bug in the published loading example

The WatchHand README's minimal loading example reads the profile filename from
`config['audio']['files'][session_idx]` and claims it yields
`"audio001_fmcw_16bit_profiles.npy"`. It does not. That list holds
`"audio001.raw"`, the original recording names, and the `.raw` files were
never released. The loader derives session indices from the filenames on disk
instead and does not read that field.
