<h1 align="center">Wristsonar</h1>

<p align="center"><strong>3D Hand Pose From a Smartwatch You Already Own</strong></p>

Wristsonar turns the speaker and microphone already inside a stock smartwatch
into a hand tracker. The watch emits a quiet chirp just above the range most
people can hear, listens to how the hand reflects it, and estimates where the
fingers are. No camera, no extra hardware, no line of sight required.

The part that matters most is not the model. It is that every number this
project reports arrives with the conditions that produced it, because in this
field the conditions are most of what a number means.

## Why this exists

Acoustic hand sensing works, and it has been published repeatedly. EchoWrist
reports 4.81 mm on a custom wristband. Ring-a-Pose does 20 degrees of freedom
from a ring. WatchHand does 20 finger joints on an unmodified Galaxy Watch at
7.87 mm across sessions.

None of them released a model. There is no acoustic hand-pose system you can
download and run, anywhere. The field publishes datasets and keeps its weights.

There is also a measurement problem underneath that, and it is the more
interesting one. Range resolution for this kind of sensing is `c / 2B`. With
the roughly 3 kHz of usable bandwidth that consumer hardware allows above the
18 kHz floor, that is **5.7 cm**, which is wider than a hand. So one reporting
4.81 mm
is not resolving geometry. It is regressing a coarse echo signature onto the
low-dimensional manifold that human hands actually occupy, because tendon
coupling means twenty joints have far fewer than sixty effective degrees of
freedom.

That is a description of the method, not an accusation, and it predicts
everything the field observes: excellent accuracy on the user who trained it,
two to three times worse on someone new, near-total immunity to background
noise, and collapse on gestures it has not seen. Performance is bounded by how
well the training data covers the poses you will actually make.

Which means the split a number was measured under is not a detail about the
number. It is most of the number.

## Install

```bash
git clone https://github.com/ReverseZoom2151/wristsonar.git
cd wristsonar
pip install -e ".[dev]"
make check
```

Python 3.11 or newer. The whole test suite runs on synthetic signals, so
nothing needs downloading and no watch is required to develop against it.

## What a result looks like

```text
14.88 mm  [split=cross-user zero-shot n=24 gt=video-fitted data=watchhand@v1]
```

That string is what a `Measurement` prints. There is no code path that produces
a reportable number without the protocol attached, because `Measurement` cannot
be constructed without one and `Protocol` refuses a dataset without a version.

Four splits. A single report covers exactly one of them, because rows inside a
table have to be comparable for the comparison to mean anything, and a
cross-user model row beside a within-session baseline row would not be.
Reporting the full set is a separate type that holds one report per split and
refuses to render unless the honest splits are all present, so the weak number
cannot be published without the strong ones beside it:

| Split | What it tests |
| --- | --- |
| Within-session | Train and test from one continuous wearing. The weakest possible claim |
| Cross-session | A different wearing by the same person. Remounting changes the geometry the signal depends on |
| Cross-user | Someone who contributed no training data. Expect two to three times the error |
| Cross-device | A watch model absent from training. Response varies by tens of dB between models up at 20 kHz |

A report holding only a within-session number refuses to render as a headline
result and says why. Trivial baselines sit in the same table as the model, not
in an appendix, because a mean-pose predictor scores embarrassingly well on
MPJPE and a reader deserves to see that next to the real figure.

Calibration is reported as a curve rather than a best point. The norm in this
field has settled at roughly two minutes of per-user data, which works and is
not zero-shot, and quoting only the tuned number overstates a system by however
much the tuning bought.

## How it works

```text
smartwatch speaker  ->  18 to 21 kHz FMCW chirp, 12.5 ms, about 80 per second
smartwatch mic      ->  echoes off the hand, 10 cm away
signal layer        ->  matched filter, range-resolved echo profiles, differential
model               ->  21 joints in 3D, wrist relative
sinks               ->  JSON Lines over stdout or TCP, and Blender
```

Putting the sensor on the same arm as the hand is what makes this tractable. A
phone across the room sees an arbitrary hand at arbitrary range whose multipath
changes as it articulates, and the published numbers reflect that: 39 mm
cross-user on a stock phone against 14.88 mm on a watch. On the wrist the
sensor sits 10 cm from the joints it estimates, with a repeatable relationship
to them.

The differential profile, the frame-to-frame difference, is what the model
actually eats. It suppresses the static reflections from the watch body and the
wrist and leaves what moved.

## Where it stands

Everything from the archive through to a live pose stream is written and tested
on synthetic signals. Nothing has been trained on WatchHand yet, and no
benchmark result exists, honest or otherwise.

The piece worth naming is the preprocessing contract. A training pipeline and a
live capture path can each be correct on their own and still disagree about
what a model input is, and when they do the failure is silent: a broken capture
still emits well formed pose JSON, just wrong. So the crop origin, the window
length, the normalisation and the differential alignment live in one object
that both paths read, a checkpoint records the one it was trained under, and
loading refuses a checkpoint whose contract disagrees with the pipeline about
to feed it. A test pushes the same synthetic audio through both paths and
requires the windows to match.

The Wear OS module has never been compiled. It is written against the
documented API surface and reviewed, which is not the same thing as run.

The evaluation harness preceded training deliberately. A benchmark that arrives
after the model is a benchmark shaped by the model.

WatchHand confirms that it ships precomputed original and differential echo
profiles, not raw audio, along with the chirp and synchronization parameters.
The remaining risk is empirical: a target watch must still be shown to emit
profiles compatible with the public archive. That cannot be proved until a Wear
OS capture path is built and real hardware is available.

One consequence deserves stating before any number is published here. The
release does not include the hand landmarks the original paper regressed. Its
ground truth files carry gesture class labels, and the 21 landmark positions
were never distributed. Pose targets therefore have to be regenerated by
running MediaPipe over the released video, with a version that will not be the
one the authors used. Any pose error this project reports is measured against
labels it produced itself, so it is a number about this pipeline and is not
comparable with WatchHand's published figures. Gesture classification is
unaffected and can be trained against the shipped labels directly. See
[docs/DATA.md](docs/DATA.md).

## Live host listener

The Wear OS module in [`wear/`](wear/) emits raw signed-16-bit PCM to the host.
The host finds the chirp by correlation once and then counts samples forward,
rescoring the lock every second, because an Android capture callback is stamped
milliseconds late and a range bin is 21 microseconds wide. Timestamps are kept
for the two jobs they can actually do: spotting a dropped buffer, and
estimating the true sample rate over a long enough baseline to catch the system
resampling underneath us. The DSP that follows is the same code the model input
builder runs, and pose lands on standard output as JSON Lines:

```bash
pip install -e ".[train]"
wristsonar live listen \
  --weights path/to/model.pt \
  --bundle path/to/model.bundle.json \
  --host 0.0.0.0 --port 8766 > poses.jsonl
```

The command accepts exactly one connection. It verifies the checkpoint digest
and normalization bundle before listening. It produces no pose until the live
stream has acquired a chirp lock, established a differential reference, and
filled the causal 96-frame input window. There are no released weights yet, so
this is an executable integration path rather than a claimed live result.

## Scope

This project does active sensing from a wrist. It does not do the thing a
predecessor project claimed, which was recovering arm and finger positions from
a passive clap. That is not achievable, and the reason is physical rather than
a matter of effort: the swing is acoustically silent, the sound is generated at
contact, and hand configuration is independent of arm configuration given the
contact state. A single impulse carries a couple of bits about the hand where
an arm pose needs roughly thirty.

What a clap genuinely does carry is the shape the hands were in when they met,
via the resonance of the air cavity between them. That is implemented, bounded,
and documented in [docs/CLAP.md](docs/CLAP.md), including a module that
computes how many bits a given classifier actually recovers so the component
can state its own limit rather than assert it.

## Documentation

- [Physics](docs/PHYSICS.md), why this works and why the passive version does not
- [Evaluation](docs/EVALUATION.md), the protocol committed to in advance
- [Architecture](docs/ARCHITECTURE.md)
- [Prior art](docs/PRIOR_ART.md), the published systems and what they released
- [Data](docs/DATA.md)
- [Wear OS sender](wear/README.md), including the hardware validation protocol
- [Roadmap](docs/ROADMAP.md)

## Contributing

Run `make check` before submitting. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the rules that matter here, chiefly that no result is reported without its
protocol and that the test suite must always pass with no dataset present.

## License

[MIT](LICENSE).
