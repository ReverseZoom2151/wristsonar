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
the roughly 3 kHz of usable inaudible bandwidth that consumer hardware allows,
that is **5.7 cm**, which is wider than a hand. So a system reporting 4.81 mm
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
14.88 mm  [split=cross-user zero-shot n=40 gt=depth-tracker data=watchhand@v1]
```

That string is what a `Measurement` prints. There is no code path that produces
a reportable number without the protocol attached, because `Measurement` cannot
be constructed without one and `Protocol` refuses a dataset without a version.

Four splits, and a report shows them together or not at all:

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
sinks               ->  OpenXR, Blender, WebSocket
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

The offline and host-side substrate is built: data integrity, signal processing,
landmark preparation, causal model inputs, guarded aggregate evaluation,
training, checkpoint provenance/loading, capture health checks, realtime
inference and a Blender sink. Nothing has been trained on WatchHand yet, and no
benchmark result exists, honest or otherwise.

The evaluation harness preceded training deliberately. A benchmark that arrives
after the model is a benchmark shaped by the model.

WatchHand confirms that it ships precomputed original and differential echo
profiles, not raw audio, along with the chirp and synchronization parameters.
The remaining risk is empirical: a target watch must still be shown to emit
profiles compatible with the public archive. That cannot be proved until a Wear
OS capture path and real hardware are available.

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
- [Roadmap](docs/ROADMAP.md)

## Contributing

Run `make check` before submitting. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the rules that matter here, chiefly that no result is reported without its
protocol and that the test suite must always pass with no dataset present.

## License

[MIT](LICENSE).
