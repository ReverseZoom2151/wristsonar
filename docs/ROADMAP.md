# Roadmap

Six phases, each with a deliverable and an exit criterion that can be checked
rather than argued about. The ordering is itself a claim, and the important part
of it is that phase 1 precedes phase 2.

## Current status

The offline signal layer is complete and synthetic-verified. The software for
manifest-gated data ingest, landmark preparation, split construction, guarded
evaluation, Torch training, checkpoint export/loading and host-side Blender
inference is also complete. No public WatchHand archive has been downloaded,
no landmark sidecars have been generated, and no trained weights or benchmark
result exists. Those are external-data milestones, not wording changes this
repository can make true on its own.

## Phase 0. Signal layer, offline

**Status: implemented and synthetic-verified.**

Chirp synthesis, matched filtering, echo profiles, differential profiles.
Property-based tests against synthetic echoes with known delay and amplitude. No
device, no dataset, no model.

Exits when a synthetic reflector placed at a known range is recovered to within
one range bin.

This is first because it is the layer where a reproducibility bug hides for
months. A sign error in a delay or an off-by-one in a lag mapping does not
crash; it produces plausible-looking numbers that stay plausible until something
downstream is blamed for them. Pinning it against a forward model, before any
real data can supply an excuse, is the only cheap moment to catch that.

## Phase 1. Data and evaluation harness

**Status: software implemented; empirical exit pending the public archive.**

Ingest the WatchHand dataset. Build the four-split harness, the trivial
baselines, and the calibration-curve reporter.

Exits when a mean-pose baseline produces a full honest report: all four splits,
the calibration curve, the held-out-pose evaluation, and the guard report, each
row carrying its protocol.

This deliberately precedes any modelling. A benchmark that arrives after the
model is a benchmark shaped by the model, whatever the intentions of whoever
builds it. Every choice about which split to feature, which metric to lead with,
which guard to relax, becomes a choice about which number to show, once there is
a number to show. Making those choices while the only thing being measured is a
predictor that ignores its input costs nothing and buys the credibility of every
figure that comes later.

It is also the phase that makes the worst outcome of phase 2 survivable. See the
risks below.

## Phase 2. Reproduce

**Status: training, checkpoint and aggregate-evaluation software implemented;
weights and results pending the public archive and generated sidecars.**

Train the published architecture on the public data. Land inside the published
error bars on all four splits, and say plainly where that does not happen.

Exits with released weights, which is the thing that does not currently exist
anywhere in this field. See PRIOR_ART.md: eight published systems, zero runnable
artifacts.

## Phase 3. Capture app

**Status: host capture contract, callback synchronization, and versioned raw
PCM transport implemented; Wear OS sender and hardware validation pending.**

Wear OS duplex capture. This is where platform reality bites: sample-rate lies,
automatic gain control, buffer underruns, and the fact that two hours of
continuous sensing drains 78 percent of a Galaxy Watch 3 battery.

Exits with a live echo profile from a real wrist.

## Phase 4. Close the loop

**Status: host inference, TCP/JSONL and Blender sink implemented; live model,
OpenXR adapter and user calibration flow pending.**

Live inference, OpenXR and Blender sinks, and the two-minute calibration flow as
a real user-facing thing rather than a training-time detail.

Exits with a hand moving on screen, driven by a watch nobody modified.

## Phase 5. Extend

**Status: the bounded clap-configuration side project is implemented offline;
cross-device transfer, calibration reduction and new ground-truth data remain
future empirical work.**

Only now, and in rough order of value: cross-device transfer, reducing the
calibration budget, a second dataset with genuine motion-capture ground truth
rather than video-derived labels, and arm-level joints if the data turns out to
support them.

The plan also identifies an honest, small, genuinely novel side quest here:
clap-configuration classification at four to five acoustically defensible
classes, following the confusion structure in Jylha and Erkut, which nobody has
done at scale or cross-subject. It is a side quest, not the flagship, and it is
pose estimation of nothing. It is described in CLAP.md, and PHYSICS.md explains
why the larger version of that idea is impossible.

## Risks

### The dataset may not be enough

This one can end the project, so it is first, and the plan says to check it in
week one before anything else is built.

The WatchHand release is echo profiles, not raw audio. If the released form is
post-processed in a way that cannot be reproduced from a live device, then the
model trained in phase 2 and the capture stack built in phase 3 will never meet.
There would be a model that works on the dataset and a device that produces
something else, with no bridge between them, and no amount of later engineering
recovers that. Everything downstream of phase 1 is conditional on this check.

### Wear OS duplex audio may not behave

Playing and recording simultaneously at 48 kHz with predictable timing on a
watch is not a solved consumer use case. If it fails, the project degrades to
phone-based, and the research says phone-based is much worse: SonicHand on a
stock Pixel 2 gets 23.25 mm within-user and 39.28 mm cross-user, which its own
authors call unsatisfactory.

### Battery makes it a demo, not a daily driver

Two hours of continuous sensing takes 78 percent of a Galaxy Watch 3. Duty
cycling is a research problem in its own right and is not in scope for phases 0
through 4.

### Audible leakage

Across 30 commodity devices transmitting 18 to 22 kHz at 80 percent volume,
audible low-frequency leakage from amplifier nonlinearity was present on most
and exceeded 65 dB on some. Chirps leak least, which is fortunate. It still has
to be measured on the target hardware rather than assumed, and it belongs in
phase 3.

### The published numbers may not reproduce

This is a real possibility. It is also a legitimate and valuable outcome, and it
is the one outcome whose value depends entirely on the ordering above: a failed
reproduction is a contribution if the harness was built first and is credible,
and is indistinguishable from incompetence if it was not.

### Cross-user is the number that decides whether this is useful

WatchHand reports 14.88 mm cross-user, leave-one-out. That is good. If this
reproduction lands at 25 mm, the honest report says 25 mm, and the project is
what it is.

## Decisions and remaining external prerequisites

The project uses public WatchHand data first, and retains the physically bounded
clap-configuration component as a side project. The remaining prerequisites are
not open product choices: obtain the approximately 175 GB archive with enough
storage, generate and manifest landmark sidecars, train and publish a model,
then validate a Wear OS capture app on one of the watches used in WatchHand.
Target-hardware selection remains open between Galaxy Watch 7, Xiaomi Watch 2
Pro and Pixel Watch 3.
