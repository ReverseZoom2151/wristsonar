# Evaluation

This is the protocol the project commits to before it has a model, which is the
only time such a commitment is worth anything. It is written down here so that
later results can be checked against it rather than against whatever protocol
happened to flatter them.

The reasoning behind it is in PHYSICS.md. In short: range resolution is about
5.7 cm, the wavelength is about 17.6 mm, and the reported millimetre accuracies
come from regressing a coarse echo signature onto the low-dimensional manifold
of real hand poses. When the mechanism is inference over a prior rather than
measurement of geometry, performance is bounded by how well the training
distribution covers the test distribution. Under those conditions the split is
not metadata attached to a number. It is most of the number's meaning.

The code makes that structural rather than a matter of discipline. In
`wristsonar.protocol`, a `Measurement` cannot be constructed without a
`Protocol`, and a `Protocol` cannot be constructed without a split, a named and
version-pinned dataset, a ground-truth source and a subject count. There is no
code path that emits a reportable bare float.

## The four splits

Reported together, always. A result that shows one of these and not the others
is not a result, it is a selection.

| Split | What it tests | Enum |
|---|---|---|
| Within-session | Interpolation inside a single continuous wearing. The device never moved between fitting and testing. The weakest possible claim. | `Split.WITHIN_SESSION` |
| Cross-session, with remount | Whether the system survives the watch coming off and going back on. Remounting changes the occlusion geometry the signal depends on. | `Split.CROSS_SESSION` |
| Cross-user, leave-one-out | Whether it works for a person who contributed no training data. | `Split.CROSS_USER` |
| Cross-device | Whether a model trained on one watch model works on another. Transducer response in the 18 to 22 kHz band varies by tens of decibels between models, so this is a distinct failure axis and not a variant of cross-user. | `Split.CROSS_DEVICE` |

The enum ordinals encode claim strength rather than order of construction, and
`Split.is_honest` is false for exactly one member. `Measurement.comparable_to`
refuses to treat two measurements as answering the same question unless the
split, the ground-truth source, the calibration budget and the unit all match.
Dataset may differ, since cross-dataset comparison is legitimate when everything
else is held.

For reference, the published WatchHand figures across the analogous protocols
are 6.02 mm within-session, 7.87 mm cross-session with remounting, 14.88 mm
cross-user leave-one-out, and 22.60 mm on dynamic transitions. The spread
between the first and the third of those is the entire argument for reporting
all four.

## Why a within-session number is never a headline

Within-session measures how well a model interpolates between frames it
effectively already has. The wearing geometry is fixed, the user is fixed, the
hand is fixed, and the pose distribution at test time is the pose distribution
at training time. Given that the mechanism is manifold interpolation, this
measures the smoothness of an interpolator on the exact manifold it was fitted
to. It cannot fail, and a number that cannot fail is not evidence.

It is still reported, because it bounds the others from below and because its
gap to cross-session is itself informative: a large gap means the model has
learned mounting geometry, a small gap means it has learned hands. But it is
reported alongside, never alone, and never as the headline. The report layer
treats a lone within-session figure as an incomplete report rather than a
result.

## Trivial baselines ship as first-class citizens

On MPJPE, a predictor that ignores its input entirely and always emits the mean
training pose scores embarrassingly well. This is a direct consequence of the
manifold argument: if hands occupy a small region of pose space, the centre of
that region is close to everything in it. A nearest-neighbour lookup over the
training set does better still, and does it without learning anything.

Both ship in the harness rather than being mentioned in a footnote:
`MeanPoseBaseline`, `PerSubjectMeanPoseBaseline` and `NearestNeighbourBaseline`
in `wristsonar.eval.baselines`, run through the same splits, the same metrics
and the same report path as any model. The per-subject mean-pose baseline is the
sharpest of the three, because it isolates how much of a within-user result is
explained by knowing whose hand it is and nothing else.

Any model that does not clearly beat all three is reported as not clearly
beating them. That sentence is the whole reason the baselines exist.

## Calibration is a swept axis, not a setting

The field norm has settled at roughly two minutes of per-user fine-tuning.
EchoWrist reports 12.2 mm cold, 6.92 mm after one minute, plateauing at twenty.
WatchHand reports 9.96 mm zero-shot on an unseen posture and 7.65 mm after one
two-minute session. Those are large effects, and a tuned number reported without
its calibration budget overstates the system by however much the tuning bought.

So calibration minutes is a required field on `Protocol`, defaulting to zero,
with `is_zero_shot` derived from it and `at_calibration` producing the same
protocol at a different budget. The harness sweeps rather than picks:
`sweep_calibration` over `CALIBRATION_BUDGETS_MIN` produces a `CalibrationCurve`,
and the curve is what gets reported. Not its best point.

The plan specifies results at 0, 1, 2 and 20 minutes of per-user data, matching
the budgets at which the published figures above were taken, so that the curve
is comparable to the literature at every point on it. The curve is the honest
description of the system, and it is the thing the field currently buries.

## Held-out poses

Manifold interpolation is the mechanism, so the manifold's edge is where these
systems fail, and a benchmark that never leaves the training distribution never
finds out. `Protocol.held_out_poses` marks an evaluation whose test set contains
gestures absent from training, and `Protocol.describe` stamps it onto the
protocol line so it cannot be quietly dropped when tables are copied around.

The held-out gesture set is fixed in advance and never trained on. Its purpose
is not to produce a good number. Its purpose is to produce the number that says
how far outside its training distribution the system can be pushed before it
stops meaning anything, which is the question every real deployment asks
immediately.

## The shortcut-learning guard

The first independent reproduction study in knee acoustics (arXiv 2405.15085)
documented 96 percent leave-one-session-out accuracy obtained from a single
healthy subject recorded over five days, with no pathology present. The
classifier was reporting the presence of a condition that was not there. It had
learned the session, or the mounting, or the subject, and leave-one-session-out
was not a strong enough split to notice.

The lesson generalises to every body-sound result including this one: assume
contamination until subject-independent splits are shown. A high number on a
weak split is weak evidence for the system and strong evidence about the split.

The harness therefore makes the contaminated configuration inconvenient and the
honest one the default. `assert_no_leakage` raises `LeakageError` rather than
warning when subjects or sessions appear on both sides of a split.
`check_participant_concentration` flags an evaluation whose test set leans on too
few people, which is the exact shape of the knee-acoustics failure.
`check_evaluation_size` flags a figure resting on too little evidence to mean
much, which is also what `Measurement.samples` is for.
`check_label_distribution_match` flags a train and test pair whose label
distributions differ enough that the comparison is measuring the shift rather
than the model. Findings come back as a `GuardReport`, and a `Waiver` requires
naming the guard being waived, so waiving one is a recorded act rather than a
silent omission.

## Ground truth, and what a millimetre means

`GroundTruth` is a required field on every protocol, with four members, because
error against different references is not the same quantity and averaging across
them is meaningless.

| Source | What an error figure against it means |
|---|---|
| `OPTICAL_MOCAP` | Marker-based optical capture. The only option that bounds its own error, so the only one where a millimetre figure is a millimetre. |
| `DEPTH_TRACKER` | A depth-camera hand tracker such as Leap Motion. Error against a commercial estimator with its own unpublished failure modes. |
| `VIDEO_FITTED` | A skeleton or mesh fitted to monocular video. Error against another estimator, which has its own error, unbounded and correlated with pose. |
| `SYNTHETIC` | Simulated, exact by construction, and honest only about the simulation. |

Several systems in this literature, including the dataset this project starts
from, supervise against skeletons fitted to monocular video. A system reporting 6
mm against a video-fitted skeleton is claiming agreement with a monocular
estimator to 6 mm. It is not claiming that the fingertip is where it says within
6 mm. Those are different claims, and only the first is supported. Since the
video fitter's own error is itself pose-dependent and largest exactly where
occlusion is worst, which is exactly where an acoustic system is also weakest,
the two errors are not independent and the composite is not bounded by their sum
in any usable way.

This is not a reason to reject the dataset. It is a reason to stamp the
ground-truth source onto every figure and to treat acquiring genuine motion
capture as a distinct piece of work with its own value, which is where it sits
in ROADMAP.md, phase 5.

## What a complete report looks like

Every row carries its protocol line, produced by `Protocol.describe`, which
renders the split, the calibration budget or the word zero-shot, the subject
count, the ground-truth source, the dataset with its pinned version, and a
held-out-poses marker where it applies. A `Report` is a set of `ResultRow`
entries covering all four splits, the three trivial baselines, the calibration
curve, the held-out-pose evaluation, and the guard report, for every model
compared.

Metrics are MPJPE as the headline, PA-MPJPE alongside it since Procrustes
alignment removes the global rotation a wrist-mounted sensor has no business
claiming, PCK at fixed thresholds, and a per-joint breakdown with fingertips
called out separately because fingertips are where the error concentrates and
where averaging hides it.

The single-number version of this document: a number without its split is not a
result, and the harness refuses to print one.

## Sources

WatchHand: arXiv 2602.21610, doi 10.1145/3772318.3790932.
EchoWrist: arXiv 2401.17409, CHI 2024.
Knee acoustics reproduction study: arXiv 2405.15085.
