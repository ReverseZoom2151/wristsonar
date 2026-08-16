# Contributing to wristsonar

Thanks for taking an interest. This file covers how to get the checks running
locally and the few rules that are specific to this project. The rules about
reporting results are the ones that matter most, so please read those even if
you skim the rest.

## Setting up

Python 3.12 is the development version and the repository pins it in
`.python-version`. CI also runs 3.11, so keep the code working on both.

```sh
python -m venv .venv
. .venv/bin/activate        # Windows: .venv/Scripts/activate
make install
```

`make install` installs the package in editable mode with its `dev` extra,
which brings in pytest, hypothesis, ruff and mypy.

## The four checks

Every change has to pass the same four things CI runs.

```sh
make lint        # ruff check .
make typecheck   # mypy
make test        # pytest
make check       # all three, in that order
```

`make format` runs `ruff format` and then `ruff check --fix`, which fixes most
lint failures for you. mypy runs in strict mode over `src` and `tests`, so new
code needs annotations.

If you do not have `make`, the underlying commands are `ruff check .`, `mypy`,
and `pytest`.

## No result without its protocol

A bare number is not a result in this project. Millimetre figures from acoustic
hand pose depend almost entirely on how they were measured, so a value reported
without its conditions is not a weaker claim, it is an unreadable one.

Everything you need is in `src/wristsonar/protocol.py`. A `Measurement` cannot
be constructed without a `Protocol`, and the `Protocol` carries the split, the
dataset and its pinned version, the ground truth source, the subject count, the
calibration budget, and whether the test set contained held-out poses. That is
deliberate. If you find yourself wanting to pass a float around instead, that is
a sign the protocol needs to be threaded further through the call chain, not
that the protocol should be dropped.

Concretely:

- Any function that returns an accuracy, error or score returns a `Measurement`,
  not a float.
- Any table, log line, README claim or plot label carries `protocol.describe()`
  alongside the number.
- Comparisons between measurements go through `Measurement.comparable_to`.
  Two numbers taken under different splits, ground truths or calibration
  budgets answer different questions and must not be put in the same column.
- Ground truth fitted to monocular video is error against another estimator,
  not error against the world. Say so where the number appears.

## Within-session numbers are never a headline

`Split.WITHIN_SESSION` means the model was fitted and tested inside one
continuous wearing of the device. Nothing moved, so it measures interpolation
within a single mounting geometry and says very little about a real system.

Report within-session figures only as supporting detail, never as the headline
number, never in an abstract or a README summary, and never without a
cross-session or cross-user figure next to them. `Measurement.is_honest` and
`Split.is_honest` exist so this can be checked in code rather than remembered.
If the honest number is much worse than the within-session one, that gap is
information, so publish both rather than the flattering one alone.

## The suite runs with no dataset

`pytest` must pass on a clean checkout with nothing downloaded, no hardware
attached, and no network. This keeps CI honest and keeps a new contributor
about sixty seconds away from a green suite.

That means tests use synthetic fixtures: generated chirps, simulated echoes,
and hand-built pose arrays. Write them so the expected answer follows from how
the signal was constructed rather than from a recorded file. Hypothesis is
available and is a good fit for the signal processing layers.

If a test genuinely needs recorded data, mark it so that it skips cleanly when
the data is absent, and make sure the default `pytest` invocation stays green
without it. A test that fails because a dataset is missing is a broken test.

## Commit messages

Write the subject line in the imperative mood, as an instruction to the
codebase, under about seventy characters, with no trailing full stop.

```
Thread the protocol through the report layer
```

Not `Threaded the protocol...`, not `Fixes report layer`, not `updates`.

Use the body to explain why the change exists. The diff already shows what
changed, so a body that restates it is wasted. What the diff cannot show is the
constraint you were working against, the option you rejected, or the surprising
behaviour that made the change necessary. Wrap the body at about seventy-two
characters and separate it from the subject with a blank line.

```
Refuse to average measurements across ground truth sources

Optical mocap error and video-fitted error are not the same quantity,
so the mean of the two was a number with no interpretation. Callers
that want a single figure now have to pick a source explicitly.
```

## Pull requests

Keep the pull request focused on one thing, make sure `make check` passes, and
fill in the template. If the change touches evaluation in any way, the template
asks which splits you ran, and that field is not optional.
