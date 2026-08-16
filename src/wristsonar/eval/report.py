"""Rendering results so that a number cannot travel without its protocol.

The failure this module is built against is social rather than technical. A
number gets computed under a careful protocol, lands in a table, gets copied
into a slide, and arrives at a reader as four point eight one millimetres with
no split, no calibration budget and no ground truth source. By then it is
indistinguishable from a claim about sensing resolution, which it never was.

So every row printed here carries Protocol.describe() on it, the baselines sit
in the same table as the model rather than in an appendix nobody reaches, the
calibration curve is printed in full rather than at its best point, and every
rendering ends with a statement of the conditions the result holds under. The
closing statement is not a disclaimer. It is the shortest correct form of the
result, and it is printed last so it survives a screenshot of the bottom of
the table as well as the top.

A report that holds only a within session measurement will not render a
headline. It renders a refusal instead, naming what is missing. Within session
is the weakest protocol in the field, RAM-Hand's 10.71 mm is an 80/20 split
inside single sessions, and a harness that lets that shape of number be a
headline is the harness doing the damage.

One Report holds one split, because rows inside a table have to be comparable
for the comparison to mean anything and a cross user model row beside a within
session baseline row is not. Reporting the full set is therefore a second
container, MultiSplitReport, which holds one Report per split, renders them
together, and refuses a headline while any of the honest splits is missing. A
selection of splits is not a result, and choosing which ones to show is the
easiest way to produce a wrong headline out of entirely correct numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from wristsonar.eval.calibration import CalibrationCurve
from wristsonar.eval.guard import GuardReport
from wristsonar.eval.metrics import MetricSet
from wristsonar.protocol import Measurement, Protocol, Split

__all__ = [
    "CLEAR_MARGIN",
    "MultiSplitReport",
    "Report",
    "ResultRow",
]

CLEAR_MARGIN = 0.10
"""How much better than a baseline counts as clearly better.

Ten percent relative on MPJPE. Chosen rather than derived, and stated here
because a threshold that lives only in a comparison operator is a threshold
nobody can argue with. Ten percent is roughly the gap between EchoWrist's
cross-user 12.2 mm and a plausible per-subject-mean baseline on the same data,
which is to say: a model landing inside this margin has not demonstrated that
it is reading the echo rather than the population prior.
"""

WITHIN_SESSION_ALONE = "every measurement here is WITHIN_SESSION"
"""Opening of the refusal a lone within-session report renders.

Named because it is the one refusal a multi-split container can cure. Holding
a within-session report beside the honest splits is exactly what the refusal
asks for, so MultiSplitReport drops this reason and keeps every other one.
"""


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One predictor's numbers under one protocol."""

    name: str
    kind: str
    """Either model or baseline. Both print in the same table."""

    metrics: MetricSet

    @property
    def mpjpe(self) -> Measurement:
        return self.metrics.mpjpe

    @property
    def is_baseline(self) -> bool:
        return self.kind == "baseline"


@dataclass
class Report:
    """Measurements across splits, plus the machinery that refuses to lie.

    Rows are checked for mutual comparability as they are added, using
    Measurement.comparable_to, so a table cannot silently mix a zero shot
    cross user number with a two minute within session one.
    """

    title: str
    rows: list[ResultRow] = field(default_factory=list)
    calibration: CalibrationCurve | None = None
    guards: GuardReport | None = None
    notes: str = ""

    def _check_comparable(self, metrics: MetricSet) -> None:
        if not self.rows:
            return
        reference = self.rows[0].metrics
        if not reference.comparable_to(metrics):
            raise ValueError(
                "refusing to place non-comparable measurements in one table.\n"
                f"  existing: {reference.mpjpe.protocol.describe()} "
                f"alignment={reference.alignment.value} "
                f"pck@{reference.pck_threshold_m:g}m\n"
                f"  incoming: {metrics.mpjpe.protocol.describe()} "
                f"alignment={metrics.alignment.value} "
                f"pck@{metrics.pck_threshold_m:g}m\n"
                "Split, ground truth, calibration budget, alignment and PCK "
                "threshold must match for a row-to-row comparison to mean "
                "anything. A Procrustes-aligned row in a root-aligned column "
                "is the flattering version of this mistake. Build a second "
                "Report if the question is different."
            )

    def add_model(self, name: str, metrics: MetricSet) -> None:
        self._check_comparable(metrics)
        self.rows.append(ResultRow(name=name, kind="model", metrics=metrics))

    def add_baseline(self, name: str, metrics: MetricSet) -> None:
        self._check_comparable(metrics)
        self.rows.append(ResultRow(name=name, kind="baseline", metrics=metrics))

    def add_baselines(self, results: dict[str, MetricSet]) -> None:
        for name, metrics in results.items():
            self.add_baseline(name, metrics)

    def set_calibration(self, curve: CalibrationCurve) -> None:
        self.calibration = curve

    def set_guards(self, guards: GuardReport) -> None:
        """Attach guards, refusing any that were computed somewhere else.

        A GuardReport carries a binding naming the dataset, the version and the
        split it was run on. Checking it here is what stops guards from an
        easier split being attached to a harder result, which is otherwise
        invisible: the rendered report looks identical either way.
        """
        mismatch = self._guard_mismatch(guards)
        if mismatch is not None:
            raise ValueError(
                f"refusing guards that were not computed on these rows: "
                f"{mismatch}.\n  {guards.binding.describe()}"
            )
        self.guards = guards

    def _guard_mismatch(self, guards: GuardReport | None = None) -> str | None:
        report = self.guards if guards is None else guards
        if report is None or not self.rows:
            return None
        return report.binding.disagreement_with(self.rows[0].metrics.protocol)

    @property
    def models(self) -> tuple[ResultRow, ...]:
        return tuple(r for r in self.rows if not r.is_baseline)

    @property
    def baselines(self) -> tuple[ResultRow, ...]:
        return tuple(r for r in self.rows if r.is_baseline)

    @property
    def splits(self) -> tuple[Split, ...]:
        return tuple({r.metrics.protocol.split for r in self.rows})

    @property
    def split(self) -> Split | None:
        """The one split this report covers, or None while it is empty.

        Rows are checked for comparability as they are added and comparability
        requires an identical split, so there is never more than one.
        """
        if not self.rows:
            return None
        return self.rows[0].metrics.protocol.split

    @property
    def protocol(self) -> Protocol | None:
        if not self.rows:
            return None
        return self.rows[0].metrics.protocol

    @property
    def best_model(self) -> ResultRow | None:
        models = self.models
        if not models:
            return None
        return min(models, key=lambda r: r.mpjpe.value)

    def beaten_baselines(self) -> dict[str, bool]:
        """Per baseline, whether the best model clearly beats it.

        Clearly means by more than CLEAR_MARGIN relative on MPJPE. Anything
        inside that margin counts as not beaten, and the report says so in the
        verdict line rather than leaving the reader to compare two numbers
        that differ in the second decimal.
        """
        model = self.best_model
        if model is None:
            return {b.name: False for b in self.baselines}
        return {
            b.name: model.mpjpe.value < b.mpjpe.value * (1.0 - CLEAR_MARGIN)
            for b in self.baselines
        }

    @property
    def refusal_reasons(self) -> tuple[str, ...]:
        """Why this report cannot stand as a headline result. Empty means it can."""
        reasons: list[str] = []
        if not self.rows:
            reasons.append("the report contains no measurements")
        splits = set(self.splits)
        if splits and splits == {Split.WITHIN_SESSION}:
            reasons.append(
                f"{WITHIN_SESSION_ALONE}. Train and test came "
                "from one continuous wearing, so the device never moved "
                "between fitting and testing and the number measures "
                "interpolation inside one mounting geometry. It is the "
                "weakest protocol in the field. Add at least a cross-session "
                "measurement before quoting anything."
            )
        if self.rows and not self.models:
            reasons.append(
                "no model was evaluated, so this table is baselines only. "
                "There is nothing here to headline: the trivial predictors are "
                "the comparison, not the result."
            )
        if self.rows and not self.baselines:
            reasons.append(
                "no baselines were evaluated. A model result without the mean "
                "pose, per-subject mean pose and nearest neighbour numbers "
                "beside it is not interpretable, because those three land "
                "within millimetres of published systems on this task."
            )
        if self.guards is None:
            reasons.append(
                "the shortcut-learning guards were not run. Call "
                "guard.run_guards on the evaluation set and attach the result."
            )
        elif self.guards.blocking:
            names = ", ".join(f.guard.value for f in self.guards.blocking)
            reasons.append(f"guards are firing and have not been waived: {names}")
        mismatch = self._guard_mismatch()
        if mismatch is not None:
            reasons.append(
                f"the attached guard report does not describe these rows: "
                f"{mismatch}"
            )
        return tuple(reasons)

    def refusal_reasons_in_a_set(self) -> tuple[str, ...]:
        """Refusals that holding the other splits alongside cannot answer.

        Exactly one refusal is curable that way: a lone within-session report
        is refused because it needs a stronger split beside it, and a
        MultiSplitReport that holds every honest split has supplied one.
        Everything else, missing baselines and firing guards included, is
        still a refusal in any company.
        """
        return tuple(
            reason
            for reason in self.refusal_reasons
            if not reason.startswith(WITHIN_SESSION_ALONE)
        )

    @property
    def can_headline(self) -> bool:
        return not self.refusal_reasons

    def headline(self) -> str:
        """The one line result, or a refusal explaining what is missing.

        The absent model case is a refusal reason rather than an assertion.
        An assert here vanishes under python -O, which would turn a
        baselines-only report into an AttributeError at best and a headline
        about a baseline at worst.
        """
        reasons = self.refusal_reasons
        if reasons:
            body = "\n".join(f"  - {r}" for r in reasons)
            return "NO HEADLINE RESULT. This report cannot be quoted:\n" + body
        model = self.best_model
        if model is None:
            return (
                "NO HEADLINE RESULT. This report cannot be quoted:\n"
                "  - no model was evaluated, so the table is baselines only"
            )
        beaten = self.beaten_baselines()
        qualifier = (
            ""
            if all(beaten.values())
            else "  (does NOT clearly beat: "
            + ", ".join(n for n, ok in beaten.items() if not ok)
            + ")"
        )
        return (
            f"{model.name}: {model.mpjpe.value:.2f} mm MPJPE "
            f"[{model.mpjpe.protocol.describe()}]{qualifier}"
        )

    def verdict(self) -> str:
        """Whether the model did anything the trivial predictors did not."""
        model = self.best_model
        if model is None:
            return "no model evaluated; the table is baselines only"
        beaten = self.beaten_baselines()
        if not beaten:
            return "no baselines evaluated, so the model result is uninterpretable"
        failed = [n for n, ok in beaten.items() if not ok]
        if not failed:
            return (
                f"{model.name} clearly beats all {len(beaten)} trivial "
                f"baselines by more than {CLEAR_MARGIN:.0%} relative MPJPE"
            )
        return (
            f"{model.name} does NOT clearly beat {', '.join(failed)}. "
            f"A margin under {CLEAR_MARGIN:.0%} on MPJPE against a predictor "
            f"that never reads the microphone is not evidence of sensing."
        )

    def validity_statement(self) -> str:
        """The conditions the result holds under, printed last, every time."""
        if not self.rows:
            return "no measurements, so there is nothing to qualify"
        protocol = self.rows[0].metrics.protocol
        alignment = self.rows[0].metrics.alignment
        strongest = max(self.splits, key=lambda s: s.value)
        lines = [
            "VALID ONLY UNDER THESE CONDITIONS",
            f"  protocol            {protocol.describe()}",
            f"  strongest split     {strongest.name}",
            f"  alignment           {alignment.describe()}",
            "  joint convention    both printed. 21 joints includes the wrist,",
            "                      which root alignment forces to zero and which",
            "                      therefore understates the error by 4.76",
            "                      percent against the 20-joint convention.",
            f"  ground truth        {protocol.ground_truth.value}",
            f"  participants        {protocol.subjects}",
            f"  calibration         {protocol.calibration_minutes:g} min per user",
            f"  held-out poses      {protocol.held_out_poses}",
            "  sensing resolution  about 5.7 cm range resolution at 3 kHz of",
            "                      usable bandwidth. Every millimetre figure",
            "                      here is a regression onto the hand pose",
            "                      manifold, not a distance measurement.",
        ]
        if strongest is Split.WITHIN_SESSION:
            lines.append(
                "  WARNING             within-session only. Not a claim about "
                "any real deployment."
            )
        if strongest.value < Split.CROSS_USER.value:
            lines.append(
                "  UNMEASURED          cross-user performance. Expect two to "
                "three times this error on a person who contributed no "
                "training data."
            )
        if strongest.value < Split.CROSS_DEVICE.value:
            lines.append(
                "  UNMEASURED          cross-device performance. Speaker and "
                "microphone response varies by tens of dB between models in "
                "the 18 to 22 kHz band."
            )
        if not protocol.held_out_poses:
            lines.append(
                "  UNMEASURED          poses outside the training "
                "distribution, which is where manifold regression fails."
            )
        lines.append(
            "  This number is meaningless detached from the lines above. If it "
            "appears without them, it is being misquoted."
        )
        return "\n".join(lines)

    def render_text(self) -> str:
        """A plain text report for humans. Everything, in one place."""
        width = 78
        out: list[str] = [
            "=" * width,
            self.title,
            "=" * width,
            "",
            self.headline(),
            "",
        ]
        if self.rows:
            out.append("RESULTS  (baselines are rows in this table, not an appendix)")
            out.append(
                f"  {'predictor':<20}{'kind':<8}{'MPJPE21':>9}{'MPJPE20':>9}"
                f"{'PA':>9}{'tips':>9}{'PCK':>8}"
            )
            out.append("  " + "-" * (width - 4))
            for row in sorted(self.rows, key=lambda r: r.mpjpe.value):
                m = row.metrics
                out.append(
                    f"  {row.name:<20}{row.kind:<8}"
                    f"{m.mpjpe.value:9.2f}{m.mpjpe_excl_root.value:9.2f}"
                    f"{m.pa_mpjpe.value:9.2f}"
                    f"{m.breakdown.fingertip_mm:9.2f}{m.pck.value:8.3f}"
                )
                out.append(f"      protocol: {m.mpjpe.protocol.describe()}")
            out.append("")
            out.append(
                "  MPJPE21 averages all 21 joints, which is what most of this "
                "field prints;"
            )
            out.append(
                "  MPJPE20 drops the wrist, which root alignment forces to "
                "exactly zero, and"
            )
            out.append(
                "  is the convention the published figures here are compared "
                "against. The"
            )
            out.append(
                "  gap between the two columns is 1/21, or 4.76 percent, which "
                "is half of"
            )
            out.append(
                f"  the {CLEAR_MARGIN:.0%} margin used to decide whether a "
                "baseline was beaten."
            )
            out.append(
                "  PA is Procrustes aligned, tips is the mean over the five "
                "fingertips, and"
            )
            out.append(
                "  PCK is the fraction of joints inside "
                f"{self.rows[0].metrics.pck_threshold_m * 100:g} cm."
            )
            out.append("")
            out.append(f"VERDICT  {self.verdict()}")
            out.append("")

        best = self.best_model or (self.rows[0] if self.rows else None)
        if best is not None:
            out.append(f"PER-JOINT BREAKDOWN  ({best.name})")
            out.append(best.metrics.breakdown.render())
            out.append("")

        if self.calibration is not None:
            out.append(self.calibration.render())
            out.append("")
        else:
            out.append(
                "CALIBRATION  not swept. The field norm is roughly two minutes "
                "of per-user\n  data and results are usually quoted at the "
                "tuned end without the curve."
            )
            out.append("")

        out.append("SHORTCUT-LEARNING GUARDS")
        if self.guards is None:
            out.append("  NOT RUN. No result here can be trusted until they are.")
        else:
            out.append(self.guards.render())
        out.append("")

        if self.notes:
            out.append("NOTES")
            out.append("  " + self.notes)
            out.append("")

        out.append(self.validity_statement())
        out.append("=" * width)
        return "\n".join(out)

    def to_dict(self) -> dict[str, object]:
        """Machine readable, and carrying every refusal the text version does."""
        return {
            "title": self.title,
            "can_headline": self.can_headline,
            "headline": self.headline(),
            "refusal_reasons": list(self.refusal_reasons),
            "verdict": self.verdict(),
            "beaten_baselines": self.beaten_baselines(),
            "rows": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "mpjpe_mm": r.metrics.mpjpe.value,
                    "mpjpe_excl_root_mm": r.metrics.mpjpe_excl_root.value,
                    "pa_mpjpe_mm": r.metrics.pa_mpjpe.value,
                    "fingertip_mm": r.metrics.breakdown.fingertip_mm,
                    "non_fingertip_mm": r.metrics.breakdown.non_fingertip_mm,
                    "pck": r.metrics.pck.value,
                    "pck_excl_root": r.metrics.pck_excl_root.value,
                    "pck_threshold_m": r.metrics.pck_threshold_m,
                    "alignment": r.metrics.alignment.value,
                    "n_frames": r.metrics.n_frames,
                    "protocol": r.metrics.mpjpe.protocol.describe(),
                    "split": r.metrics.protocol.split.name,
                    "per_joint_mm": r.metrics.breakdown.per_joint_mm,
                }
                for r in self.rows
            ],
            "calibration_curve": (
                None if self.calibration is None else self.calibration.to_json()
            ),
            "guards": None if self.guards is None else self.guards.to_json(),
            "guard_binding": (
                None if self.guards is None else self.guards.binding.describe()
            ),
            "split": None if self.split is None else self.split.name,
            "validity_statement": self.validity_statement(),
            "notes": self.notes,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


REQUIRED_SPLITS: tuple[Split, ...] = tuple(s for s in Split if s.is_honest)
"""The splits a multi-split report must hold before it may headline.

Every split for which Split.is_honest is true: cross-session, cross-user and
cross-device. Within-session is rendered when present and is never required,
because it cannot fail and a report is not improved by including a number that
cannot fail. The other three are required together because showing one of them
and not the others is a selection, and a selection made after seeing the
numbers is the cheapest wrong headline available.
"""


@dataclass
class MultiSplitReport:
    """One Report per split, rendered together or not at all.

    A Report covers exactly one split, because rows inside a table must be
    comparable and comparability requires an identical split. That constraint
    is right for a table and useless as a way to report a system, since the
    thing a reader needs is the spread across splits: WatchHand goes 6.02 mm
    within session, 7.87 mm cross session and 14.88 mm cross user, and only the
    third of those describes a stranger's experience. This container holds the
    set, renders the splits side by side with the strongest claim last, and
    refuses a headline while any honest split is missing.
    """

    title: str
    reports: dict[Split, Report] = field(default_factory=dict)
    notes: str = ""

    def add(self, report: Report) -> None:
        """File a report under the split its rows were measured on.

        The split is read off the rows rather than passed in, so a report
        cannot be filed under a stronger split than it was computed with.
        """
        split = report.split
        if split is None:
            raise ValueError(
                f"report {report.title!r} holds no measurements, so there is no "
                f"split to file it under"
            )
        if split in self.reports:
            raise ValueError(
                f"a {split.name} report is already present ("
                f"{self.reports[split].title!r}). Two reports for one split "
                f"means one of them is being chosen after the fact."
            )
        self.reports[split] = report

    @property
    def ordered(self) -> tuple[Report, ...]:
        """Present reports, weakest claim first, so the strongest reads last."""
        order = sorted(self.reports, key=lambda s: s.value)
        return tuple(self.reports[s] for s in order)

    @property
    def missing_splits(self) -> tuple[Split, ...]:
        return tuple(s for s in REQUIRED_SPLITS if s not in self.reports)

    def model_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for report in self.reports.values():
            names.update(row.name for row in report.models)
        return tuple(sorted(names))

    @property
    def refusal_reasons(self) -> tuple[str, ...]:
        """Why this set cannot stand as a headline result."""
        reasons: list[str] = []
        if not self.reports:
            reasons.append("no reports have been added")
        missing = self.missing_splits
        if missing:
            reasons.append(
                "the honest splits are not all present. Missing: "
                + ", ".join(s.name for s in missing)
                + ". A result that shows some splits and not others is a "
                "selection, and the missing ones are the expensive ones to run "
                "and the ones a reader most needs."
            )
        for report in self.ordered:
            label = report.split.name if report.split else "?"
            for reason in report.refusal_reasons_in_a_set():
                reasons.append(f"[{label}] {reason}")
        for split, report in sorted(self.reports.items(), key=lambda kv: kv[0].value):
            missing_models = set(self.model_names()) - {r.name for r in report.models}
            if missing_models:
                reasons.append(
                    f"[{split.name}] does not evaluate "
                    f"{sorted(missing_models)}, which other splits do. A system "
                    f"measured on the easy splits and a different one measured "
                    f"on the hard splits is not one result."
                )
        return tuple(reasons)

    @property
    def can_headline(self) -> bool:
        return not self.refusal_reasons

    def headline(self) -> str:
        reasons = self.refusal_reasons
        if reasons:
            body = "\n".join(f"  - {r}" for r in reasons)
            return (
                "NO HEADLINE RESULT. This set of reports cannot be quoted:\n" + body
            )
        strongest = self.reports[max(self.reports, key=lambda s: s.value)]
        return strongest.headline()

    def comparison_table(self) -> str:
        """Every split's model number in one column, which is the whole point."""
        lines = [
            f"  {'split':<16}{'model':<20}{'MPJPE21':>9}{'MPJPE20':>9}"
            f"{'best baseline':>16}",
            "  " + "-" * 74,
        ]
        for report in self.ordered:
            split = report.split
            best = report.best_model
            baselines = report.baselines
            baseline_value = (
                min(b.mpjpe.value for b in baselines) if baselines else float("nan")
            )
            name = "none" if best is None else best.name
            model_21 = float("nan") if best is None else best.mpjpe.value
            model_20 = (
                float("nan") if best is None else best.metrics.mpjpe_excl_root.value
            )
            lines.append(
                f"  {(split.name if split else '?'):<16}{name:<20}"
                f"{model_21:9.2f}{model_20:9.2f}{baseline_value:16.2f}"
            )
        for split in self.missing_splits:
            lines.append(f"  {split.name:<16}{'NOT MEASURED':<20}")
        return "\n".join(lines)

    def render_text(self) -> str:
        """Every split, weakest first, with the refusal at the top and bottom."""
        width = 78
        out: list[str] = [
            "#" * width,
            self.title,
            "#" * width,
            "",
            self.headline(),
            "",
            "SPLITS  (one report each, strongest claim last)",
            self.comparison_table(),
            "",
        ]
        if self.notes:
            out.extend(["NOTES", "  " + self.notes, ""])
        for report in self.ordered:
            out.append(report.render_text())
            out.append("")
        reasons = self.refusal_reasons
        if reasons:
            out.append("WHY THIS SET IS NOT A RESULT")
            out.extend(f"  - {reason}" for reason in reasons)
        else:
            out.append(
                "Every honest split is present and every one of them cleared "
                "its own refusal checks."
            )
        out.append("#" * width)
        return "\n".join(out)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "can_headline": self.can_headline,
            "headline": self.headline(),
            "refusal_reasons": list(self.refusal_reasons),
            "missing_splits": [s.name for s in self.missing_splits],
            "models": list(self.model_names()),
            "reports": {
                report.split.name if report.split else "?": report.to_dict()
                for report in self.ordered
            },
            "notes": self.notes,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)
