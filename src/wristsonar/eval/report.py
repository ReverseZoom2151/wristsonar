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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from wristsonar.eval.calibration import CalibrationCurve
from wristsonar.eval.guard import GuardReport
from wristsonar.eval.metrics import MetricSet
from wristsonar.protocol import Measurement, Split

__all__ = [
    "CLEAR_MARGIN",
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
        reference = self.rows[0].metrics.mpjpe
        if not reference.comparable_to(metrics.mpjpe):
            raise ValueError(
                "refusing to place non-comparable measurements in one table.\n"
                f"  existing: {reference.protocol.describe()}\n"
                f"  incoming: {metrics.mpjpe.protocol.describe()}\n"
                "Split, ground truth and calibration budget must match for a "
                "row-to-row comparison to mean anything. Build a second "
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
        self.guards = guards

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
                "every measurement here is WITHIN_SESSION. Train and test came "
                "from one continuous wearing, so the device never moved "
                "between fitting and testing and the number measures "
                "interpolation inside one mounting geometry. It is the "
                "weakest protocol in the field. Add at least a cross-session "
                "measurement before quoting anything."
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
        return tuple(reasons)

    @property
    def can_headline(self) -> bool:
        return not self.refusal_reasons

    def headline(self) -> str:
        """The one line result, or a refusal explaining what is missing."""
        reasons = self.refusal_reasons
        if reasons:
            body = "\n".join(f"  - {r}" for r in reasons)
            return "NO HEADLINE RESULT. This report cannot be quoted:\n" + body
        model = self.best_model
        assert model is not None
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
                f"  {'predictor':<24}{'kind':<10}{'MPJPE':>9}{'PA':>9}"
                f"{'tips':>9}{'PCK':>8}"
            )
            out.append("  " + "-" * (width - 4))
            for row in sorted(self.rows, key=lambda r: r.mpjpe.value):
                m = row.metrics
                out.append(
                    f"  {row.name:<24}{row.kind:<10}"
                    f"{m.mpjpe.value:9.2f}{m.pa_mpjpe.value:9.2f}"
                    f"{m.breakdown.fingertip_mm:9.2f}{m.pck.value:8.3f}"
                )
                out.append(f"      protocol: {m.mpjpe.protocol.describe()}")
            out.append("")
            out.append(
                "  MPJPE and PA are millimetres, tips is the mean over the five "
                "fingertips,"
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
                    "pa_mpjpe_mm": r.metrics.pa_mpjpe.value,
                    "fingertip_mm": r.metrics.breakdown.fingertip_mm,
                    "non_fingertip_mm": r.metrics.breakdown.non_fingertip_mm,
                    "pck": r.metrics.pck.value,
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
            "validity_statement": self.validity_statement(),
            "notes": self.notes,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)
