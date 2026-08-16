"""What the report refuses to say, and what it insists on saying."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from tests.eval.synthetic import make_dataset
from wristsonar.eval.baselines import MeanPoseBaseline, evaluate_baselines
from wristsonar.eval.calibration import sweep_calibration
from wristsonar.eval.guard import GuardBinding, GuardReport, Waiver, run_guards
from wristsonar.eval.guard import GuardName as G
from wristsonar.eval.metrics import Alignment, MetricSet, PoseArray, evaluate
from wristsonar.eval.report import CLEAR_MARGIN, MultiSplitReport, Report
from wristsonar.eval.splits import (
    Dataset,
    IndexArray,
    SplitIndices,
    cross_device_split,
    cross_session_split,
    cross_user_folds,
    within_session_split,
)
from wristsonar.protocol import GroundTruth, Protocol, Split
from wristsonar.types import N_JOINTS


def _protocol(split: Split, subjects: int = 6) -> Protocol:
    return Protocol(
        split=split,
        dataset="synthetic",
        dataset_version="0",
        ground_truth=GroundTruth.SYNTHETIC,
        subjects=subjects,
    )


def _metrics(
    data: Dataset, test: IndexArray, protocol: Protocol, *, error_m: float
) -> MetricSet:
    truth = data.poses[test]
    pred = truth + error_m
    return evaluate(pred, truth, protocol, alignment=Alignment.ROOT)


def _fitter(dataset: Dataset, train: IndexArray, test: IndexArray) -> PoseArray:
    baseline = MeanPoseBaseline()
    baseline.fit(dataset, train)
    return baseline.predict(dataset, test)


def _waivers() -> list[Waiver]:
    return [
        Waiver(
            guard=guard,
            reason=(
                "synthetic smoke data used to exercise the reporting path, "
                "not a claim about any hardware or any person"
            ),
            approved_by="test-suite",
            expires_on=date.today() + timedelta(days=30),
        )
        for guard in G
    ]


def test_a_within_session_only_report_refuses_to_headline() -> None:
    data = make_dataset(n_participants=3, n_sessions=1, n_frames=20)
    fold = within_session_split(data, test_fraction=0.3)
    protocol = _protocol(Split.WITHIN_SESSION, subjects=3)

    report = Report(title="within session only")
    report.add_model("model", _metrics(data, fold.test, protocol, error_m=0.004))
    report.add_baselines(evaluate_baselines(data, fold, protocol))
    report.set_guards(
        run_guards(
            data,
            fold,
            np.full(fold.n_test, 4.0),
            protocol,
            waivers=_waivers(),
        )
    )

    assert not report.can_headline
    reasons = " ".join(report.refusal_reasons)
    assert "WITHIN_SESSION" in reasons
    assert "weakest protocol" in reasons

    headline = report.headline()
    assert headline.startswith("NO HEADLINE RESULT")
    assert "cross-session" in headline

    text = report.render_text()
    assert "NO HEADLINE RESULT" in text
    assert "within-session only" in text


def test_missing_baselines_and_missing_guards_both_block_a_headline() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=15)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)

    bare = Report(title="model only")
    bare.add_model("model", _metrics(data, fold.test, protocol, error_m=0.002))
    reasons = " ".join(bare.refusal_reasons)
    assert "no baselines" in reasons
    assert "guards were not run" in reasons

    empty = Report(title="nothing")
    assert "no measurements" in " ".join(empty.refusal_reasons)


def test_a_cross_user_report_with_everything_renders_a_headline() -> None:
    data = make_dataset(n_participants=6, n_sessions=2, n_frames=25)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=6)

    report = Report(title="cross-user leave-one-out, fold P00")
    report.add_baselines(evaluate_baselines(data, fold, protocol))
    # A model good enough to clearly beat every trivial predictor.
    report.add_model("wristsonar", _metrics(data, fold.test, protocol, error_m=0.0002))
    report.set_guards(
        run_guards(
            data, fold, np.full(fold.n_test, 1.0), protocol, waivers=_waivers()
        )
    )
    report.set_calibration(
        sweep_calibration(
            data,
            fold,
            _fitter,
            protocol,
            sample_seconds=1.0,
            budgets=(0.0,),
        )
    )

    assert report.can_headline
    assert all(report.beaten_baselines().values())
    assert "clearly beats all" in report.verdict()

    headline = report.headline()
    assert headline.startswith("wristsonar")
    assert "split=cross-user" in headline

    text = report.render_text()
    # Baselines share the table with the model, not an appendix.
    table = text.split("RESULTS")[1].split("VERDICT")[0]
    for name in ("mean-pose", "per-subject-mean-pose", "nearest-neighbour"):
        assert name in table
    assert "wristsonar" in table
    # Every printed number carries its protocol.
    assert text.count("protocol: split=cross-user") == len(report.rows)
    assert "calibration curve" in text
    assert "PER-JOINT BREAKDOWN" in text
    assert "VALID ONLY UNDER THESE CONDITIONS" in text
    assert text.rstrip().endswith("=" * 78)


def test_a_model_inside_the_margin_is_reported_as_not_beating_the_baselines() -> None:
    data = make_dataset(n_participants=5, n_sessions=2, n_frames=20)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=5)

    report = Report(title="a model that has not earned it")
    baselines = evaluate_baselines(data, fold, protocol)
    report.add_baselines(baselines)
    best_baseline = min(m.mpjpe.value for m in baselines.values())
    # Land just inside the clear-margin, which counts as not beaten.
    target = best_baseline * (1.0 - CLEAR_MARGIN / 2.0)
    truth = data.poses[fold.test]
    # Root alignment removes a uniform translation, so move one non-wrist
    # joint.  Spread the desired MPJPE over all joints by scaling the one
    # displaced joint accordingly.
    pred = truth.copy()
    pred[:, 1, 0] += target * N_JOINTS / 1000.0
    report.add_model("wristsonar", evaluate(pred, truth, protocol))
    report.set_guards(
        run_guards(
            data, fold, np.full(fold.n_test, 9.0), protocol, waivers=_waivers()
        )
    )

    beaten = report.beaten_baselines()
    assert not all(beaten.values())
    assert "does NOT clearly beat" in report.verdict()
    assert "does NOT clearly beat" in report.headline()


def test_non_comparable_rows_are_refused() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]

    report = Report(title="mixed")
    report.add_model(
        "a", _metrics(data, fold.test, _protocol(Split.CROSS_USER), error_m=0.001)
    )
    with pytest.raises(ValueError, match="non-comparable measurements"):
        report.add_baseline(
            "b",
            _metrics(data, fold.test, _protocol(Split.WITHIN_SESSION), error_m=0.001),
        )


def test_json_output_carries_the_refusals_and_the_curve() -> None:
    data = make_dataset(n_participants=4, n_sessions=1, n_frames=20)
    fold = within_session_split(data, test_fraction=0.25)
    protocol = _protocol(Split.WITHIN_SESSION, subjects=4)

    report = Report(title="machine readable", notes="smoke run")
    report.add_baselines(evaluate_baselines(data, fold, protocol))
    report.add_model("wristsonar", _metrics(data, fold.test, protocol, error_m=0.0005))
    report.set_guards(
        run_guards(
            data,
            fold,
            np.full(fold.n_test, 2.0),
            protocol,
            raise_on_violation=False,
        )
    )
    report.set_calibration(
        sweep_calibration(
            data, fold, _fitter, protocol, sample_seconds=1.0, budgets=(0.0,)
        )
    )

    payload = json.loads(report.to_json())
    assert payload["can_headline"] is False
    assert payload["refusal_reasons"]
    assert len(payload["rows"]) == len(report.rows)
    assert payload["rows"][0]["split"] == "WITHIN_SESSION"
    assert "split=within-session" in payload["rows"][0]["protocol"]
    assert len(payload["rows"][0]["per_joint_mm"]) == 21
    assert payload["calibration_curve"] is not None
    assert payload["guards"] is not None
    assert "VALID ONLY" in payload["validity_statement"]
    assert payload["notes"] == "smoke run"


def test_validity_statement_names_what_was_never_measured() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=15)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)

    report = Report(title="cross-user only")
    report.add_model("wristsonar", _metrics(data, fold.test, protocol, error_m=0.001))
    statement = report.validity_statement()

    assert "strongest split     CROSS_USER" in statement
    assert "cross-device performance" in statement
    assert "poses outside the training" in statement
    assert "5.7 cm range resolution" in statement
    assert "misquoted" in statement


def _binding(split: Split) -> GuardBinding:
    return GuardBinding(
        split=split,
        split_name="fabricated",
        dataset="synthetic",
        dataset_version="0",
        n_train=1,
        n_test=1,
        fingerprint="0" * 64,
    )


def _full_report(
    data: Dataset,
    fold: SplitIndices,
    protocol: Protocol,
    *,
    title: str,
    model_name: str = "wristsonar",
    error_m: float = 0.0002,
) -> Report:
    """A report with baselines, a model and waived guards, for container tests."""
    report = Report(title=title)
    report.add_baselines(evaluate_baselines(data, fold, protocol))
    report.add_model(
        model_name, _metrics(data, fold.test, protocol, error_m=error_m)
    )
    report.set_guards(
        run_guards(
            data,
            fold,
            np.full(fold.n_test, 3.0),
            protocol,
            waivers=_waivers(),
            raise_on_violation=False,
        )
    )
    return report


def test_an_empty_guard_report_cannot_be_built_let_alone_attached() -> None:
    with pytest.raises(ValueError, match="finding for every guard"):
        GuardReport(findings=(), binding=_binding(Split.CROSS_USER))


def test_guards_computed_on_an_easier_split_are_refused() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=15)
    easy = within_session_split(data, test_fraction=0.3)
    hard = cross_user_folds(data)[0]
    easy_protocol = _protocol(Split.WITHIN_SESSION, subjects=4)
    hard_protocol = _protocol(Split.CROSS_USER, subjects=4)

    easy_guards = run_guards(
        data,
        easy,
        np.full(easy.n_test, 3.0),
        easy_protocol,
        raise_on_violation=False,
    )
    report = Report(title="cross-user rows with within-session guards")
    report.add_baselines(evaluate_baselines(data, hard, hard_protocol))
    report.add_model(
        "wristsonar", _metrics(data, hard.test, hard_protocol, error_m=0.0002)
    )
    with pytest.raises(ValueError, match="not computed on these rows"):
        report.set_guards(easy_guards)


def test_a_baselines_only_report_refuses_instead_of_asserting() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=15)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)

    report = Report(title="baselines only")
    report.add_baselines(evaluate_baselines(data, fold, protocol))
    report.set_guards(
        run_guards(
            data,
            fold,
            np.full(fold.n_test, 3.0),
            protocol,
            waivers=_waivers(),
            raise_on_violation=False,
        )
    )

    assert not report.can_headline
    assert "baselines only" in " ".join(report.refusal_reasons)
    assert report.headline().startswith("NO HEADLINE RESULT")
    assert "baselines only" in report.verdict()


def test_a_procrustes_row_cannot_join_a_root_aligned_table() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)
    truth = data.poses[fold.test]

    report = Report(title="alignment mixture")
    report.add_baseline(
        "mean-pose", evaluate(truth + 0.01, truth, protocol, alignment=Alignment.ROOT)
    )
    with pytest.raises(ValueError, match="Procrustes-aligned row"):
        report.add_model(
            "wristsonar",
            evaluate(truth, truth, protocol, alignment=Alignment.PROCRUSTES),
        )


def test_the_alignment_is_printed_beside_every_number() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)
    metrics = _metrics(data, fold.test, protocol, error_m=0.001)

    assert "root aligned" in metrics.mpjpe.protocol.describe()
    assert "notes=" in metrics.mpjpe.protocol.describe()


def test_both_joint_conventions_are_reported() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)
    metrics = _metrics(data, fold.test, protocol, error_m=0.003)

    # Root alignment zeroes the wrist, so the 21-joint mean is exactly 20/21
    # of the 20-joint mean, which is the 4.76 percent understatement.
    assert metrics.mpjpe.value == pytest.approx(
        metrics.mpjpe_excl_root.value * 20.0 / 21.0
    )

    report = Report(title="both conventions")
    report.add_model("wristsonar", metrics)
    text = report.render_text()
    assert "MPJPE21" in text
    assert "MPJPE20" in text
    assert "4.76 percent" in text


def test_a_multi_split_report_refuses_while_an_honest_split_is_missing() -> None:
    data = make_dataset(
        n_participants=6, n_sessions=3, n_frames=20, devices=("watch-a", "watch-b")
    )
    within = within_session_split(data, test_fraction=0.3)
    cross_session = cross_session_split(data, participant="P00")

    combined = MultiSplitReport(title="wristsonar, every split")
    combined.add(
        _full_report(
            data,
            within,
            _protocol(Split.WITHIN_SESSION, subjects=6),
            title="within-session",
        )
    )
    combined.add(
        _full_report(
            data,
            cross_session,
            _protocol(Split.CROSS_SESSION, subjects=6),
            title="cross-session",
        )
    )

    assert not combined.can_headline
    reasons = " ".join(combined.refusal_reasons)
    assert "CROSS_USER" in reasons
    assert "CROSS_DEVICE" in reasons
    text = combined.render_text()
    assert "NO HEADLINE RESULT" in text
    # The reports it does hold are still rendered in full, weakest first.
    assert text.index("WITHIN_SESSION") < text.index("CROSS_SESSION")
    assert "NOT MEASURED" in text


def test_a_multi_split_report_with_every_honest_split_renders_one_headline() -> None:
    data = make_dataset(
        n_participants=6, n_sessions=3, n_frames=20, devices=("watch-a", "watch-b")
    )
    folds = {
        Split.WITHIN_SESSION: within_session_split(data, test_fraction=0.3),
        Split.CROSS_SESSION: cross_session_split(data, participant="P00"),
        Split.CROSS_USER: cross_user_folds(data)[0],
        Split.CROSS_DEVICE: cross_device_split(data, held_out_device="watch-b"),
    }
    combined = MultiSplitReport(title="wristsonar, every split")
    for split, fold in folds.items():
        combined.add(
            _full_report(
                data,
                fold,
                _protocol(split, subjects=6),
                title=split.name.lower(),
            )
        )

    assert combined.missing_splits == ()
    assert combined.can_headline, combined.refusal_reasons
    # The headline is the strongest split's, not the most flattering one's.
    assert "split=cross-device" in combined.headline()

    text = combined.render_text()
    assert text.index("WITHIN_SESSION") < text.index("CROSS_DEVICE")
    payload = json.loads(combined.to_json())
    assert set(payload["reports"]) == {s.name for s in Split}
    assert payload["missing_splits"] == []
    assert payload["models"] == ["wristsonar"]


def test_a_multi_split_report_refuses_two_reports_for_one_split() -> None:
    data = make_dataset(n_participants=4, n_sessions=2, n_frames=10)
    fold = cross_user_folds(data)[0]
    protocol = _protocol(Split.CROSS_USER, subjects=4)
    combined = MultiSplitReport(title="duplicate")
    combined.add(_full_report(data, fold, protocol, title="first"))
    with pytest.raises(ValueError, match="already present"):
        combined.add(_full_report(data, fold, protocol, title="second"))


def test_a_multi_split_report_notices_a_different_model_per_split() -> None:
    data = make_dataset(
        n_participants=6, n_sessions=3, n_frames=20, devices=("watch-a", "watch-b")
    )
    combined = MultiSplitReport(title="mixed systems")
    combined.add(
        _full_report(
            data,
            cross_session_split(data, participant="P00"),
            _protocol(Split.CROSS_SESSION, subjects=6),
            title="cross-session",
        )
    )
    combined.add(
        _full_report(
            data,
            cross_user_folds(data)[0],
            _protocol(Split.CROSS_USER, subjects=6),
            title="cross-user",
            model_name="wristsonar-lite",
        )
    )
    combined.add(
        _full_report(
            data,
            cross_device_split(data, held_out_device="watch-b"),
            _protocol(Split.CROSS_DEVICE, subjects=6),
            title="cross-device",
        )
    )

    reasons = " ".join(combined.refusal_reasons)
    assert "is not one result" in reasons
