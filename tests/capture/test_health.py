from __future__ import annotations

from wristsonar.capture import DuplexValidator


def test_capture_validator_accepts_the_actual_watchhand_contract() -> None:
    validator = DuplexValidator()
    validator.observe(sample_rate=48_000, frame_samples=600, continuous=True)
    assert validator.report().accepted


def test_capture_validator_refuses_silent_resampling_or_gap() -> None:
    validator = DuplexValidator()
    validator.observe(sample_rate=44_100, frame_samples=600, continuous=True)
    validator.observe(sample_rate=44_100, frame_samples=600, continuous=False)
    report = validator.report()
    assert not report.accepted
    assert report.discontinuities == 1


def test_capture_validator_reports_an_empty_session_without_exploding() -> None:
    """``report`` used to read ``_last_samples`` before anything had set it."""
    report = DuplexValidator().report()
    assert not report.accepted
    assert report.frames == 0
    assert report.reason == "no frames observed"


def test_capture_validator_acceptance_is_not_read_back_out_of_its_own_prose() -> None:
    """Acceptance is decided, not recovered by matching the reason string."""
    validator = DuplexValidator()
    validator.observe(sample_rate=48_000, frame_samples=599, continuous=True)
    report = validator.report()
    assert not report.accepted
    assert "599" in report.reason
