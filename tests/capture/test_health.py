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
