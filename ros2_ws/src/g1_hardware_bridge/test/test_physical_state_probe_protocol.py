"""Contract for sanitized physical-state probe observations."""

import math

import pytest

from g1_hardware_bridge.state_probe.protocol import (
    APPROVED_TOPICS,
    OUTPUT_PREFIX,
    REASON_FIELDS_MISSING,
    REASON_INSUFFICIENT_SAMPLES,
    REASON_NO_SAMPLES,
    REASON_SOURCE_NOT_ADVANCING,
    REASON_TYPE_UNAVAILABLE,
    REASON_VALID,
    ProbeSummary,
    encode_summary,
    normalize_observation,
)


def test_protocol_constants_are_exact():
    assert APPROVED_TOPICS == (
        "rt/lowstate",
        "rt/secondary_imu",
        "rt/sportmodestate",
    )
    assert OUTPUT_PREFIX == "PHYSICAL_STATE_PROBE="
    assert REASON_VALID == "VALID"


def test_normalizes_valid_observation_without_retaining_values():
    observation = normalize_observation(
        topic="rt/lowstate",
        progress=42,
        fields={
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "gyroscope": [0.1, 0.2, 0.3],
            "accelerometer": [0.0, 0.0, 9.81],
        },
    )

    assert observation == {
        "topic": "rt/lowstate",
        "progress": 42,
        "field_lengths": {
            "accelerometer": 3,
            "gyroscope": 3,
            "quaternion": 4,
        },
        "finite": True,
    }

    serialized = repr(observation)
    assert "9.81" not in serialized
    assert "0.3" not in serialized


def test_rejects_unapproved_topic():
    with pytest.raises(ValueError, match="approved"):
        normalize_observation(
            topic="rt/private",
            progress=1,
            fields={"quaternion": [1.0, 0.0, 0.0, 0.0]},
        )


@pytest.mark.parametrize("progress", [True, 1.5, "1", None])
def test_rejects_invalid_progress_token(progress):
    with pytest.raises(TypeError, match="progress"):
        normalize_observation(
            topic="rt/lowstate",
            progress=progress,
            fields={"quaternion": [1.0, 0.0, 0.0, 0.0]},
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_field_values(value):
    with pytest.raises(ValueError, match="finite"):
        normalize_observation(
            topic="rt/lowstate",
            progress=1,
            fields={"gyroscope": [0.0, value, 0.0]},
        )


@pytest.mark.parametrize(
    "fields",
    [
        None,
        [],
        {},
        {"quaternion": "invalid"},
        {"": [1.0]},
        {"quaternion": []},
    ],
)
def test_rejects_invalid_field_shapes(fields):
    with pytest.raises((TypeError, ValueError), match="field"):
        normalize_observation(
            topic="rt/lowstate",
            progress=1,
            fields=fields,
        )

def observation(progress, value=0.1):
    return normalize_observation(
        topic="rt/lowstate",
        progress=progress,
        fields={
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "gyroscope": [value, 0.0, 0.0],
        },
    )


def test_summarizes_advancing_valid_observations():
    summary = ProbeSummary(minimum_samples=2)
    summary.observe(observation(10), observed_at=1.0)
    summary.observe(observation(11), observed_at=1.02)

    result = summary.finalize(
        requested_duration_seconds=10.0,
        actual_elapsed_seconds=10.1,
    )

    assert set(result) == {
        "schema_version",
        "requested_duration_seconds",
        "actual_elapsed_seconds",
        "topics",
    }
    assert result["schema_version"] == 1
    assert result["topics"]["rt/lowstate"] == {
        "available": True,
        "sample_count": 2,
        "cadence_hz": 50.0,
        "progress_advanced": True,
        "field_lengths": {
            "gyroscope": 3,
            "quaternion": 4,
        },
        "reason": REASON_VALID,
    }
    assert result["topics"]["rt/secondary_imu"]["reason"] == (
        REASON_NO_SAMPLES
    )


def test_reports_repeated_progress_without_raw_values():
    summary = ProbeSummary(minimum_samples=2)
    summary.observe(observation(10, value=123.456), observed_at=1.0)
    summary.observe(observation(10, value=654.321), observed_at=1.02)

    result = summary.finalize(10.0, 10.0)
    topic = result["topics"]["rt/lowstate"]

    assert topic["progress_advanced"] is False
    assert topic["reason"] == REASON_SOURCE_NOT_ADVANCING
    assert "123.456" not in repr(result)
    assert "654.321" not in repr(result)


def test_reports_insufficient_samples():
    summary = ProbeSummary(minimum_samples=2)
    summary.observe(observation(10), observed_at=1.0)

    result = summary.finalize(10.0, 10.0)

    assert result["topics"]["rt/lowstate"]["reason"] == (
        REASON_INSUFFICIENT_SAMPLES
    )


def test_reports_unavailable_candidate_type():
    summary = ProbeSummary(minimum_samples=2)
    summary.mark_unavailable("rt/sportmodestate")

    result = summary.finalize(10.0, 10.0)

    assert result["topics"]["rt/sportmodestate"] == {
        "available": False,
        "sample_count": 0,
        "cadence_hz": None,
        "progress_advanced": False,
        "field_lengths": {},
        "reason": REASON_TYPE_UNAVAILABLE,
    }


def test_reports_inconsistent_field_shapes():
    summary = ProbeSummary(minimum_samples=2)
    summary.observe(observation(10), observed_at=1.0)
    changed = normalize_observation(
        topic="rt/lowstate",
        progress=11,
        fields={"quaternion": [1.0, 0.0, 0.0, 0.0]},
    )
    summary.observe(changed, observed_at=1.02)

    result = summary.finalize(10.0, 10.0)

    assert result["topics"]["rt/lowstate"]["reason"] == (
        REASON_FIELDS_MISSING
    )


@pytest.mark.parametrize(
    "minimum_samples",
    [True, 0, -1, 1.5],
)
def test_rejects_invalid_minimum_samples(minimum_samples):
    with pytest.raises((TypeError, ValueError), match="minimum"):
        ProbeSummary(minimum_samples=minimum_samples)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0])
def test_rejects_invalid_observation_time(value):
    summary = ProbeSummary(minimum_samples=1)
    with pytest.raises(ValueError, match="time"):
        summary.observe(observation(10), observed_at=value)


def test_encodes_only_the_allowlisted_summary_schema():
    summary = ProbeSummary(minimum_samples=1)
    summary.observe(observation(10), observed_at=1.0)
    result = summary.finalize(10.0, 10.0)

    line = encode_summary(result)

    assert line.startswith(OUTPUT_PREFIX)
    assert "quaternion" in line
    assert "123.456" not in line


def test_rejects_extra_summary_fields():
    summary = ProbeSummary(minimum_samples=1)
    result = summary.finalize(10.0, 10.0)
    result["private"] = "forbidden"

    with pytest.raises(ValueError, match="fields"):
        encode_summary(result)

