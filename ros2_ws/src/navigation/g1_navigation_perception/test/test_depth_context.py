"""Diagnostic depth input-context regression checks."""

import pytest

from g1_navigation_perception.depth_context import cloud_context_reason


@pytest.mark.parametrize("changes, expected", [
    ({}, "READY"),
    ({"cloud_stamp_ns": 0}, "INVALID_STAMP"),
    ({"cloud_frame": "pelvis"}, "CLOUD_FRAME_INVALID"),
    ({"imu_frame": "wrong"}, "IMU_FRAME_INVALID"),
    ({"previous_stamp_ns": 2_000_000_000}, "NONADVANCING_CLOUD"),
    ({"now_ns": 1_949_999_999}, "CLOUD_FUTURE"),
    ({"now_ns": 2_350_000_001}, "CLOUD_STALE"),
    ({"imu_stamp_ns": 2_050_000_001}, "IMU_UNSYNCHRONIZED"),
    ({"now_ns": 2_350_000_000, "imu_stamp_ns": 2_050_000_000}, "READY"),
])
def test_diagnostic_context_boundaries(changes, expected):
    arguments = dict(cloud_stamp_ns=2_000_000_000,
                     imu_stamp_ns=2_000_000_000, now_ns=2_100_000_000,
                     cloud_frame="d435_depth_optical_frame",
                     imu_frame="imu_in_pelvis")
    arguments.update(changes)
    assert cloud_context_reason(**arguments) == expected
