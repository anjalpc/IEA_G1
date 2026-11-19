"""Input-context checks for diagnostic depth processing."""

from numbers import Integral


def cloud_context_reason(cloud_stamp_ns, imu_stamp_ns, now_ns, *,
                         cloud_frame, imu_frame, previous_stamp_ns=None):
    """Return READY or an explicit diagnostic rejection reason."""
    stamps = [cloud_stamp_ns, imu_stamp_ns, now_ns]
    if previous_stamp_ns is not None:
        stamps.append(previous_stamp_ns)
    if any(not isinstance(t, Integral) or isinstance(t, bool) or t <= 0
           for t in stamps):
        return "INVALID_STAMP"
    if cloud_frame != "d435_depth_optical_frame":
        return "CLOUD_FRAME_INVALID"
    if imu_frame != "imu_in_pelvis":
        return "IMU_FRAME_INVALID"
    if previous_stamp_ns is not None and cloud_stamp_ns <= previous_stamp_ns:
        return "NONADVANCING_CLOUD"
    age = now_ns - cloud_stamp_ns
    if age < -50_000_000:
        return "CLOUD_FUTURE"
    if age > 350_000_000:
        return "CLOUD_STALE"
    if abs(imu_stamp_ns - cloud_stamp_ns) > 50_000_000:
        return "IMU_UNSYNCHRONIZED"
    return "READY"
