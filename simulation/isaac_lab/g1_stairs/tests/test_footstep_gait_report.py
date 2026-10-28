"""Gait report on a synthetic alternating trace."""
import importlib.util
import pathlib
import unittest

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / "scripts" / "evaluation" / "footstep_gait_report.py")
spec = importlib.util.spec_from_file_location("footstep_gait_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def synthetic(steps=300, period=14, stride=0.1, terminate_at=None):
    rows = [{"sample": "initial"}]
    for k in range(steps):
        phase = (k // period) % 2           # 0: left loaded only, 1: right loaded only
        touch = k // period
        left_x = stride * (touch + (1 if phase == 0 else 0)) / 1.0
        right_x = stride * (touch + (1 if phase == 1 else 0))
        rows.append({
            "sample": "post_physics_pre_reset",
            "env_origin_w": [0.0, 0.0, 0.0],
            "root_pos_w": [0.0, 0.0, 0.8],
            "root_lin_vel_w": [0.35, 0.0, 0.0],
            "command_base_velocity": [0.35, 0.0, 0.0],
            "terminated": terminate_at is not None and k == terminate_at,
            "time_out": False,
            "footstep_landing_audit": {
                "supported_lr": [phase == 0, phase == 1],
                "sole_centre_local": [[left_x, 0.1175, 0.0], [right_x, -0.1175, 0.0]],
            },
            "planned_footsteps": {"completed": 0},
        })
    return rows


class GaitReportTests(unittest.TestCase):
    def test_symmetric_alternating_gait(self):
        m = report.gait_metrics(synthetic())
        self.assertEqual(m["alternation_ratio"], 1.0)
        self.assertEqual(m["timing_symmetry"], 1.0)
        self.assertAlmostEqual(m["interval_to_left_s"], 0.28)
        self.assertEqual(m["ended_by"], "trace_end")
        self.assertEqual(m["mean_root_vx"], 0.35)

    def test_stops_at_first_termination(self):
        m = report.gait_metrics(synthetic(terminate_at=99))
        self.assertEqual(m["duration_s"], 2.0)
        self.assertEqual(m["ended_by"], "fall/terminated")


if __name__ == "__main__":
    unittest.main()
