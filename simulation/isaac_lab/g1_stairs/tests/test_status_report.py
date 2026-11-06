"""training_status.py (simctl status) and trace_report.py gate logic."""
import importlib.util
import pathlib
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "evaluation"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


status = load("training_status")
report = load("trace_report")


def log_text(iterations):
    out = []
    for it, falls in iterations:
        out += [f"\x1b[1m Learning iteration {it}/999 \x1b[0m",
                "             Mean action noise std: 0.35",
                f"Episode_Termination/bad_orientation: {falls:.4f}",
                "   Episode_Reward/foot_stumble: -0.0500",
                "[POLICY_SHAPING] means beyond clip 0.500, summed excess per state 1.0, bound loss 0.01"]
    return "\n".join(out) + "\n"


class TrainingStatusTests(unittest.TestCase):
    def parse(self, iterations):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(log_text(iterations))
        return status.parse_log(f.name)

    def test_parses_blocks_colour_codes_and_shaping(self):
        blocks = self.parse([(5, 0.2), (6, 0.1)])
        self.assertEqual(sorted(blocks), [5, 6])
        self.assertAlmostEqual(blocks[6]["Episode_Termination/bad_orientation"], 0.1)
        self.assertAlmostEqual(blocks[5]["Mean action noise std"], 0.35)
        self.assertAlmostEqual(blocks[5]["policy_shaping_beyond_clip"], 0.5)
        self.assertAlmostEqual(status.series(blocks, "Episode_Reward/foot_stumble", True)[0][1], 0.05)

    def test_trend_direction_and_warning(self):
        self.assertEqual(status.trend([0.2] * 100 + [0.1] * 100, "down")[1], False)
        text, worse = status.trend([0.1] * 100 + [0.2] * 100, "down")
        self.assertTrue(worse)
        self.assertIn("WORSE", text)
        self.assertIn("steady", status.trend([1.0] * 100 + [1.02] * 100, "up")[0])
        self.assertEqual(status.trend([1.0] * 10, "up"), ("too early", False))

    def test_sparkline_runs_low_to_high(self):
        line = status.sparkline(list(range(120)))
        self.assertEqual((line[0], line[-1], len(line)), ("▁", "█", 12))


def ep(clean=True, ended="time_out", heel=2.0, strikes=0, drift=0.1, lateral=5.0, vx=0.37, cmd=0.35):
    return dict(clean=clean, ended=ended, first_heel=heel, strikes=strikes, drift=drift,
                lateral=lateral, vx=vx, cmd=cmd)


class TraceGateTests(unittest.TestCase):
    def verdict(self, height, eps, top=4):
        return {name: ok for name, ok in report.judge(height, eps, top)}

    def test_top_height_needs_clean_climbs_heels_and_no_strikes(self):
        good = [ep()] * 8 + [ep(clean=False)] * 2
        self.assertTrue(all(self.verdict(4, good).values()))
        bad = [ep()] * 7 + [ep(clean=False, heel=-1.0)] * 3
        checks = self.verdict(4, bad)
        self.assertFalse(checks["clean >= 80%"])
        self.assertFalse(checks["first heel >= 1 cm in 90%"])
        self.assertFalse(self.verdict(4, [ep(strikes=1)] + [ep()] * 9)["no toe strikes"])

    def test_lower_height_stricter_and_falls_fail(self):
        nine = [ep()] * 9 + [ep(clean=False)]
        self.assertTrue(all(self.verdict(2, nine).values()))
        self.assertFalse(self.verdict(2, [ep()] * 8 + [ep(clean=False)] * 2)["clean >= 90%"])
        self.assertFalse(self.verdict(2, [ep()] * 9 + [ep(ended="fall")])["no falls/stalls"])
        self.assertNotIn("no toe strikes", self.verdict(2, nine))

    def test_flat_turning_sideways_and_speed(self):
        self.assertTrue(all(self.verdict(0, [ep()] * 3).values()))
        self.assertFalse(self.verdict(0, [ep(drift=1.0)] + [ep()] * 2)["turn <= 0.7 deg/s"])
        self.assertFalse(self.verdict(0, [ep(lateral=60.0)] * 3)["sideways <= 35 cm"])
        self.assertFalse(self.verdict(0, [ep(vx=0.5)] * 3)["speed 0.85-1.20 x cmd"])


if __name__ == "__main__":
    unittest.main()
