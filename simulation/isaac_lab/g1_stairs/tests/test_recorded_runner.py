"""Regression tests for recorded training-process supervision."""

from pathlib import Path
import sys
import tempfile
import time
import unittest

from simulation.isaac_lab.g1_stairs.scripts.training.run_recorded_experiment import (
    stream_training_process,
)


class RecordedRunnerTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(__import__("os"), "fork"), "requires POSIX fork")
    def test_parent_returns_when_failed_child_leaves_inherited_pipe_open(self):
        program = (
            "import os,time; pid=os.fork(); "
            "time.sleep(2) if pid == 0 else os._exit(7)"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = time.monotonic()
            result = stream_training_process(
                [sys.executable, "-c", program],
                root / "console.log",
                root,
                root / "resolved_config.yaml",
            )
            elapsed = time.monotonic() - started

        self.assertEqual(result, 7)
        self.assertLess(elapsed, 1.5)


if __name__ == "__main__":
    unittest.main()
