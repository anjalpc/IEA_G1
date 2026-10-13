"""Pure regression test for C3 deployment-export compatibility."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from simulation.isaac_lab.g1_stairs.deployment_export import (
    export_deploy_cfg_compatible,
)


class _Term:
    def __init__(self, action_dim):
        self.action_dim = action_dim


class _Manager:
    def __init__(self):
        self._live_terms = {
            "JointPositionAction": _Term(29),
            "gait_frequency": _Term(1),
        }
        self.total_action_dim = 30
        self.action = "live-action"
        self.prev_action = "live-prev-action"

    @property
    def active_terms(self):
        return list(self._live_terms)

    def get_term(self, name):
        return self._live_terms[name]


class _Env:
    def __init__(self):
        self.action_manager = _Manager()
        self.observation_probe = None


class TestDeploymentExport(unittest.TestCase):
    def test_c3_exports_only_physical_action_and_preserves_live_manager(self):
        env = _Env()
        original_manager = env.action_manager

        def fake_unitree_exporter(export_env, log_dir):
            # Unitree's action loop sees only the physical term.
            self.assertEqual(
                export_env.action_manager.active_terms,
                ["JointPositionAction"],
            )
            self.assertEqual(
                list(export_env.action_manager._terms),
                ["JointPositionAction"],
            )
            # Observation functions can still resolve the internal gait term.
            self.assertEqual(
                export_env.action_manager.get_term("gait_frequency").action_dim,
                1,
            )
            path = Path(log_dir) / "params" / "deploy.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Mirror Unitree's exporter: yaml.dump may emit Python-specific
            # tags, including both tuple and slice objects. The compatibility
            # wrapper must not need to parse these back in.
            path.write_text(
                yaml.dump(
                    {
                        "actions": {
                            "JointPositionAction": {
                                "clip": (-1.0, 1.0),
                                "joint_ids": slice(None),
                            }
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

        resolved = {
            "actions": {"dimension": 30, "physical_joint_dimension": 29},
            "gait": {"enabled": True, "mode": "learned"},
        }

        with TemporaryDirectory() as temp:
            export_deploy_cfg_compatible(
                env, temp, resolved, fake_unitree_exporter
            )

            deploy_text = (Path(temp) / "params" / "deploy.yaml").read_text()
            self.assertIn("JointPositionAction:", deploy_text)
            self.assertIn("python/tuple", deploy_text)
            self.assertIn("python/object/apply:builtins.slice", deploy_text)
            self.assertNotIn("gait_frequency", deploy_text)

            gait = yaml.safe_load(
                (Path(temp) / "params" / "gait.yaml").read_text()
            )
            self.assertEqual(gait["physical_action_dim"], 29)
            self.assertEqual(gait["policy_action_dim"], 30)
            self.assertEqual(gait["frequency_min_hz"], 0.7)
            self.assertEqual(gait["frequency_nominal_hz"], 1.0)
            self.assertEqual(gait["frequency_max_hz"], 1.3)
            self.assertEqual(gait["phase_offset"], 0.5)
            self.assertEqual(gait["phase_observation"], ["sin", "cos"])

        self.assertIs(env.action_manager, original_manager)
        self.assertEqual(
            env.action_manager.active_terms,
            ["JointPositionAction", "gait_frequency"],
        )
        self.assertEqual(env.action_manager.total_action_dim, 30)

    def test_non_gait_export_passes_through(self):
        env = _Env()
        calls = []

        def exporter(export_env, log_dir):
            calls.append((export_env, log_dir))

        export_deploy_cfg_compatible(
            env, "/tmp/example", {"gait": {"enabled": False}}, exporter
        )
        self.assertEqual(calls, [(env, "/tmp/example")])


if __name__ == "__main__":
    unittest.main()
