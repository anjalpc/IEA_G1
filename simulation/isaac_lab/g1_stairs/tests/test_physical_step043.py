"""043 contract and counterexample tests; tensor tests need CPU PyTorch, not Kit."""

import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from simulation.isaac_lab.g1_stairs.physical_step_config import (
    CUSTOM_COMPONENTS, PhysicalStepConfig, configure_physical_step,
    parse_physical_step_config,
)
from simulation.isaac_lab.g1_stairs import physical_step_runtime as runtime
from simulation.isaac_lab.g1_stairs.reward_runtime import active_reward_weights
from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import resolve_config

ROOT = Path(__file__).resolve().parents[4]


def smoke_config():
    return resolve_config(ROOT / "experiments/g1_stairs/configs/043a_physical_single_step_smoke.yaml")[0]


class ContractTests(unittest.TestCase):
    def test_resolved_config_is_one_step_one_iteration_and_original_actor(self):
        cfg = smoke_config()
        physical = parse_physical_step_config(cfg)
        self.assertEqual((physical.height_m, cfg["terrain"]["step_count"]), (0.06, 1))
        self.assertEqual(cfg["execution"]["maximum_iterations"], 1)
        self.assertEqual(cfg["execution"]["environments"], 32)
        self.assertEqual(cfg["execution"]["training_seed"], 42)
        self.assertEqual(cfg["policy_initialization"]["type"], "unitree_velocity_onnx_transfer")
        self.assertEqual(cfg["policy_initialization"]["stair_observations"], 485)
        self.assertFalse(cfg["observations"]["terrain_scan_in_actor"])
        self.assertFalse(cfg["observations"]["additions"]["local_terrain_height_scan"]["actor_input"])
        self.assertEqual(cfg["task"]["episode_length_s"], 20)
        self.assertEqual(cfg["ppo"]["learning_rate"], 1e-5)

    def test_043_weights_do_not_accumulate_historical_achievements(self):
        cfg = smoke_config()
        cfg.setdefault("rewards", {})["stage_plan"] = {
            "old": {"enabled_terms": {"stair_progress_raw": 100}}
        }
        weights = active_reward_weights(cfg)
        self.assertNotIn("stair_progress", weights)
        self.assertEqual({n for n in weights if n.startswith("step043_")},
                         {"step043_" + n for n in CUSTOM_COMPONENTS})

    def test_disabled_path_is_noop_without_isaac(self):
        self.assertIsNone(parse_physical_step_config({}))
        configure_physical_step(object(), {}, {})

    def test_inconsistent_or_nonfinite_contract_rejected(self):
        for key, value in (("step_count", 3), ("riser_height_m", .1), ("riser_height_range_m", [.06, .12])):
            cfg = smoke_config()
            cfg["terrain"][key] = value
            with self.assertRaises(ValueError):
                parse_physical_step_config(cfg)
        with self.assertRaises(ValueError):
            PhysicalStepConfig(support_dwell_s=math.nan).validate()

    def test_event_payout_is_literal_for_multiple_control_timesteps(self):
        for dt in (.01, .02, .04):
            env = SimpleNamespace(step_dt=dt)
            with patch.object(runtime, "snapshot", return_value={"components": {"complete": 1.0, "early_lift": .1}}):
                self.assertAlmostEqual(runtime.reward(env, {}, "complete") * 5 * dt, 5)
                self.assertAlmostEqual(runtime.reward(env, {}, "early_lift") * dt, .1 * dt)

    def test_flat_foot_placement_interval_accounts_for_collision_spheres(self):
        minimum_offset = min(p[0] for p in runtime.SOLE_CENTRES) - runtime.SOLE_RADIUS
        maximum_offset = max(p[0] for p in runtime.SOLE_CENTRES) + runtime.SOLE_RADIUS
        c = PhysicalStepConfig()
        self.assertAlmostEqual(c.first_riser_m + c.footprint_margin_m - minimum_offset, 1.06)
        self.assertAlmostEqual(c.first_riser_m + c.tread_m - c.footprint_margin_m - maximum_offset, 1.12)

    def test_configuration_adds_plain_terms_without_scene_access(self):
        class Term:
            def __init__(self, func=None, weight=0, params=None, **kw):
                self.func, self.weight, self.params = func, weight, params or {}
        managers = ModuleType("isaaclab.managers")
        managers.EventTermCfg = managers.RewardTermCfg = managers.TerminationTermCfg = Term
        cfg = smoke_config()
        weights = active_reward_weights(cfg)
        rewards = SimpleNamespace(**{name: Term() for name in weights if not name.startswith("step043_")})
        rewards.stair_progress = Term(weight=99)
        env_cfg = SimpleNamespace(rewards=rewards, events=SimpleNamespace(), terminations=SimpleNamespace())
        env_cfg.scene = SimpleNamespace(terrain=SimpleNamespace(terrain_generator=SimpleNamespace(
            sub_terrains={"three_step_ascent": SimpleNamespace(start_clearance=1., platform_length=3.)})))
        with patch.dict(sys.modules, {"isaaclab": ModuleType("isaaclab"), "isaaclab.managers": managers}):
            configure_physical_step(env_cfg, cfg, weights)
        self.assertIsNone(env_cfg.rewards.stair_progress)
        self.assertFalse(hasattr(env_cfg, "_physical_step043"))
        self.assertIs(env_cfg.terminations.step043_failure.func, runtime.failure)
        self.assertEqual(env_cfg.rewards.step043_complete.params["component"], "complete")
        self.assertEqual(env_cfg.episode_length_s, 20)


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "CPU PyTorch unavailable; run these tests in the existing AWS container before smoke")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.state = runtime.PhysicalStepState(2, "cpu", PhysicalStepConfig())
        self.tick = 0
        self.frame = dict(root=torch.tensor([[0., 0., .8], [0., 0., .8]]),
                          gravity=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
                          foot_pos=torch.tensor([[[0., -.1, .035], [0., .1, .035]]] * 2),
                          foot_quat=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]] * 2),
                          foot_velocity=torch.zeros(2, 2, 3), foot_angular_velocity=torch.zeros(2, 2, 3),
                          forces=torch.tensor([[[0., 0., 150.], [0., 0., 150.]]] * 2),
                          forbidden=torch.zeros(2, dtype=torch.bool), bodyweight=torch.full((2,), 300.))
        self.advance(7)

    def advance(self, count=1):
        events = {name: 0.0 for name in ("first_support", "base_transfer", "both_feet", "complete")}
        for _ in range(count):
            self.tick += 1
            result = self.state.update(self.tick, .02, **self.frame)
            for name in events:
                events[name] += float(result["components"][name].sum())
        return result, events

    def place_lead(self, leg=0):
        self.frame["foot_pos"][:, leg, 0] = 1.09
        self.frame["foot_pos"][:, leg, 2] = .095
        self.frame["root"][:, 0] = .90

    def test_valid_left_and_right_led_sequences_pay_each_event_once(self):
        self.frame["foot_pos"][0, 0, 0] = 1.09
        self.frame["foot_pos"][0, 0, 2] = .095
        self.frame["foot_pos"][1, 1, 0] = 1.09
        self.frame["foot_pos"][1, 1, 2] = .095
        _, events = self.advance(10)
        self.assertEqual(events["first_support"], 2)
        self.frame["root"][:, 0] = 1.05
        self.frame["root"][:, 2] = .86
        _, events = self.advance(3)
        self.assertEqual(events["base_transfer"], 2)
        self.frame["foot_pos"][:, :, 0] = 1.09
        self.frame["foot_pos"][:, :, 2] = .095
        _, events = self.advance(10)
        self.assertEqual(events["both_feet"], 2)
        self.frame["root"][:, 0] = 1.7
        self.frame["foot_pos"][:, :, 0] = 1.65
        _, events = self.advance(10)
        self.assertEqual(events["complete"], 2)
        _, events = self.advance(10)
        self.assertEqual(sum(events.values()), 0)

    def test_no_contact_or_sliding_or_overhang_or_forbidden_is_not_support(self):
        original = copy.deepcopy(self.frame)
        cases = ("unloaded", "sliding", "toe_overhang", "lateral", "heel_pivot", "forbidden", "airborne")
        for case in cases:
            with self.subTest(case=case):
                self.state = runtime.PhysicalStepState(2, "cpu", PhysicalStepConfig())
                self.frame = copy.deepcopy(original)
                self.advance(7)
                self.place_lead()
                if case == "unloaded": self.frame["forces"][:, 0, 2] = 0
                if case == "sliding": self.frame["foot_velocity"][:, 0, 0] = .3
                if case == "toe_overhang": self.frame["foot_pos"][:, 0, 0] = 1.20
                if case == "lateral": self.frame["foot_pos"][:, 0, 1] = .59
                if case == "heel_pivot": self.frame["foot_quat"][:, 0] = self.torch.tensor([math.cos(.15), 0., math.sin(.15), 0.])
                if case == "forbidden": self.frame["forbidden"][:] = True
                if case == "airborne": self.frame["foot_pos"][:, 0, 2] = .2
                _, events = self.advance(10)
                self.assertEqual(events["first_support"], 0)

    def test_contact_pulses_cannot_satisfy_dwell(self):
        self.place_lead()
        for _ in range(5):
            self.frame["forces"][:, 0, 2] = 150
            self.advance(3)
            self.frame["forces"][:, 0, 2] = 0
            self.advance(1)
        self.assertEqual(int(self.state.stage.sum()), 0)

    def test_early_kick_does_not_get_clearance_credit(self):
        self.frame["foot_pos"][:, 0, 0] = .1
        self.frame["foot_pos"][:, 0, 2] = .25
        self.frame["forces"][:, 0, 2] = 0
        result, _ = self.advance(1)
        self.assertEqual(float(result["components"]["clearance"].sum()), 0)
        self.assertGreater(float(result["components"]["early_lift"].sum()), 0)

    def test_hover_and_repeated_height_do_not_repay(self):
        self.frame["foot_pos"][:, 0, 0] = .8
        self.frame["foot_pos"][:, 0, 2] = .12
        self.frame["forces"][:, 0, 2] = 0
        first, _ = self.advance(1)
        self.assertGreater(float(first["components"]["clearance"].sum()), 0)
        second, _ = self.advance(10)
        self.assertEqual(float(second["components"]["clearance"].sum()), 0)
        self.frame["foot_pos"][:, 0, 2] = .04
        self.advance(1)
        self.frame["foot_pos"][:, 0, 2] = .12
        third, _ = self.advance(1)
        self.assertEqual(float(third["components"]["clearance"].sum()), 0)

    def test_root_crossing_without_support_never_completes(self):
        self.frame["root"][:, 0] = 1.8
        self.frame["root"][:, 2] = 1.0
        _, events = self.advance(20)
        self.assertEqual(sum(events.values()), 0)

    def test_idempotent_snapshot_and_selective_reset(self):
        self.place_lead()
        self.advance(8)
        saved = self.state.snapshot
        totals = {k: int(v) for k, v in self.state.totals.items()}
        again = self.state.update(self.tick, .02, **self.frame)
        self.assertIs(again, saved)
        self.assertEqual({k: int(v) for k, v in self.state.totals.items()}, totals)
        self.state.reset([0])
        self.assertIs(self.state.snapshot, saved)
        self.assertEqual(self.state.stage.tolist(), [0, 1])
        self.advance(10)  # Already on tread, no new lower-floor reference.
        self.assertEqual(self.state.stage.tolist(), [0, 1])

    def test_rapid_touchdown_penalty_uses_pre_contact_speed(self):
        self.place_lead()

        # Previous control step: foot is descending rapidly without contact.
        self.frame["forces"][:, 0, 2] = 0
        self.frame["foot_velocity"][:, 0, 2] = -1
        self.advance(1)

        # Contact step: current velocity is already zero. Both the touchdown
        # penalty and the diagnostic payload must still use the previous
        # -1 m/s vertical velocity.
        self.frame["forces"][:, 0, 2] = 150
        self.frame["foot_velocity"][:, 0, 2] = 0

        result, _ = self.advance(1)

        self.assertGreater(
            float(
                result[
                    "components"
                ]["touchdown"].sum()
            ),
            0,
        )

        self.assertTrue(
            self.torch.allclose(
                result[
                    "diagnostics"
                ]["precontact_velocity"][
                    :, 0, 2
                ],
                self.torch.full(
                    (2,),
                    -1.0,
                ),
            )
        )

        self.assertTrue(
            self.torch.allclose(
                result[
                    "diagnostics"
                ]["foot_velocity"][
                    :, 0, 2
                ],
                self.torch.zeros(2),
            )
        )

        result, _ = self.advance(1)

        self.assertEqual(
            float(
                result[
                    "components"
                ]["touchdown"].sum()
            ),
            0,
        )

    def test_diagnostic_payload_exposes_existing_physical_quantities(self):
        result, _ = self.advance(1)

        diagnostics = result["diagnostics"]

        expected = {
            "root",
            "foot_x_min",
            "foot_x_max",
            "sole_bottom_min",
            "sole_bottom_max",
            "foot_normal_z",
            "vertical_force",
            "bodyweight",
            "loaded",
            "slow_translation",
            "slow_rotation",
            "lateral_contained",
            "support_quality",
            "lower_geometry",
            "upper_geometry",
            "landing_rectangle",
            "stable_lower",
            "stable_upper",
            "swing_zone",
            "swing_ok",
            "clearance_value",
            "distance_to_riser_face",
            "contact",
            "contact_onset",
            "near_upper",
            "precontact_velocity",
            "foot_velocity",
            "foot_angular_velocity",
            "upright",
            "safe",
            "forbidden",
        }

        self.assertEqual(
            set(diagnostics),
            expected,
        )

        self.assertEqual(
            tuple(diagnostics["root"].shape),
            (2, 3),
        )

        for name in (
            "foot_x_min",
            "foot_x_max",
            "sole_bottom_min",
            "sole_bottom_max",
            "foot_normal_z",
            "vertical_force",
            "loaded",
            "stable_lower",
            "stable_upper",
            "distance_to_riser_face",
            "contact",
            "contact_onset",
        ):
            self.assertEqual(
                tuple(diagnostics[name].shape),
                (2, 2),
                name,
            )

        self.assertEqual(
            tuple(
                diagnostics[
                    "precontact_velocity"
                ].shape
            ),
            (2, 2, 3),
        )

        self.assertEqual(
            tuple(
                diagnostics[
                    "foot_velocity"
                ].shape
            ),
            (2, 2, 3),
        )

    def test_diagnostic_payload_is_storage_isolated_from_task_outputs(self):
        result, _ = self.advance(1)

        authoritative = {
            "failure": result["failure"].clone(),
            "success": result["success"].clone(),
            "stage": result["stage"].clone(),
            "components": {
                name: value.clone()
                for name, value
                in result["components"].items()
            },
            "measurements": {
                name: value.clone()
                for name, value
                in result["measurements"].items()
            },
        }

        for value in result["diagnostics"].values():
            if value.dtype == self.torch.bool:
                value.logical_not_()
            else:
                value.fill_(123.0)

        self.assertTrue(
            self.torch.equal(
                result["failure"],
                authoritative["failure"],
            )
        )
        self.assertTrue(
            self.torch.equal(
                result["success"],
                authoritative["success"],
            )
        )
        self.assertTrue(
            self.torch.equal(
                result["stage"],
                authoritative["stage"],
            )
        )

        for name, expected in (
            authoritative["components"].items()
        ):
            self.assertTrue(
                self.torch.equal(
                    result["components"][name],
                    expected,
                ),
                name,
            )

        for name, expected in (
            authoritative["measurements"].items()
        ):
            self.assertTrue(
                self.torch.equal(
                    result["measurements"][name],
                    expected,
                ),
                name,
            )

    def test_nonfinite_state_fails_closed(self):
        self.frame["foot_pos"][0, 0, 0] = math.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.advance(1)


if __name__ == "__main__":
    unittest.main()
