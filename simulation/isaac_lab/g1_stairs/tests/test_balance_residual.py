from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from simulation.isaac_lab.g1_stairs.balance_residual import (
    _policy_observation,
    compose_balance_residual_action,
    load_frozen_rsl_actor,
    residual_clip_vector,
    residual_scale_vector,
)
from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS


class _TensorDictLike:
    """Minimal non-dict stand-in for RSL-RL 2.3 TensorDict."""

    def __init__(self, values):
        self._values = dict(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


class BalanceResidualTests(unittest.TestCase):
    def test_policy_observation_accepts_tensordict_like_container(self):
        expected = torch.randn((4, 98))
        observations = _TensorDictLike({"policy": expected})
        self.assertIs(_policy_observation(observations), expected)

    def test_policy_observation_accepts_legacy_tuple_container(self):
        expected = torch.randn((2, 98))
        observations = (_TensorDictLike({"policy": expected}), {"extras": 1})
        self.assertIs(_policy_observation(observations), expected)

    def test_scale_map_requires_exact_joint_contract(self):
        mapping = {name: 0.1 for name in BODY_JOINTS}
        scale = residual_scale_vector(mapping, device="cpu")
        self.assertEqual(tuple(scale.shape), (29,))
        self.assertTrue(torch.allclose(scale, torch.full((29,), 0.1)))

        broken = dict(mapping)
        broken.pop(BODY_JOINTS[0])
        with self.assertRaises(ValueError):
            residual_scale_vector(broken, device="cpu")

    def test_zero_residual_exactly_preserves_teacher(self):
        teacher = torch.linspace(-0.7, 0.7, 58).reshape(2, 29)
        residual = torch.zeros_like(teacher)
        scale = torch.ones(29)
        executed = compose_balance_residual_action(
            teacher,
            residual,
            residual_scale=scale,
            residual_clip=1.0,
            executed_clip=5.0,
        )
        self.assertTrue(torch.equal(executed, teacher))

    def test_residual_is_clipped_before_joint_scaling(self):
        teacher = torch.zeros((1, 29))
        residual = torch.full((1, 29), 10.0)
        scale = torch.linspace(0.0, 1.4, 29)
        executed = compose_balance_residual_action(
            teacher,
            residual,
            residual_scale=scale,
            residual_clip=1.0,
            executed_clip=5.0,
        )
        self.assertTrue(torch.allclose(executed[0], scale))

    def test_per_joint_residual_clip(self):
        """Patch 64: knee and hip pitch get a wider clip, the rest keep the base."""
        control = {"residual_clip": 2.0, "residual_clip_by_joint": {
            "left_knee_joint": 3.0, "right_knee_joint": 3.0,
            "left_hip_pitch_joint": 3.0, "right_hip_pitch_joint": 3.0}}
        clip = residual_clip_vector(control, device="cpu")
        wide = [BODY_JOINTS.index(n) for n in control["residual_clip_by_joint"]]
        self.assertEqual(clip.shape, (29,))
        self.assertEqual(sorted(i for i in range(29) if clip[i] == 3.0), sorted(wide))
        self.assertEqual(float(clip.min()), 2.0)
        teacher = torch.zeros((2, 29))
        residual = torch.stack([torch.full((29,), 10.0), torch.full((29,), -10.0)])
        executed = compose_balance_residual_action(
            teacher, residual, residual_scale=torch.ones(29), residual_clip=clip, executed_clip=8.0)
        self.assertTrue(torch.equal(executed[0], clip))
        self.assertTrue(torch.equal(executed[1], -clip))
        # Without overrides: the scalar everywhere (old behaviour).
        self.assertTrue(torch.equal(residual_clip_vector({"residual_clip": 2.0}, device="cpu"),
                                    torch.full((29,), 2.0)))

    def test_per_joint_residual_clip_is_validated(self):
        for bad in ({"left_knee_joint": 3.0},                              # right partner differs
                    {"left_knee_joint": 3.0, "right_knee_joint": 2.5},
                    {"no_such_joint": 3.0},
                    {"left_knee_joint": 0.0, "right_knee_joint": 0.0}):
            with self.assertRaises(ValueError):
                residual_clip_vector({"residual_clip": 2.0, "residual_clip_by_joint": bad}, device="cpu")
        with self.assertRaises(ValueError):
            compose_balance_residual_action(
                torch.zeros((1, 29)), torch.zeros((1, 29)), residual_scale=torch.ones(29),
                residual_clip=torch.ones(5), executed_clip=5.0)

    def test_executed_action_has_final_safety_clip(self):
        teacher = torch.full((1, 29), 4.9)
        residual = torch.ones((1, 29))
        scale = torch.ones(29)
        executed = compose_balance_residual_action(
            teacher,
            residual,
            residual_scale=scale,
            residual_clip=1.0,
            executed_clip=5.0,
        )
        self.assertEqual(float(executed.max()), 5.0)

    def test_load_frozen_actor_reconstructs_checkpoint_network(self):
        actor = torch.nn.Sequential(
            torch.nn.Linear(98, 16),
            torch.nn.ELU(),
            torch.nn.Linear(16, 29),
        )
        state = {
            f"actor.{key}": value.detach().clone()
            for key, value in actor.state_dict().items()
        }
        state["critic.0.weight"] = torch.zeros((8, 8))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"model_state_dict": state, "iter": 1500}, path)
            frozen, metadata = load_frozen_rsl_actor(path, device="cpu")

            x = torch.randn((4, 98))
            with torch.no_grad():
                expected = actor(x)
                actual = frozen(x)

            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertEqual(metadata["observation_dim"], 98)
            self.assertEqual(metadata["action_dim"], 29)
            self.assertEqual(metadata["source_iteration"], 1500)
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))


class MirrorAveragedTeacherTests(unittest.TestCase):
    def test_average_is_equivariant_and_keeps_symmetric_teachers(self):
        from simulation.isaac_lab.g1_stairs import footstep_symmetry as fs
        from simulation.isaac_lab.g1_stairs.balance_residual import MirrorAveragedTeacher
        torch.manual_seed(3)
        lopsided = torch.nn.Linear(480, 29)
        sym = MirrorAveragedTeacher(lopsided)
        x = torch.randn(5, 480)
        with torch.no_grad():
            self.assertTrue(torch.allclose(sym(fs.mirror_teacher_obs(x)), fs.mirror_actions(sym(x)), atol=1e-6))
            self.assertFalse(torch.allclose(lopsided(fs.mirror_teacher_obs(x)), fs.mirror_actions(lopsided(x)), atol=1e-3))
            # an already equivariant teacher is returned unchanged
            self.assertTrue(torch.allclose(MirrorAveragedTeacher(sym)(x), sym(x), atol=1e-6))


if __name__ == "__main__":
    unittest.main()


class BalanceResidualConfigTests(unittest.TestCase):
    def test_v01_config_resolves_teacher_and_zero_residual_actor(self):
        from simulation.isaac_lab.g1_stairs.scripts.training.prepare_experiment import (
            REPO_ROOT,
            resolve_config,
        )

        resolved, _ = resolve_config(
            REPO_ROOT
            / "experiments/g1_stairs/configs/g1_stair_v10.yaml"
        )
        self.assertEqual(
            resolved["policy_initialization"]["type"],
            "zero_residual_actor",
        )
        cfg = resolved["balance_residual_control"]
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["teacher_checkpoint"].endswith("model_1500.pt"))
        self.assertEqual(set(cfg["scale_by_joint"]), set(BODY_JOINTS))
        self.assertEqual(resolved["terrain"]["riser_height_m"], 0.06)
        self.assertEqual(resolved["terrain"]["step_count"], 1)
        self.assertEqual(
            resolved["observations"]["proprioception_history_steps"],
            1,
        )


class UnitreeOnnxTeacherTests(unittest.TestCase):
    def _write_onnx(self, directory: Path) -> Path:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        from simulation.isaac_lab.g1_stairs.policy_initialization import (
            EXPECTED_ONNX_SHAPES,
        )

        rng = np.random.default_rng(0)
        initializers = [
            numpy_helper.from_array(
                rng.standard_normal(shape).astype(np.float32) * 0.05, name
            )
            for name, shape in EXPECTED_ONNX_SHAPES.items()
        ]
        graph = helper.make_graph([], "unitree_actor", [], [], initializers)
        path = directory / "policy.onnx"
        onnx.save(helper.make_model(graph), str(path))
        return path

    def test_teacher_reads_only_the_480_channel_prefix(self):
        from simulation.isaac_lab.g1_stairs.balance_residual import (
            load_frozen_unitree_onnx_teacher, teacher_observation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_onnx(Path(tmp))
            teacher, meta = load_frozen_unitree_onnx_teacher(
                path, device="cpu", observation_dim=480
            )
        self.assertEqual(meta["observation_dim"], 480)
        self.assertEqual(meta["teacher_type"], "unitree_onnx")
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))

        obs = torch.randn(3, 688)
        changed = obs.clone()
        changed[:, 480:] = torch.randn(3, 208)   # height scan / footsteps etc.
        a = teacher(teacher_observation(obs, 480, prefix=True))
        b = teacher(teacher_observation(changed, 480, prefix=True))
        self.assertEqual(tuple(a.shape), (3, len(BODY_JOINTS)))
        self.assertTrue(torch.equal(a, b))

    def test_prefix_and_exact_dimension_checks(self):
        from simulation.isaac_lab.g1_stairs.balance_residual import teacher_observation

        with self.assertRaises(RuntimeError):
            teacher_observation(torch.zeros(2, 400), 480, prefix=True)
        with self.assertRaises(RuntimeError):
            teacher_observation(torch.zeros(2, 688), 480, prefix=False)
        self.assertEqual(tuple(teacher_observation(torch.zeros(2, 480), 480,
                                                   prefix=False).shape), (2, 480))

    def test_onnx_teacher_rejects_wrong_observation_dim(self):
        from simulation.isaac_lab.g1_stairs.balance_residual import (
            load_frozen_unitree_onnx_teacher,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_onnx(Path(tmp))
            with self.assertRaises(ValueError):
                load_frozen_unitree_onnx_teacher(path, device="cpu", observation_dim=688)


def test_residual_gate_weight_on_stairs_fades_after_climb():
    from simulation.isaac_lab.g1_stairs.balance_residual import residual_gate_weight

    prev = torch.tensor([1.0, 1.0, 0.0, 0.5, 0.3])
    active = torch.tensor([True, False, False, False, True])
    reset = torch.tensor([False, False, True, True, False])
    w = residual_gate_weight(prev, active, reset, dt=0.02, fade_s=0.8)
    torch.testing.assert_close(w, torch.tensor([1.0, 0.975, 0.0, 0.0, 1.0]))
    for _ in range(60):
        w = residual_gate_weight(w, torch.zeros(5, dtype=torch.bool), torch.zeros(5, dtype=torch.bool), 0.02, 0.8)
    assert bool((w == 0.0).all())


def test_residual_gate_weight_rejects_bad_fade():
    import pytest
    from simulation.isaac_lab.g1_stairs.balance_residual import residual_gate_weight

    with pytest.raises(ValueError):
        residual_gate_weight(torch.ones(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), 0.02, 0.0)


def test_gated_residual_off_on_flat_tiles_and_after_the_climb(monkeypatch):
    from types import SimpleNamespace
    from simulation.isaac_lab.g1_stairs import terrain_state
    from simulation.isaac_lab.g1_stairs.balance_residual import FrozenBalanceResidualWrapper

    env = SimpleNamespace(step_dt=0.02, episode_length_buf=torch.tensor([0, 0, 5]),
                          _g1_rolling_clock=SimpleNamespace(climbed=torch.tensor([False, False, True])))
    monkeypatch.setattr(terrain_state, "riser_height_per_env", lambda e: torch.tensor([0.0, 0.2, 0.2]))
    w = FrozenBalanceResidualWrapper.__new__(FrozenBalanceResidualWrapper)
    w._wrapped = SimpleNamespace(unwrapped=env)
    w._gate, w._gate_fade_s, w._gate_weight = "stair_tiles", 0.8, torch.ones(3)
    w._residual_clip = torch.tensor(2.0)
    out = w._gated(torch.full((3, len(BODY_JOINTS)), 3.0))
    # flat tile (reset): 0; stair tile, not climbed: clipped residual; climbed: fading
    torch.testing.assert_close(out[:, 0], torch.tensor([0.0, 2.0, 2.0 * 0.975]))


def test_ramp_forward_command():
    import pytest
    from simulation.isaac_lab.g1_stairs.command_runtime import ramp_forward_command

    cmd = torch.full((3, 3), 9.0)
    ramp_forward_command(cmd, 1.0, torch.tensor([0.0, 0.5, 2.0]), 1.0)
    torch.testing.assert_close(cmd[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert bool((cmd[:, 1:] == 9.0).all())
    with pytest.raises(ValueError):
        ramp_forward_command(cmd, 1.0, torch.zeros(3), 0.0)


def test_start_raw_mask_first_seconds_of_a_move():
    import pytest
    from simulation.isaac_lab.g1_stairs.balance_residual import start_raw_mask

    t = torch.zeros(3)
    moving = torch.tensor([True, True, False])
    reset = torch.tensor([True, False, False])
    t, raw = start_raw_mask(t, moving, reset, 0.02, 0.3)
    assert raw.tolist() == [True, True, False]
    torch.testing.assert_close(t, torch.tensor([0.0, 0.02, 0.0]))
    for _ in range(20):
        t, raw = start_raw_mask(t, moving, torch.zeros(3, dtype=torch.bool), 0.02, 0.3)
    assert raw.tolist() == [False, False, False]
    # standing again restarts the count
    t, raw = start_raw_mask(t, torch.tensor([False, True, False]), torch.zeros(3, dtype=torch.bool), 0.02, 0.3)
    t, raw = start_raw_mask(t, torch.tensor([True, True, False]), torch.zeros(3, dtype=torch.bool), 0.02, 0.3)
    assert raw.tolist() == [True, False, False]
    with pytest.raises(ValueError):
        start_raw_mask(t, moving, reset, 0.02, 0.0)


def test_raw_start_only_where_the_teacher_walks_alone():
    from types import SimpleNamespace
    from simulation.isaac_lab.g1_stairs.balance_residual import FrozenBalanceResidualWrapper

    cmd = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(step_dt=0.02, episode_length_buf=torch.tensor([0, 0, 0]),
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    w = FrozenBalanceResidualWrapper.__new__(FrozenBalanceResidualWrapper)
    w._wrapped = SimpleNamespace(unwrapped=env)
    w._start_raw_s, w._moving_s, w._start_until_step = 0.3, None, False
    import simulation.isaac_lab.g1_stairs.footstep_symmetry as fs
    fs_mirror_obs, fs_mirror_act = fs.mirror_teacher_obs, fs.mirror_actions
    fs.mirror_teacher_obs, fs.mirror_actions = (lambda o: o), (lambda a: a)
    w._gate_weight = torch.tensor([0.0, 1.0, 0.0])
    w._raw_teacher = lambda obs: torch.ones(3, 29)
    out = w._raw_start(torch.zeros(3, 480), torch.zeros(3, 29))
    fs.mirror_teacher_obs, fs.mirror_actions = fs_mirror_obs, fs_mirror_act
    # env 0: flat, moving -> raw; env 1: residual on (stairs) -> averaged; env 2: standing -> averaged
    assert out[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_start_side_redrawn_only_at_the_start_of_a_move():
    from simulation.isaac_lab.g1_stairs.balance_residual import start_side_draw
    side = torch.tensor([0, 1, 1, 0])
    moving_s = torch.tensor([0.02, 0.10, 0.0, 0.02])   # new move, ongoing, standing, new move
    out = start_side_draw(side, moving_s, 0.02, torch.tensor([1, 0, 0, 0]))
    assert out.tolist() == [1, 1, 1, 0]


def test_walk_in_gate_starts_off_fades_in_and_out():
    from simulation.isaac_lab.g1_stairs.balance_residual import residual_gate_walk_in_weight
    on = torch.ones(2, dtype=torch.bool); off = torch.zeros(2, dtype=torch.bool)
    w = residual_gate_walk_in_weight(torch.ones(2), off, on, 0.02, 0.4, 0.8)
    assert torch.equal(w, torch.zeros(2))                 # reset: teacher alone
    for _ in range(10):
        w = residual_gate_walk_in_weight(w, on, off, 0.02, 0.4, 0.8)
    assert torch.allclose(w, torch.full((2,), 0.5))       # half way after 0.2 s
    for _ in range(20):
        w = residual_gate_walk_in_weight(w, on, off, 0.02, 0.4, 0.8)
    assert torch.equal(w, torch.ones(2))                  # capped at 1
    for _ in range(20):
        w = residual_gate_walk_in_weight(w, off, off, 0.02, 0.4, 0.8)
    assert torch.allclose(w, torch.full((2,), 0.5))       # out over 0.8 s


def test_walk_in_gate_rejects_bad_fades():
    import pytest
    from simulation.isaac_lab.g1_stairs.balance_residual import residual_gate_walk_in_weight
    z = torch.zeros(1, dtype=torch.bool)
    with pytest.raises(ValueError):
        residual_gate_walk_in_weight(torch.zeros(1), z, z, 0.02, 0.0, 0.8)


def test_ramp_forward_command_with_stop():
    from simulation.isaac_lab.g1_stairs.command_runtime import ramp_forward_command
    cmd = torch.zeros(4, 3)
    t = torch.tensor([0.5, 2.0, 4.25, 5.0])
    ramp_forward_command(cmd, 0.4, t, 1.0, stop_s=4.0, stop_ramp_s=0.5)
    torch.testing.assert_close(cmd[:, 0], torch.tensor([0.2, 0.4, 0.2, 0.0]))


def test_start_step_done_needs_a_real_lift_and_landing():
    from simulation.isaac_lab.g1_stairs.balance_residual import start_step_done
    dt = 0.02
    unloaded = torch.zeros(3, 2)
    done = torch.zeros(3, dtype=torch.bool)
    # robot 0: left foot lifts 0.2 s then lands; robot 1: a 0.04 s weight shift;
    # robot 2: unloaded only during the first 0.1 s (spawn settling), then lands
    for k in range(1, 40):
        t = k * dt
        sup = torch.ones(3, 2, dtype=torch.bool)
        if 0.3 <= t < 0.5: sup[0, 0] = False
        if 0.3 <= t < 0.34: sup[1, 1] = False
        if t < 0.1: sup[2, 0] = False
        moving_s = torch.full((3,), t)
        unloaded, done = start_step_done(unloaded, done, moving_s, sup, dt)
    assert done.tolist() == [True, False, False]
    # a new move clears it
    unloaded, done = start_step_done(unloaded, done, torch.full((3,), dt), torch.ones(3, 2, dtype=torch.bool), dt)
    assert done.tolist() == [False, False, False]
