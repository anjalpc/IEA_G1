"""Replay-only visual markers: footstep targets and foot contact force bars.

Draws into the rendered video (Isaac Lab VisualizationMarkers, no physics):
- the assigned target (bright) and the next target (faint) as foot-sized
  pads, blue for the left foot and orange for the right;
- one vertical bar per loaded foot, height proportional to the vertical
  contact force (0.3 m per body weight), green below 1.2 BW, yellow below
  2 BW, red above.
Reads the same target state as the policy observation
(footstep_phase_observation.planned_footsteps_with_phase); never mutates it.
"""
from __future__ import annotations


import torch

from .contact_sensor_contract import FOOT_BODY_NAMES
from .footstep_isaac_adapter import course_tensors
from .footstep_live_progress import ensure_coordinator, ensure_rolling_clock
from .footstep_tensor import NOMINAL_FOOT_SEPARATION_M

BODY_WEIGHT_N = 35.0 * 9.81
BAR_M_PER_BW = 0.3
BAR_MIN_N = 30.0                 # same support threshold as the task
PAD_SIZE = (0.20, 0.08, 0.006)
LEFT_RGB, RIGHT_RGB = (0.10, 0.45, 1.00), (1.00, 0.55, 0.10)


def force_band(force_n: torch.Tensor) -> torch.Tensor:
    """[K] marker index per bar: 0 green (< 1.2 BW), 1 yellow (< 2 BW), 2 red."""
    bw = force_n / BODY_WEIGHT_N
    return (bw >= 1.2).long() + (bw >= 2.0).long()


def target_marker_index(foot: torch.Tensor, slot: int) -> torch.Tensor:
    """[K] 0 left-now, 1 right-now, 2 left-next, 3 right-next."""
    return foot.long() + 2 * slot


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """[K,4] (w, x, y, z) rotation about z."""
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


class FootstepMarkers:
    def __init__(self, env):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        def pad(rgb, opacity):
            return sim_utils.CuboidCfg(
                size=PAD_SIZE,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, opacity=opacity))

        def bar(rgb):
            return sim_utils.CylinderCfg(
                radius=0.025, height=1.0, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, emissive_color=rgb))

        self._targets = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/FootstepTargets",
            markers={"left_now": pad(LEFT_RGB, 1.0), "right_now": pad(RIGHT_RGB, 1.0),
                     "left_next": pad(LEFT_RGB, 0.35), "right_next": pad(RIGHT_RGB, 0.35)}))
        self._bars = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/FootForces",
            markers={"low": bar((0.1, 0.8, 0.2)), "mid": bar((1.0, 0.85, 0.1)),
                     "high": bar((0.95, 0.1, 0.1))}))
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        self._foot_ids = [list(robot.body_names).index(n) for n in FOOT_BODY_NAMES]
        self._sensor_ids = [list(sensor.body_names).index(n) for n in FOOT_BODY_NAMES]

    def _target_state(self, env):
        coord = ensure_coordinator(env)
        flat = bool(getattr(env.cfg, "flat_ground_footsteps", False))
        course = course_tensors(env, foot_separation_m=NOMINAL_FOOT_SEPARATION_M, flat_ground=flat)
        plan, feet, valid = coord.runtime.targets(**course, flat_ground=flat)
        plan, feet, valid = plan.clone(), feet.clone(), valid.clone()
        state = ensure_rolling_clock(env)
        rows = getattr(env, "_g1_rolling_rows", None)
        if state is not None and rows is not None:
            plan[rows] = state.targets[rows]
            feet[rows] = state.feet[rows]
            valid[rows] = state.active[rows, None]
        return plan, feet, valid

    def update(self, env):
        # Footstep targets of env 0 (replay uses one env), course frame -> world.
        plan, feet, valid = self._target_state(env)
        origin = env.scene.env_origins[0]
        pos, quat, idx = [], [], []
        for slot in (0, 1):
            if bool(valid[0, slot]):
                p = plan[0, slot, :3] + origin
                p[2] = p[2] + 0.5 * PAD_SIZE[2] + 0.002
                pos.append(p)
                quat.append(yaw_quat(plan[0, slot, 3:4])[0])
                idx.append(int(target_marker_index(feet[0, slot:slot + 1], slot)[0]))
        if pos:
            self._targets.set_visibility(True)
            self._targets.visualize(torch.stack(pos), torch.stack(quat), marker_indices=idx)
        else:
            self._targets.set_visibility(False)

        # Contact force bars at loaded feet.
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        force = sensor.data.net_forces_w[0, self._sensor_ids, 2]
        feet_pos = robot.data.body_pos_w[0, self._foot_ids]
        loaded = force >= BAR_MIN_N
        if bool(loaded.any()):
            f = force[loaded]
            height = BAR_M_PER_BW * f / BODY_WEIGHT_N
            centre = feet_pos[loaded].clone()
            centre[:, 2] = centre[:, 2] + 0.5 * height
            scale = torch.stack((torch.ones_like(height), torch.ones_like(height), height), dim=-1)
            self._bars.set_visibility(True)
            self._bars.visualize(centre, None, scale, marker_indices=force_band(f).tolist())
        else:
            self._bars.set_visibility(False)
