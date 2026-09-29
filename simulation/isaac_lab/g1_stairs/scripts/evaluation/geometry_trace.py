
"""Read-only state sampling after reward computation and before reset."""
import json
from pathlib import Path


class GeometryTrace:
    def __init__(self, env, output):
        from simulation.isaac_lab.g1_stairs.contracts import (
            G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M,
        )
        from simulation.isaac_lab.g1_stairs.terrain_state import (
            stair_cfg, riser_height_per_env,
        )
        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.manager = env.reward_manager
        self.original_compute = self.manager.compute
        self.feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_ids = [self.robot.body_names.index(n) for n in self.feet]
        cfg = stair_cfg(env)
        metadata = {
            "schema_version": 1,
            "sampling": "after reward computation, before automatic reset",
            "step_dt_s": float(env.step_dt),
            "physics_dt_s": float(env.physics_dt),
            "quaternion_order": "wxyz",
            "positions": "world coordinates; subtract env_origin_w for course coordinates",
            "foot_names": self.feet,
            "joint_names": list(self.robot.joint_names),
            "contact_body_names": list(self.sensor.body_names),
            "reward_names": list(self.manager.active_terms),
            "sole_sphere_centres_body_m": G1_SOLE_SPHERE_CENTRES_M,
            "sole_sphere_radius_m": G1_SOLE_SPHERE_RADIUS_M,
            "first_riser_x_local_m": float(cfg.start_clearance),
            "tread_depth_m": float(cfg.tread_depth),
            "step_count": int(cfg.step_count),
            "riser_height_m": float(riser_height_per_env(env)[0].item()),
            "reward_contributions": "weighted term values multiplied by step_dt",
        }
        output = Path(output)
        with (output / "trace_metadata.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
        self.stream = (output / "trace.jsonl").open("x", buffering=1)
        self.capture(None)
        self.manager.compute = self.compute

    @staticmethod
    def values(tensor):
        return tensor.detach().cpu().tolist()

    def capture(self, reward):
        env = self.env
        data = self.robot.data
        row = {
            "sample": "initial" if reward is None else "post_physics_pre_reset",
            "global_step": int(env.common_step_counter),
            "episode_step": int(env.episode_length_buf[0].item()),
            "episode_time_s": float(env.episode_length_buf[0].item() * env.step_dt),
            "env_origin_w": self.values(env.scene.env_origins[0]),
            "root_pos_w": self.values(data.root_pos_w[0]),
            "root_quat_w": self.values(data.root_quat_w[0]),
            "root_lin_vel_w": self.values(data.root_lin_vel_w[0]),
            "root_ang_vel_w": self.values(data.root_ang_vel_w[0]),
            "command_base_velocity": self.values(
                env.command_manager.get_command("base_velocity")[0]
            ),
            "foot_pos_w": self.values(data.body_pos_w[0, self.foot_ids]),
            "foot_quat_w": self.values(data.body_quat_w[0, self.foot_ids]),
            "foot_lin_vel_w": self.values(data.body_lin_vel_w[0, self.foot_ids]),
            "foot_ang_vel_w": self.values(data.body_ang_vel_w[0, self.foot_ids]),
            "contact_net_forces_w": self.values(self.sensor.data.net_forces_w[0]),
            "joint_pos_rad": self.values(data.joint_pos[0]),
            "joint_vel_rad_s": self.values(data.joint_vel[0]),
            "applied_torque_nm": self.values(data.applied_torque[0]),
            "joint_pos_target_rad": self.values(data.joint_pos_target[0]),
            "action": self.values(env.action_manager.action[0]),
        }
        if reward is not None:
            row.update({
                "terminated": bool(env.termination_manager.terminated[0].item()),
                "time_out": bool(env.termination_manager.time_outs[0].item()),
                "reward_total": float(reward[0].item()),
                "reward_contributions": self.values(
                    self.manager._step_reward[0] * env.step_dt
                ),
            })
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def compute(self, dt):
        reward = self.original_compute(dt=dt)
        self.capture(reward)
        return reward

    def close(self):
        self.manager.compute = self.original_compute
        self.stream.close()
