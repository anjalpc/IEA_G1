"""Left/right mirror symmetry for the footstep task (RSL-RL symmetry_cfg).

Used as ``data_augmentation_func`` for rsl-rl-lib 3.0.1 PPO: every training
batch is extended by its left/right mirror image (observations and residual
actions), so a gait and its mirror are learned as the same skill (Mittal et
al. 2024). Run 20261102_112206 model_6200 walked with a straight left and a
bent right stance knee (0.14 vs 0.44 rad; Unitree 0.26/0.25).

Mirror = reflection in the robot's sagittal (x-z) plane:
- vectors (linear velocity, gravity): y flips;
- angular velocity (pseudo-vector): x and z flip;
- command (vx, vy, wz): vy and wz flip;
- joints: left and right swap; roll and yaw joints also flip sign (the G1
  default pose satisfies this: shoulder/wrist roll +-0.25/+-0.15, the rest
  equal or zero);
- height scan (17 x 11 grid, ordering "xy": index = iy * 17 + ix): y rows
  reversed;
- stair_geometry: relative stair yaw flips;
- footstep targets (2 x [valid, left, right, x, y, z, yaw]): left/right flags
  swap, y and yaw flip;
- gait clock (sin, cos of the phase): the mirror is the phase half a cycle
  later (the other foot's turn), i.e. both flip.

Layouts follow env_cfg.StairObservationsCfg (term order, five-frame history
for proprioception, flattened per term) and are checked by width; any
change to the observation terms must be reflected here.
"""
from __future__ import annotations

import torch

from .contracts import BODY_JOINTS

__all__ = [
    "JOINT_PERMUTATION", "JOINT_SIGN", "POLICY_WIDTH", "CRITIC_WIDTH", "TEACHER_WIDTH",
    "mirror_policy_obs", "mirror_critic_obs", "mirror_teacher_obs", "mirror_actions",
    "compute_symmetric_states",
]

HISTORY = 5
NUM_JOINTS = len(BODY_JOINTS)
SCAN_NX, SCAN_NY = 17, 11            # GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
STAIR_GEOMETRY_WIDTH = 5
STAIR_YAW_INDEX = 3
FOOTSTEP_SLOT_WIDTH = 7
FOOTSTEP_WIDTH = 2 * FOOTSTEP_SLOT_WIDTH + 2   # two targets + clock (sin, cos)


def _joint_maps():
    names = list(BODY_JOINTS)
    perm, sign = [], []
    for name in names:
        if name.startswith("left_"):
            partner = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            partner = "left_" + name[len("right_"):]
        else:
            partner = name
        if partner not in names:
            raise ValueError(f"no mirror partner for joint {name}")
        perm.append(names.index(partner))
        sign.append(-1.0 if ("_roll_" in name or "_yaw_" in name) else 1.0)
    return perm, sign


_PERM, _SIGN = _joint_maps()
JOINT_PERMUTATION = tuple(_PERM)
JOINT_SIGN = tuple(_SIGN)

# Per-term mirror for one history frame: (width, index permutation, sign).
_VEC = ((0, 1, 2), (1.0, -1.0, 1.0))           # linear velocity, gravity
_ANG = ((0, 1, 2), (-1.0, 1.0, -1.0))          # angular velocity
_CMD = ((0, 1, 2), (1.0, -1.0, -1.0))          # vx, vy, wz
_JNT = (JOINT_PERMUTATION, JOINT_SIGN)


def _scan_map():
    perm = []
    for iy in range(SCAN_NY):
        for ix in range(SCAN_NX):
            perm.append((SCAN_NY - 1 - iy) * SCAN_NX + ix)
    return tuple(perm), (1.0,) * (SCAN_NX * SCAN_NY)


def _stair_map():
    sign = [1.0] * STAIR_GEOMETRY_WIDTH
    sign[STAIR_YAW_INDEX] = -1.0
    return tuple(range(STAIR_GEOMETRY_WIDTH)), tuple(sign)


def _footstep_map():
    perm, sign = [], []
    for slot in range(2):
        base = slot * FOOTSTEP_SLOT_WIDTH
        # valid, left<-right, right<-left, x, y, z, yaw
        perm += [base, base + 2, base + 1, base + 3, base + 4, base + 5, base + 6]
        sign += [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
    perm += [2 * FOOTSTEP_SLOT_WIDTH, 2 * FOOTSTEP_SLOT_WIDTH + 1]
    sign += [-1.0, -1.0]
    return tuple(perm), tuple(sign)


_PROPRIO = (_ANG, _VEC, _CMD, _JNT, _JNT, _JNT)   # ang vel, gravity, cmd, q, dq, last action
_TERRAIN = (_scan_map(), _stair_map(), _footstep_map())


def _build(term_maps_with_history):
    perm, sign, offset = [], [], 0
    for (term_perm, term_sign), history in term_maps_with_history:
        width = len(term_perm)
        for h in range(history):
            base = offset + h * width
            perm += [base + p for p in term_perm]
            sign += list(term_sign)
        offset += width * history
    return torch.tensor(perm, dtype=torch.long), torch.tensor(sign)


_POLICY_PERM, _POLICY_SIGN = _build(
    [(m, HISTORY) for m in _PROPRIO] + [(m, 1) for m in _TERRAIN])
_CRITIC_PERM, _CRITIC_SIGN = _build(
    [(_VEC, HISTORY)] + [(m, HISTORY) for m in _PROPRIO] + [(m, 1) for m in _TERRAIN])
_TEACHER_PERM, _TEACHER_SIGN = _build([(m, HISTORY) for m in _PROPRIO])
POLICY_WIDTH = int(_POLICY_PERM.numel())     # 688
TEACHER_WIDTH = int(_TEACHER_PERM.numel())   # 480 (proprio prefix the Unitree teacher reads)
CRITIC_WIDTH = int(_CRITIC_PERM.numel())     # 703
_ACTION_PERM = torch.tensor(JOINT_PERMUTATION, dtype=torch.long)
_ACTION_SIGN = torch.tensor(JOINT_SIGN)


def _apply(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor, name: str):
    if x.shape[-1] != perm.numel():
        raise ValueError(f"{name}: expected width {perm.numel()}, got {x.shape[-1]}")
    return x[..., perm.to(x.device)] * sign.to(device=x.device, dtype=x.dtype)


def mirror_policy_obs(x: torch.Tensor) -> torch.Tensor:
    return _apply(x, _POLICY_PERM, _POLICY_SIGN, "policy observation")


def mirror_teacher_obs(x: torch.Tensor) -> torch.Tensor:
    return _apply(x, _TEACHER_PERM, _TEACHER_SIGN, "teacher observation")


def mirror_critic_obs(x: torch.Tensor) -> torch.Tensor:
    return _apply(x, _CRITIC_PERM, _CRITIC_SIGN, "critic observation")


def mirror_actions(a: torch.Tensor) -> torch.Tensor:
    return _apply(a, _ACTION_PERM, _ACTION_SIGN, "actions")


_GROUP_MIRROR = {"policy": mirror_policy_obs, "critic": mirror_critic_obs}


@torch.no_grad()
def compute_symmetric_states(env=None, obs=None, actions=None):
    """RSL-RL data_augmentation_func: returns (obs, actions) with the mirror
    appended after the original along the batch dimension (original first,
    as rsl-rl's mirror loss assumes). Unknown observation groups are an
    error, so a new group cannot silently go unmirrored."""
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            if key not in _GROUP_MIRROR:
                raise KeyError(f"no mirror defined for observation group '{key}'")
            obs_aug[key][n:] = _GROUP_MIRROR[key](obs[key])
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug
