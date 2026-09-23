from pathlib import Path
import numpy as np


ROOT = Path("/workspace/IEA_G1")

SOURCE = (
    ROOT
    / "experiments/g1_stairs/reference_data/"
      "stair_ascent_reference_50hz.npz"
)

OUTPUT = (
    ROOT
    / "port/pretrained_controller/complete_task/reference/"
      "complete_task_reference_50hz.npz"
)

#
# Native reference:
#
# right foot is on our third tread approximately frames 240..278.
# Stop before the source motion starts using the next staircase.
#
STAIR_END = 278

#
# Find a repeatable walking cycle from the ground-approach portion.
# This stays well before stair interaction.
#
SEARCH_START = 55
SEARCH_END = 135

#
# Need enough continuation for > 2 m after the platform.
# 650 total frames = 13 s at 50 Hz.
#
TOTAL_FRAMES = 650

data = np.load(SOURCE)

q = data["dof_pos"].astype(np.float32)
dq = data["dof_vel"].astype(np.float32)
body_pos = data["body_pos_w"].astype(np.float32)
body_quat = data["body_quat_w"].astype(np.float32)

if q.shape != (499, 29):
    raise RuntimeError(f"Unexpected q shape: {q.shape}")

#
# Automatically locate a reasonably periodic pre-stair walking segment.
# Endpoint cost uses both q and dq.
#
best = None

for i in range(SEARCH_START, SEARCH_END):
    for j in range(i + 25, min(i + 80, SEARCH_END + 1)):
        q_cost = np.sqrt(np.mean((q[i] - q[j]) ** 2))
        dq_cost = np.sqrt(np.mean((dq[i] - dq[j]) ** 2))

        cost = q_cost + 0.05 * dq_cost

        if best is None or cost < best[0]:
            best = (cost, i, j, q_cost, dq_cost)

_, loop_start, loop_end, q_cost, dq_cost = best

#
# Include start, exclude duplicate matching endpoint.
#
loop_q = q[loop_start:loop_end].copy()
loop_dq = dq[loop_start:loop_end].copy()

loop_len = loop_q.shape[0]

print("=" * 80)
print("COMPLETE TASK REFERENCE BUILDER")
print("=" * 80)
print(f"source       : {SOURCE}")
print(f"stair end    : {STAIR_END}")
print(
    f"walk cycle   : {loop_start}..{loop_end - 1} "
    f"({loop_len} frames)"
)
print(f"endpoint q RMS  : {q_cost:.6f}")
print(f"endpoint dq RMS : {dq_cost:.6f}")

#
# Preserve the genuine stair motion through the third-step region.
#
out_q = [q[: STAIR_END + 1]]
out_dq = [dq[: STAIR_END + 1]]
out_pos = [body_pos[: STAIR_END + 1]]
out_quat = [body_quat[: STAIR_END + 1]]

remaining = TOTAL_FRAMES - (STAIR_END + 1)

#
# Desired pelvis height on the 0.60 m platform:
# initial reference pelvis height + platform elevation.
#
platform_root_z = float(body_pos[0, 0, 2] + 0.60)

#
# Hold final reference heading during platform walking.
#
continuation_quat = body_quat[STAIR_END].copy()

#
# We only need body root Z and root orientation downstream for the
# tokenizer. Other body positions are retained as harmless placeholders.
#
last_body = body_pos[STAIR_END].copy()

continuation_pos = np.repeat(
    last_body[None, :, :],
    remaining,
    axis=0,
)

continuation_quat_array = np.repeat(
    continuation_quat[None, :, :],
    remaining,
    axis=0,
)

#
# Smoothly bring desired root height from stair-end value to the
# correct 0.60 m platform height over 25 frames.
#
z0 = float(body_pos[STAIR_END, 0, 2])
blend_frames = min(25, remaining)

if blend_frames > 0:
    continuation_pos[:blend_frames, 0, 2] = np.linspace(
        z0,
        platform_root_z,
        blend_frames,
        dtype=np.float32,
    )

if remaining > blend_frames:
    continuation_pos[blend_frames:, 0, 2] = platform_root_z

#
# Repeat the successful approach gait for platform locomotion.
#
cycles_q = []
cycles_dq = []

while sum(x.shape[0] for x in cycles_q) < remaining:
    cycles_q.append(loop_q)
    cycles_dq.append(loop_dq)

continuation_q = np.concatenate(cycles_q, axis=0)[:remaining]
continuation_dq = np.concatenate(cycles_dq, axis=0)[:remaining]

out_q.append(continuation_q)
out_dq.append(continuation_dq)
out_pos.append(continuation_pos)
out_quat.append(continuation_quat_array)

out_q = np.concatenate(out_q, axis=0)
out_dq = np.concatenate(out_dq, axis=0)
out_pos = np.concatenate(out_pos, axis=0)
out_quat = np.concatenate(out_quat, axis=0)

assert out_q.shape == (TOTAL_FRAMES, 29)
assert out_dq.shape == (TOTAL_FRAMES, 29)
assert out_pos.shape[0] == TOTAL_FRAMES
assert out_quat.shape[0] == TOTAL_FRAMES

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,
    dof_pos=out_q,
    dof_vel=out_dq,
    body_pos_w=out_pos,
    body_quat_w=out_quat,
)

print(f"platform root z : {platform_root_z:.6f}")
print(f"total frames    : {TOTAL_FRAMES}")
print(f"duration        : {TOTAL_FRAMES / 50.0:.2f} s")
print(f"saved           : {OUTPUT}")
print("PASS")
