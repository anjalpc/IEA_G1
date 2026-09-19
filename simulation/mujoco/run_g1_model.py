#!/usr/bin/env python3
"""Load the G1 with dual BrainCo hands and an elastic safety harness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import termios
import time

import mujoco
import numpy as np

from model_composition import (
    BODY_ACTUATOR_COUNT,
    HAND_CHANNELS_PER_SIDE,
    build_combined_model,
    build_combined_spec,
    validate_combined_model,
)


class ElasticHarness:
    """Unitree-style overhead elastic tether for controller bring-up."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.body_id < 0:
            raise RuntimeError("G1 model does not contain torso_link")
        self.enabled = True
        self.anchor = np.array([0.0, 0.0, 3.0])
        self.length = 0.0
        self.stiffness = 200.0
        self.damping = 100.0

    def key(self, keycode: int) -> None:
        if keycode == ord("9"):
            self.enabled = not self.enabled
            print(f"HARNESS: {'attached' if self.enabled else 'released'}", flush=True)
        elif keycode == ord("8"):
            self.length += 0.10
            print(f"HARNESS LENGTH: {self.length:.2f} m", flush=True)
        elif keycode == ord("7"):
            self.length = max(0.0, self.length - 0.10)
            print(f"HARNESS LENGTH: {self.length:.2f} m", flush=True)

    def apply(self, data: mujoco.MjData) -> None:
        data.xfrc_applied[self.body_id] = 0.0
        if not self.enabled:
            return
        # Match Unitree's native implementation: calculate the tether from the
        # floating-base translation, then apply that force at torso_link.
        delta = self.anchor - data.qpos[:3]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-9:
            return
        direction = delta / distance
        linear_velocity = data.qvel[:3]
        along_tether = float(np.dot(linear_velocity, direction))
        magnitude = self.stiffness * (distance - self.length)
        magnitude -= self.damping * along_tether
        data.xfrc_applied[self.body_id, :3] = magnitude * direction


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.80
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def set_hand_position(model: mujoco.MjModel, data: mujoco.MjData,
                      normalized_position: float | None) -> None:
    """Set verified neutral hands, or an explicit uniform diagnostic value."""
    targets = {
        "right": (0.399, 0.399, 0.050, 0.050, 0.049, 0.050),
        "left": (0.400, 0.400, 0.049, 0.050, 0.049, 0.050),
    }
    if normalized_position is not None:
        if not 0.0 <= normalized_position <= 1.0:
            raise ValueError("hand position must be in the range 0..1")
        targets = {side: (normalized_position,) * HAND_CHANNELS_PER_SIDE
                   for side in targets}
    for side_index, side in enumerate(("right", "left")):
        first = BODY_ACTUATOR_COUNT + side_index * HAND_CHANNELS_PER_SIDE
        for channel, normalized in enumerate(targets[side]):
            actuator_id = first + channel
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = low + normalized * (high - low)


def export_mjcf(path: Path) -> None:
    """Export a native-loader-compatible combined G1 and BrainCo MJCF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_combined_spec()
    spec.compile()
    spec.to_file(str(path))
    exported = path.read_text(encoding="utf-8")
    # MjSpec emits anonymous empty nested defaults for the two attached URDFs.
    exported = re.sub(r"^\s*<default\s*/>\s*$\n?", "", exported,
                      flags=re.MULTILINE)
    g1_meshes = (
        Path(__file__).resolve().parent / "vendor" / "unitree_g1" / "meshes"
    )
    exported = exported.replace(
        'meshdir="meshes/"', f'meshdir="{g1_meshes}/"', 1)
    path.write_text(exported, encoding="utf-8")
    print(f"EXPORTED: {path}", flush=True)


def run_native_viewer(balance: bool = False, card_presentation: bool = False) -> int:
    """Export the model and run Unitree's native viewer inside Docker."""
    native_binary = os.environ.get("UNITREE_MUJOCO_BIN")
    if not native_binary:
        raise RuntimeError(
            "UNITREE_MUJOCO_BIN is unavailable; run this inside ./scripts/dev.sh")

    generated_model = Path("/tmp/iea_g1_combined.xml")
    export_mjcf(generated_model)
    renderer = subprocess.run(
        ["glxinfo", "-B"], check=True, capture_output=True, text=True)
    renderer_text = renderer.stdout + renderer.stderr
    for line in renderer_text.splitlines():
        if any(label in line for label in (
                "direct rendering:", "Accelerated:",
                "OpenGL renderer string:")):
            print(line, flush=True)
    if "llvmpipe" in renderer_text.lower():
        raise RuntimeError("Docker is using software rendering, not the host GPU")

    # ROS Humble prepends its own CycloneDDS C library when setup.bash is
    # sourced. Unitree's native simulator and controller use the matching
    # CycloneDDS C++ library under /opt/unitree_robotics, so mixing the two
    # builds causes Iceoryx assertions and invalid frees. Keep this override
    # local to the native simulation children; ROS nodes retain their normal
    # Humble runtime.
    native_env = os.environ.copy()
    unitree_library = "/opt/unitree_robotics/lib"
    library_paths = native_env.get("LD_LIBRARY_PATH", "").split(":")
    library_paths = [path for path in library_paths
                     if path and path != unitree_library]
    native_env["LD_LIBRARY_PATH"] = ":".join(
        [unitree_library, *library_paths])
    if card_presentation:
        project_root = Path(__file__).resolve().parents[2]
        overlay = project_root / "docker" / "native_card_overlay.h"
        patcher = project_root / "docker" / "patch_native_card_overlay.py"
        subprocess.run(
            ["sudo", "cp", str(overlay),
             "/opt/unitree_mujoco/simulate/src/native_card_overlay.h"],
            check=True,
        )
        subprocess.run(["sudo", "python3", str(patcher)], check=True)
        subprocess.run(
            ["sudo", "cmake", "--build",
             "/opt/unitree_mujoco/simulate/build", "--parallel"],
            check=True,
        )
        native_env["IEA_G1_LEFT_CARD"] = "1"
        print(
            "LEFT CARD: starts after 8 simulated seconds, moves for 5 seconds, "
            "holds for 5 seconds, then returns; left-hand grip ramps in 1.5 seconds",
            flush=True,
        )

    terminal_fd = None
    terminal_settings = None
    if sys.stdin.isatty():
        terminal_fd = sys.stdin.fileno()
        terminal_settings = termios.tcgetattr(terminal_fd)

    simulator = subprocess.Popen([
        native_binary, "-r", "g1", "-s", str(generated_model),
        "-i", "1", "-n", "lo",
    ], env=native_env)
    controller = None
    try:
        if balance:
            controller_binary = os.environ.get("UNITREE_BALANCE_CONTROLLER")
            if not controller_binary:
                raise RuntimeError("Unitree balance controller is not installed")
            print(
                "BALANCE: starting pinned Unitree velocity policy on DDS "
                "domain 1; keep harness attached until stable\n"
                "KEYBOARD (terminal focus): hold W/S forward/back, A/D "
                "left/right, Q/E turn; release returns to zero",
                flush=True,
            )
            time.sleep(2.0)
            controller = subprocess.Popen(
                [controller_binary, "-n", "lo"], env=native_env)

        while True:
            simulator_status = simulator.poll()
            if simulator_status is not None:
                return simulator_status
            if controller is not None:
                controller_status = controller.poll()
                if controller_status is not None:
                    raise RuntimeError(
                        "Unitree balance controller stopped with status "
                        f"{controller_status}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 130
    finally:
        # Ctrl+C is delivered to the whole foreground process group. Ignore
        # any repeated SIGINT while this parent terminates both native children.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for process in (controller, simulator):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (controller, simulator):
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        # Unitree's C++ keyboard reader disables ICANON and ECHO. A signal can
        # terminate it before its destructor restores those flags, so the
        # parent must always restore the terminal snapshot itself.
        if terminal_fd is not None and terminal_settings is not None:
            termios.tcsetattr(
                terminal_fd, termios.TCSAFLUSH, terminal_settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--native", action="store_true",
        help="run Unitree's native C++ viewer (inside the project Docker)")
    parser.add_argument(
        "--balance", action="store_true",
        help="run the pinned Unitree zero-velocity balance policy with --native")
    parser.add_argument(
        "--card-presentation", action="store_true",
        help="run the simulation-only left-card overlay with --native --balance")
    parser.add_argument(
        "--export-mjcf", type=Path, metavar="PATH",
        help="write the composed G1 + BrainCo MJCF for the native viewer")
    parser.add_argument(
        "--render-fps", type=float, default=30.0,
        help="interactive render rate; physics remains near real time")
    parser.add_argument(
        "--hand-position", type=float, default=None, metavar="0..1",
        help="uniform diagnostic target; default uses verified neutral values")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.render_fps <= 0.0:
        raise ValueError("render-fps must be positive")
    if args.native and (args.headless or args.validate_only
                        or args.export_mjcf is not None):
        parser.error("--native cannot be combined with headless/validation/export")
    if args.balance and not args.native:
        parser.error("--balance requires --native")
    if args.card_presentation and not (args.native and args.balance):
        parser.error("--card-presentation requires --native --balance")

    if args.native:
        return run_native_viewer(
            balance=args.balance, card_presentation=args.card_presentation)

    model = build_combined_model()
    validate_combined_model(model)
    data = mujoco.MjData(model)
    initialize(model, data)
    set_hand_position(model, data, args.hand_position)
    harness = ElasticHarness(model, data)

    if args.export_mjcf is not None:
        export_mjcf(args.export_mjcf)
        return 0

    if args.validate_only:
        print(
            f"VALID: combined G1 + dual BrainCo model with {model.nq} qpos, "
            f"{model.nv} qvel, {model.nu} actuators")
        print(f"VALID: G1 body-policy contract has {BODY_ACTUATOR_COUNT} actuators")
        print(
            f"VALID: BrainCo right hand has {HAND_CHANNELS_PER_SIDE} actuators "
            "and 5 mimic constraints")
        print(
            f"VALID: BrainCo left hand has {HAND_CHANNELS_PER_SIDE} actuators "
            "and 5 mimic constraints")
        print("VALID: both BrainCo bases are attached to the G1 wrist-yaw links")
        print("VALID: scene floor is loaded and gravity is 0 0 -9.81 m/s^2")
        print("VALID: torso elastic tether controls 9 toggle, 8 lower, 7 raise")
        return 0

    def step() -> None:
        harness.apply(data)
        mujoco.mj_step(model, data)

    if args.headless:
        for _ in range(args.steps):
            step()
        print(f"HEADLESS COMPLETE: pelvis_z={data.xpos[harness.body_id, 2]:.3f} m")
        return 0

    from mujoco import viewer as mj_viewer

    frame_period = 1.0 / args.render_fps
    substeps = max(1, round(frame_period / model.opt.timestep))
    print(
        f"G1 viewer: {args.render_fps:g} FPS, {substeps} physics substeps/frame; "
        "9 toggle harness, 8 raise, 7 lower.")
    with mj_viewer.launch_passive(
            model, data, key_callback=harness.key) as viewer:
        while viewer.is_running():
            started = time.monotonic()
            for _ in range(substeps):
                step()
            viewer.sync()
            remaining = frame_period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
