#!/usr/bin/env python3
"""Validate company simulation contracts and an optional external G1 model."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


SIMULATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SIMULATION_ROOT.parent
CONFIG_ROOT = SIMULATION_ROOT / "config"
SYSTEM_CONTRACT = REPOSITORY_ROOT / "docs" / "reference" / "g1_system_contract.md"


def load(name: str) -> dict:
    path = CONFIG_ROOT / name
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_contracts() -> dict:
    robot = load("robot.yaml")
    hand = load("brainco_revo2.yaml")
    mapping = load("joint_mapping.yaml")
    simulator = load("unitree_mujoco.yaml")
    balance = load("balance_gate.yaml")

    require(robot["robot"]["body_dofs"] == 29, "robot must use 29 body DoFs")
    require(robot["control"]["mode_machine"] == 5,
            "this project is confirmed for G1 mode_machine 5")
    require(robot["robot"]["hands_included_in_body_dofs"] is False,
            "hand channels must stay outside the body policy")
    require(robot["dds"]["simulation_domain_id"] == simulator["domain_id"],
            "robot and simulator DDS domains differ")
    require(simulator["domain_id"] != robot["dds"]["unitree_hardware_domain_id"],
            "simulation must not use the hardware DDS domain")
    require(simulator["interface"] == "lo", "simulation must use loopback")
    require(simulator["enable_elastic_band"] == 1,
            "elastic harness must be enabled")

    policy_order = mapping["policy_order"]
    policy_to_sdk = mapping["policy_to_sdk"]
    sdk_order = mapping["sdk_order"]
    require(len(policy_order) == 29 and len(set(policy_order)) == 29,
            "policy order must contain 29 unique joints")
    require(sorted(policy_to_sdk) == list(range(29)),
            "policy-to-SDK mapping must be a permutation of 0..28")
    require(sorted(int(index) for index in sdk_order) == list(range(29)),
            "SDK order must contain indices 0..28")
    for policy_index, sdk_index in enumerate(policy_to_sdk):
        require(policy_order[policy_index] == sdk_order[sdk_index],
                f"joint mapping mismatch at policy index {policy_index}")

    for side in ("right", "left"):
        availability = hand["hands"][side]["availability"]
        require(availability["physical"] is True,
                f"{side} BrainCo hand must be physically available")
        require(availability["physically_verified"] is True,
                f"{side} BrainCo hand physical validation is not recorded")
        require(hand["hands"][side]["dds"]["topics_verified"] is True,
                f"{side} BrainCo DDS topics are not verified")
        joints = hand["hands"][side]["joints"]
        require(len(joints) == 6, f"{side} hand must define six channels")
        require([joint["index"] for joint in joints] == list(range(6)),
                f"{side} hand channel indices must be 0..5")
        require(len({joint["dds_name"] for joint in joints}) == 6,
                f"{side} hand channel names must be unique")
    require(hand["safety"]["human_hand_testing_approved"] is False,
            "repository defaults must not approve human-hand testing")
    require(hand["safety"]["left_hardware_test_status"] ==
            "guarded_card_presentation",
            "left BrainCo validation must retain its guarded card evidence")
    require(balance["initialization"]["elastic_harness_required"] is True,
            "balance gate must require the elastic harness")
    require(balance["controller"]["included"] is True,
            "pinned balance controller must be declared")
    require(balance["controller"]["policy_execution_adapter_included"] is True,
            "balance policy execution adapter must be declared")
    require(balance["controller"]["dds_domain"] == simulator["domain_id"],
            "balance controller and simulator DDS domains differ")
    require(balance["controller"]["dds_interface"] == simulator["interface"],
            "balance controller and simulator interfaces differ")
    require(balance["controller"]["initial_command"] == "zero_velocity",
            "balance controller must start with zero velocity")
    require(balance["controller"]["command_input"] == "keyboard",
            "balance controller must use the documented keyboard input")
    keyboard = balance["controller"]["keyboard_controls"]
    require(keyboard == {
        "forward_reverse_mps": 1.0,
        "lateral_mps": 0.8,
        "yaw_rad_s": 1.0,
        "release_returns_to_zero": True,
    }, "keyboard control limits differ from the reviewed simulation values")

    for key in ("mjcf", "scene"):
        require((REPOSITORY_ROOT / robot["model"][key]).is_file(),
                f"missing vendored G1 {key}")
    require((SIMULATION_ROOT / "vendor" / "unitree_g1" / "LICENSE").is_file(),
            "missing Unitree BSD license")

    require(SYSTEM_CONTRACT.is_file(), "missing consolidated G1 system contract")
    contract_text = SYSTEM_CONTRACT.read_text(encoding="utf-8")
    require("| Machine type (`mode_machine`) | **5** |" in contract_text,
            "G1 system contract does not record mode_machine 5")
    require("simulation/config/joint_mapping.yaml" in contract_text,
            "G1 system contract does not link the joint mapping")
    require("isaac_lab/g1_stairs/task.yaml" in contract_text,
            "G1 system contract does not link the stair task")

    hand_root = REPOSITORY_ROOT / hand["reference"]["vendored_directory"]
    require((hand_root / "LICENSE").is_file(), "missing BrainCo license")
    for side in ("right", "left"):
        urdf = hand_root / "urdf" / f"revo2_{side}_hand.urdf"
        require(urdf.is_file(), f"missing BrainCo {side} URDF")
        document = ET.parse(urdf)
        urdf_joints = {
            joint.attrib.get("name", "") for joint in document.findall(".//joint")
        }
        for channel in hand["hands"][side]["joints"]:
            require(channel["urdf_joint"] in urdf_joints,
                    f"missing BrainCo joint: {channel['urdf_joint']}")
        for mesh in document.findall(".//mesh"):
            filename = mesh.attrib.get("filename", "")
            prefix = "package://revo2_description/"
            require(filename.startswith(prefix),
                    f"unexpected BrainCo mesh URI: {filename}")
            require((hand_root / filename[len(prefix):]).is_file(),
                    f"missing BrainCo mesh: {filename}")
    return robot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    validate_contracts()
    print("VALID: company simulation contracts")
    print("VALID: BrainCo right and left URDF channel mappings and mesh assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
