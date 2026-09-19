#!/usr/bin/env python3
"""Audit source assets needed for the Isaac Lab G1 + BrainCo conversion."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def urdf_mass(path: Path) -> tuple[int, float, tuple[str, ...]]:
    root = ET.parse(path).getroot()
    links = root.findall("link")
    masses = [
        float(mass.attrib["value"])
        for link in links
        if (mass := link.find("inertial/mass")) is not None
    ]
    missing = tuple(
        link.attrib["name"] for link in links if link.find("inertial/mass") is None
    )
    return len(links), sum(masses), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "simulation",
    )
    args = parser.parse_args()
    root = args.simulation_root.resolve()
    g1 = root / "vendor/unitree_g1/g1_29dof.xml"
    if not g1.is_file():
        raise SystemExit(f"missing G1 MJCF: {g1}")
    print(f"G1 source: {g1} (MJCF; convert or replace with Unitree USD/URDF for Isaac Lab)")
    for side in ("left", "right"):
        urdf = root / f"vendor/brainco_revo2/urdf/revo2_{side}_hand.urdf"
        link_count, mass, missing = urdf_mass(urdf)
        print(f"BrainCo {side}: {link_count} links, declared mass {mass:.3f} kg, {urdf}")
        if missing:
            print(f"  massless fixed tip links: {', '.join(missing)}")
    print("Mount contract: simulation/mujoco/model_composition.py HAND_MOUNTS (measurement required)")


if __name__ == "__main__":
    main()
