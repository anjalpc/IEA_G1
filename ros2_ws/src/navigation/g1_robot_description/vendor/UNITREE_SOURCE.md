# Unitree G1 Source Provenance

## Upstream

- Repository: https://github.com/unitreerobotics/unitree_ros
- Commit: `5994d4faef0a9cadd3287f8de0199a67eeb2a259`
- Source: `robots/g1_description/g1_29dof.urdf`
- Raw source: https://raw.githubusercontent.com/unitreerobotics/unitree_ros/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof.urdf
- SHA-256: `e1dc89366bf96aa33ba4ef211884f6d81320ddd2ac6f8b1ad82986e64008be31`
- License: BSD-3-Clause, retained in `vendor/LICENSE`

## Local derivation

`urdf/g1_navigation.urdf` will be a kinematic-only subset derived
from this pinned official Unitree 29-DOF description.

The local subset retains only the navigation-relevant chain from
`pelvis` through the moving waist to the torso/head, D435, MID-360,
and IMU reference frames. Visual, collision, inertial, limb, and
Unitree rubber-hand elements are intentionally excluded.

The installed physical hands are BrainCo Revo2 devices and remain
governed by their existing project descriptions. This navigation
subset must not represent them as Unitree rubber hands.

Every retained joint origin, axis, type, and limit must be covered by
contract tests against the values recorded from this pinned source.
