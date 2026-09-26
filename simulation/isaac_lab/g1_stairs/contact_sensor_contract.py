"""Shared helpers for resolving active ContactSensor body inventories."""

from __future__ import annotations


FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

FORBIDDEN_CONTACT_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_rubber_hand",
    "right_rubber_hand",
)


def resolve_available_body_indices(
    available_names,
    requested_names,
):
    """Return indices for requested bodies that exist in the active sensor.

    Contact sensors only expose bodies represented by their runtime collision
    inventory.  Fixed visual-only bodies can therefore be absent even when
    they exist in the articulation.  Optional diagnostic contact semantics
    must follow the sensor inventory instead of treating those bodies as
    mandatory.
    """

    index_by_name = {
        name: index
        for index, name in enumerate(available_names)
    }
    return {
        name: index_by_name[name]
        for name in requested_names
        if name in index_by_name
    }
