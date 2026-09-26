"""G1 stair imitation-learning package.

This package is intentionally separate from the pure-RL stair stack under
``simulation/isaac_lab/g1_stairs``.  The imitation stack may reuse validated
robot/terrain components, but the pure-RL stack must not depend on this package.
"""

from .reference import G1StairImitationReference, load_g1_stair_imitation_reference

__all__ = [
    "G1StairImitationReference",
    "load_g1_stair_imitation_reference",
]
