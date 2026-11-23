"""Containment and recorded walking regressions; no robot access."""

from pathlib import Path

import numpy as np
import pytest

from g1_navigation_perception.depth_geometry import transform_points
from g1_navigation_perception.hand_self_filter import hand_keep_mask


def configuration():
    bounds = {side: [-.1, -.1, -.1, .1, .1, .1]
              for side in ("left", "right")}
    poses = {"left": [1., 2., .7, 0., 0., np.sqrt(.5), np.sqrt(.5)],
             "right": [-5., -5., -5., 0., 0., 0., 1.]}
    return bounds, poses


def test_rotated_translated_hand_preserves_nearby_obstacle_and_floor():
    bounds, poses = configuration()
    local = np.array([[0., 0., 0.], [.1199, 0., 0.], [.121, 0., 0.]])
    pose = poses["left"]
    points = transform_points(local, pose[:3], pose[3:])
    points = np.vstack((points, [.5, 0., -.8], [.5, 0., -.68]))
    original = points.copy()
    keep = hand_keep_mask(points, poses, bounds, padding_m=.02)
    assert keep.tolist() == [False, False, True, True, True]
    np.testing.assert_array_equal(points, original)
    assert hand_keep_mask(np.empty((0, 3)), poses, bounds, padding_m=.02).shape == (0,)


@pytest.mark.parametrize("fault", ["point", "quaternion", "bounds", "padding"])
def test_invalid_context_is_rejected(fault):
    bounds, poses = configuration()
    points = np.zeros((1, 3))
    padding = .02
    if fault == "point":
        points[0, 0] = np.nan
    elif fault == "quaternion":
        poses["left"][3:] = [0., 0., 0., 0.]
    elif fault == "bounds":
        bounds["left"] = [0.] * 6
    else:
        padding = float("nan")
    with pytest.raises(ValueError):
        hand_keep_mask(points, poses, bounds, padding_m=padding)


def test_recorded_open_hand_returns_are_excluded():
    path = Path(__file__).parent / "data" / "hand_self_returns.npz"
    total = 0
    with np.load(path) as data:
        bounds = {s: data[s+"_bounds"] for s in ("left", "right")}
        for index in (16, 17, 18, 19, 20):
            name = f"manual_{index}"
            poses = {s: data[name+"_"+s] for s in ("left", "right")}
            points = data[name+"_points"]
            assert not hand_keep_mask(points, poses, bounds, padding_m=.02).any()
            total += len(points)
    assert total == 214
