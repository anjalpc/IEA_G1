"""Offline depth geometry regression tests."""

import numpy as np


def test_back_projection_uses_metres_optical_axes_and_skips_zero():
    from g1_navigation_perception.depth_geometry import depth_to_points

    depth = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)
    k = np.array([[2., 0., 1.], [0., 4., 0.], [0., 0., 1.]])
    original = depth.copy()

    points = depth_to_points(depth, k, depth_scale=0.001)

    np.testing.assert_allclose(
        points, [[-0.5, 0., 1.], [-1., 0.5, 2.], [0., 0.25, 1.]])
    np.testing.assert_array_equal(depth, original)


def test_rejects_invalid_depth_scales():
    import pytest
    from g1_navigation_perception.depth_geometry import depth_to_points

    depth = np.array([[1000]], dtype=np.uint16)
    for scale in (0., -0.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="depth_scale"):
            depth_to_points(depth, np.eye(3), depth_scale=scale)


def test_rejects_invalid_camera_intrinsics():
    import pytest
    from g1_navigation_perception.depth_geometry import depth_to_points

    depth = np.array([[1000]], dtype=np.uint16)
    for row, column, value in (
            (0, 0, 0.), (1, 1, -1.),
            (0, 2, float("nan")), (1, 2, float("inf"))):
        k = np.eye(3)
        k[row, column] = value
        with pytest.raises(ValueError, match="intrinsics"):
            depth_to_points(depth, k, depth_scale=0.001)


def test_filters_invalid_depth_and_preserves_empty_xyz_shape():
    from g1_navigation_perception.depth_geometry import depth_to_points

    depth = np.array([[0., -1., np.nan, np.inf, 1000.]])
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        points = depth_to_points(depth, np.eye(3), depth_scale=0.001)
    np.testing.assert_allclose(points, [[4., 0., 1.]])

    empty = depth_to_points(depth[:, :4], np.eye(3), depth_scale=0.001)
    assert empty.shape == (0, 3)


def test_plane_heights_are_signed_metres_with_scaled_normal():
    from g1_navigation_perception.depth_geometry import heights_above_plane

    points = [[0., 0., 1.], [2., 3., 1.15], [0., 0., 0.9]]
    heights = heights_above_plane(points, [0., 0., 2., -2.])
    np.testing.assert_allclose(heights, [0., 0.15, -0.1], atol=1e-12)


def test_rejects_invalid_floor_planes():
    import pytest
    from g1_navigation_perception.depth_geometry import heights_above_plane

    for plane in ([0., 0., 0., 1.], [0., 0., np.nan, 0.], [0., 1.]):
        with pytest.raises(ValueError, match="plane"):
            heights_above_plane([[0., 0., 0.]], plane)


def test_rigid_transform_rotates_then_translates_without_mutation():
    from g1_navigation_perception.depth_geometry import transform_points

    points = np.array([[1., 0., 0.], [0., 0., 0.]])
    original = points.copy()
    q = [0., 0., np.sqrt(0.5), np.sqrt(0.5)]
    result = transform_points(points, [1., 2., 3.], q)
    np.testing.assert_allclose(result, [[1., 3., 3.], [1., 2., 3.]])
    np.testing.assert_array_equal(points, original)
    assert transform_points(np.empty((0, 3)), [0., 0., 0.], q).shape == (0, 3)


def test_rigid_transform_rejects_invalid_inputs():
    import pytest
    from g1_navigation_perception.depth_geometry import transform_points

    for points, translation, q in (
            ([[0., 0., 0.]], [0., 0., 0.], [0., 0., 0., 0.]),
            ([[0., 0., 0.]], [0., 0., 0.], [0., 0., 0., 2.]),
            ([[0., 0., 0.]], [0., 0., np.nan], [0., 0., 0., 1.]),
            ([[np.inf, 0., 0.]], [0., 0., 0.], [0., 0., 0., 1.])):
        with pytest.raises(ValueError):
            transform_points(points, translation, q)
