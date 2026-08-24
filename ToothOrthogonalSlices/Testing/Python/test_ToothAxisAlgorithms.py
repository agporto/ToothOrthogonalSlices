from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from ToothAxisAlgorithms import (  # noqa: E402
    apply_affine,
    automatic_tooth_segmentation,
    estimate_tooth_axis,
    frame_matrices,
    otsu_threshold,
    parallel_transport_frames,
    resample_polyline,
    sample_volume_along_frames,
    validate_frames,
)


def _rotation_matrix(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=float,
    )


def _curve(t: np.ndarray) -> np.ndarray:
    """A long, smoothly curved synthetic tooth axis in RAS millimetres."""
    t = np.asarray(t, dtype=float)
    return np.column_stack(
        [
            7.0 * np.sin(0.95 * math.pi * t) + 1.5 * t,
            3.5 * np.sin(1.4 * math.pi * t + 0.2),
            62.0 * (t - 0.5),
        ]
    )


def _synthetic_tooth(
    *,
    rotated: bool = True,
    cavity: bool = True,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = np.array([100, 90, 116], dtype=int)  # NumPy z, y, x
    spacing = np.array([0.75, 0.55, 0.65], dtype=float)
    rotation = (
        _rotation_matrix(np.array([0.4, -0.2, 0.8]), math.radians(24.0))
        if rotated
        else np.eye(3)
    )
    linear = rotation @ np.diag(spacing)
    center_index = (shape - 1) / 2.0
    array_to_world = np.eye(4, dtype=float)
    array_to_world[:3, :3] = linear
    array_to_world[:3, 3] = -linear @ center_index

    indices = np.indices(tuple(shape), dtype=np.float32).reshape(3, -1).T
    world = apply_affine(indices, array_to_world)

    t_dense = np.linspace(0.0, 1.0, 1001)
    true_curve = _curve(t_dense)
    tree = cKDTree(true_curve)
    distance, nearest = tree.query(world, k=1)
    nearest_t = t_dense[nearest]
    radius = 3.1 + 2.7 * nearest_t + 0.65 * np.exp(-((nearest_t - 0.86) / 0.13) ** 2)
    inside = distance <= radius
    if cavity:
        cavity_radius = 0.65 + 0.35 * nearest_t
        inside &= distance >= cavity_radius

    rng = np.random.default_rng(seed)
    volume = rng.normal(-950.0, 18.0, size=world.shape[0]).astype(np.float32)
    tissue = 1050.0 + 260.0 * nearest_t + rng.normal(0.0, 22.0, size=world.shape[0])
    volume[inside] = tissue[inside]
    volume = volume.reshape(tuple(shape))
    truth_mask = inside.reshape(tuple(shape))
    return volume, truth_mask, array_to_world, true_curve


def _symmetric_curve_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    d1 = cKDTree(second).query(first, k=1)[0]
    d2 = cKDTree(first).query(second, k=1)[0]
    distances = np.r_[d1, d2]
    return float(np.mean(distances)), float(np.percentile(distances, 95.0))


def test_segmentation_and_centerline_accuracy():
    volume, truth_mask, matrix, true_curve = _synthetic_tooth()
    segmentation = automatic_tooth_segmentation(
        volume,
        matrix,
        threshold_method="otsu",
        closing_radius_mm=0.45,
        fill_axis_holes=True,
    )

    intersection = np.count_nonzero(segmentation.mask & truth_mask)
    union = np.count_nonzero(segmentation.mask | truth_mask)
    assert intersection / union > 0.985
    assert segmentation.threshold_method == "otsu"
    assert segmentation.axis_mask.sum() > segmentation.mask.sum()

    result = estimate_tooth_axis(
        segmentation.axis_mask,
        matrix,
        coarse_spacing_mm=1.25,
        centerline_strength=16.0,
        centerline_exponent=2.0,
        smoothing_mm=1.0,
        output_spacing_mm=0.6,
        control_point_spacing_mm=3.0,
        cross_section_refinement_iterations=1,
    )
    # Ignore a short terminal region because a geometric medial path naturally
    # terminates slightly inside rounded synthetic end caps.
    true_internal = true_curve[35:-35]
    mean_error, p95_error = _symmetric_curve_distance(result.points_world, true_internal)
    assert mean_error < 1.25
    assert p95_error < 2.6
    assert result.qc["inside_fraction"] > 0.98
    assert result.recommended_fov_mm > 12.0
    assert result.control_points_world.shape[0] >= 12


def test_parallel_transport_frames_are_orthonormal_and_continuous():
    points, _ = resample_polyline(_curve(np.linspace(0.0, 1.0, 301)), 0.45)
    frames = parallel_transport_frames(points, initial_normal_world=[1.0, 0.0, 0.0])
    metrics = validate_frames(frames, tolerance=2e-6)
    assert metrics["minimum_x_axis_continuity"] > 0.98

    matrices = frame_matrices(frames)
    assert matrices.shape == (points.shape[0], 4, 4)
    assert np.allclose(matrices[:, :3, 2], frames.tangents_world)
    assert np.allclose(matrices[:, :3, 3], points)


def test_reslicing_is_centered_and_respects_anisotropic_rotated_geometry():
    volume, _, matrix, true_curve = _synthetic_tooth(cavity=False)
    points, _ = resample_polyline(true_curve[30:-30], 1.0)
    frames = parallel_transport_frames(points, initial_normal_world=[1.0, 0.0, 0.0])
    stack, u, v = sample_volume_along_frames(
        volume,
        matrix,
        frames,
        pixel_spacing_mm=0.45,
        field_of_view_mm=18.0,
        interpolation_order=1,
        outside_value=-1000.0,
    )
    assert stack.shape[0] == points.shape[0]
    assert stack.shape[1] == stack.shape[2]
    assert u[stack.shape[2] // 2] == pytest.approx(0.0)
    assert v[stack.shape[1] // 2] == pytest.approx(0.0)

    # The central pixel follows the true axis, while all four corners remain
    # outside the tooth for this field of view.
    middle = stack[:, stack.shape[1] // 2, stack.shape[2] // 2]
    corners = np.column_stack(
        [stack[:, 0, 0], stack[:, 0, -1], stack[:, -1, 0], stack[:, -1, -1]]
    )
    assert np.percentile(middle, 5.0) > 800.0
    assert np.percentile(corners, 95.0) < -800.0


def test_axis_estimation_is_deterministic():
    volume, _, matrix, _ = _synthetic_tooth(seed=22)
    segmentation = automatic_tooth_segmentation(volume, matrix, closing_radius_mm=0.4)
    kwargs = dict(
        coarse_spacing_mm=1.5,
        centerline_strength=14.0,
        smoothing_mm=1.0,
        output_spacing_mm=0.8,
        cross_section_refinement_iterations=0,
    )
    first = estimate_tooth_axis(segmentation.axis_mask, matrix, **kwargs)
    second = estimate_tooth_axis(segmentation.axis_mask, matrix, **kwargs)
    assert np.array_equal(first.points_world, second.points_world)
    assert np.array_equal(first.control_points_world, second.control_points_world)
    assert np.array_equal(first.initial_normal_world, second.initial_normal_world)


def test_tiff_and_preview_io_round_trip(tmp_path):
    import tifffile
    from PIL import Image
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from ToothSliceIO import make_preview_rgb, write_rgb_png, write_scalar_tiff

    scalar = (np.arange(35, dtype=np.int16).reshape(5, 7) - 20).astype(np.int16)
    tiff_path = tmp_path / "slice_0000.tif"
    write_scalar_tiff(tiff_path, scalar)

    reader = vtk.vtkTIFFReader()
    reader.SetFileName(str(tiff_path))
    reader.Update()
    image = reader.GetOutput()
    recovered = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(5, 7)
    assert recovered.dtype == np.float32
    assert np.array_equal(recovered, scalar.astype(np.float32))
    external_recovered = tifffile.imread(tiff_path)
    assert external_recovered.dtype == np.float32
    assert np.array_equal(external_recovered, scalar.astype(np.float32))

    mask = np.zeros_like(scalar, dtype=bool)
    mask[1:4, 2:6] = True
    preview = make_preview_rgb(scalar, mask, (-20.0, 14.0))
    assert preview.shape == (5, 7, 3)
    png_path = tmp_path / "slice_0000.png"
    write_rgb_png(png_path, preview)
    assert png_path.stat().st_size > 0
    assert np.array_equal(np.asarray(Image.open(png_path).convert("RGB")), preview)
    assert not any(tmp_path.glob("*.writing.*"))


def test_roi_limits_automatic_component_selection():
    volume = np.zeros((40, 40, 80), dtype=np.float32)
    volume[10:25, 10:25, 5:25] = 1000.0
    volume[8:32, 8:32, 45:75] = 1200.0  # larger distractor
    matrix = np.eye(4, dtype=float)

    unrestricted = automatic_tooth_segmentation(
        volume, matrix, threshold_method="manual", manual_threshold=500.0, closing_radius_mm=0.0
    )
    assert unrestricted.mask[:, :, 50:70].sum() > 0

    restricted = automatic_tooth_segmentation(
        volume,
        matrix,
        threshold_method="manual",
        manual_threshold=500.0,
        roi_world_bounds=[5.0, 25.0, 5.0, 30.0, 5.0, 30.0],
        closing_radius_mm=0.0,
    )
    assert restricted.mask[:, :, 5:25].sum() > 0
    assert restricted.mask[:, :, 45:75].sum() == 0


def test_automatic_otsu_is_estimated_inside_roi():
    volume = np.full((30, 30, 60), 400.0, dtype=np.float32)
    volume[:, :, 30:] = 1600.0
    volume[5:25, 5:25, 5:25] = 0.0
    volume[10:20, 10:20, 10:20] = 120.0
    matrix = np.eye(4, dtype=float)
    bounds = [4.5, 24.5, 4.5, 24.5, 4.5, 24.5]

    result = automatic_tooth_segmentation(
        volume,
        matrix,
        threshold_method="otsu",
        roi_world_bounds=bounds,
        closing_radius_mm=0.0,
    )
    roi_values = volume[5:25, 5:25, 5:25]
    assert result.threshold == pytest.approx(otsu_threshold(roi_values))
    assert result.mask[10:20, 10:20, 10:20].all()
    assert result.mask[:, :, 30:].sum() == 0


def test_rotated_world_roi_is_applied_exactly():
    shape = (32, 34, 36)
    rotation = _rotation_matrix(np.array([0.2, 0.8, -0.3]), math.radians(31.0))
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ ((np.asarray(shape) - 1) / 2.0)
    volume = np.ones(shape, dtype=np.float32) * 1000.0
    bounds = [-5.0, 5.0, -4.0, 4.0, -6.0, 6.0]

    result = automatic_tooth_segmentation(
        volume,
        matrix,
        threshold_method="manual",
        manual_threshold=500.0,
        roi_world_bounds=bounds,
        closing_radius_mm=0.0,
        fill_axis_holes=False,
    )
    selected_world = apply_affine(np.argwhere(result.mask), matrix)
    assert selected_world.shape[0] > 100
    assert np.all(selected_world[:, 0] >= bounds[0] - 1e-9)
    assert np.all(selected_world[:, 0] <= bounds[1] + 1e-9)
    assert np.all(selected_world[:, 1] >= bounds[2] - 1e-9)
    assert np.all(selected_world[:, 1] <= bounds[3] + 1e-9)
    assert np.all(selected_world[:, 2] >= bounds[4] - 1e-9)
    assert np.all(selected_world[:, 2] <= bounds[5] + 1e-9)
