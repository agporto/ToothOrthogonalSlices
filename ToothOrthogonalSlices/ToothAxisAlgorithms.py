"""Numerical core for Tooth Orthogonal Slices.

This module intentionally has no 3D Slicer dependencies.  It contains the
segmentation, centerline, frame, and resampling algorithms used by the Slicer
module and can therefore be tested in a regular Python environment.

Coordinate conventions
----------------------
Arrays use NumPy ``(z, y, x)`` index order.  ``array_to_world`` is a 4x4
homogeneous matrix that maps ``[z, y, x, 1]`` to Slicer RAS world coordinates.
Curves and frames are always expressed in world millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


ALGORITHM_VERSION = "0.1.0"
_EPS = np.finfo(float).eps


@dataclass
class SegmentationResult:
    mask: np.ndarray
    axis_mask: np.ndarray
    threshold: float
    threshold_method: str
    component_voxels: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class AxisResult:
    points_world: np.ndarray
    raw_path_world: np.ndarray
    control_points_world: np.ndarray
    pca_center_world: np.ndarray
    pca_axes_world: np.ndarray
    initial_normal_world: np.ndarray
    recommended_fov_mm: float
    qc: Dict[str, object]
    warnings: List[str] = field(default_factory=list)


@dataclass
class FrameResult:
    centers_world: np.ndarray
    x_axes_world: np.ndarray
    y_axes_world: np.ndarray
    tangents_world: np.ndarray
    arc_lengths_mm: np.ndarray


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 homogeneous matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Transform contains non-finite values")
    return matrix


def normalize(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    if fallback is None:
        raise ValueError("Cannot normalize a near-zero vector")
    fallback = np.asarray(fallback, dtype=float)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-12:
        raise ValueError("Fallback vector is also near zero")
    return fallback / fallback_norm


def canonicalize_vector_sign(vector: np.ndarray) -> np.ndarray:
    """Choose a deterministic sign for a PCA eigenvector."""
    vector = normalize(vector)
    index = int(np.argmax(np.abs(vector)))
    return vector if vector[index] >= 0.0 else -vector


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to one point or an ``N x 3`` array."""
    matrix = _as_float_matrix(matrix)
    points = np.asarray(points, dtype=float)
    original_shape = points.shape
    if original_shape == (3,):
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (3,) or (N, 3)")
    homogeneous = np.c_[points, np.ones(points.shape[0], dtype=float)]
    transformed = homogeneous @ matrix.T
    transformed = transformed[:, :3] / transformed[:, 3:4]
    return transformed[0] if original_shape == (3,) else transformed


def array_axis_spacings(array_to_world: np.ndarray) -> np.ndarray:
    matrix = _as_float_matrix(array_to_world)
    spacings = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(spacings <= 0.0):
        raise ValueError("Array-to-world matrix has a degenerate axis")
    return spacings


def _sample_values(values: np.ndarray, maximum_samples: int = 2_000_000) -> np.ndarray:
    flat = np.asarray(values).ravel()
    stride = max(1, int(math.ceil(flat.size / float(maximum_samples))))
    sampled = np.asarray(flat[::stride], dtype=np.float64)
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size == 0:
        raise ValueError("Input contains no finite scalar values")
    return sampled


def otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
    """Compute a robust two-class Otsu threshold.

    Extreme tails are clipped before histogramming to prevent a few outliers
    from consuming the useful histogram range.
    """
    sampled = _sample_values(values)
    low, high = np.percentile(sampled, [0.02, 99.98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(np.median(sampled))
    sampled = np.clip(sampled, low, high)
    histogram, edges = np.histogram(sampled, bins=int(bins), range=(low, high))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total <= 0.0:
        return float(np.median(sampled))
    probability = histogram / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(probability)
    mean = np.cumsum(probability * centers)
    global_mean = mean[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-15
    between[valid] = (global_mean * omega[valid] - mean[valid]) ** 2 / denominator[valid]
    index = int(np.argmax(between[:-1]))
    return float(edges[index + 1])


def ellipsoid_structure(radius_mm: float, spacing_zyx: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing_zyx, dtype=float)
    if radius_mm <= 0.0:
        return np.ones((1, 1, 1), dtype=bool)
    radii = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    z, y, x = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    normalized = (
        (z * spacing[0] / radius_mm) ** 2
        + (y * spacing[1] / radius_mm) ** 2
        + (x * spacing[2] / radius_mm) ** 2
    )
    return normalized <= 1.0 + 1e-12


def _nearest_foreground_index(
    mask: np.ndarray,
    seed_index_zyx: Sequence[float],
    spacing_zyx: Sequence[float],
    maximum_distance_mm: float,
) -> Optional[Tuple[int, int, int]]:
    seed = np.rint(seed_index_zyx).astype(int)
    shape = np.asarray(mask.shape, dtype=int)
    if np.all(seed >= 0) and np.all(seed < shape) and mask[tuple(seed)]:
        return tuple(int(v) for v in seed)

    spacing = np.asarray(spacing_zyx, dtype=float)
    radius_voxels = np.ceil(maximum_distance_mm / spacing).astype(int)
    lower = np.maximum(0, seed - radius_voxels)
    upper = np.minimum(shape, seed + radius_voxels + 1)
    slices = tuple(slice(int(lower[d]), int(upper[d])) for d in range(3))
    local = mask[slices]
    foreground = np.argwhere(local)
    if foreground.size == 0:
        return None
    global_indices = foreground + lower
    distances = np.linalg.norm((global_indices - np.asarray(seed_index_zyx)) * spacing, axis=1)
    best = int(np.argmin(distances))
    if distances[best] > maximum_distance_mm:
        return None
    return tuple(int(v) for v in global_indices[best])


def select_connected_component(
    candidate: np.ndarray,
    spacing_zyx: Sequence[float],
    seed_index_zyx: Optional[Sequence[float]] = None,
    seed_search_radius_mm: float = 8.0,
) -> Tuple[np.ndarray, int, List[str]]:
    candidate = np.asarray(candidate, dtype=bool)
    warnings: List[str] = []
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(candidate, structure=structure)
    if count == 0:
        raise ValueError("Thresholding produced no foreground component")

    selected_label = 0
    if seed_index_zyx is not None:
        nearest = _nearest_foreground_index(
            candidate, seed_index_zyx, spacing_zyx, seed_search_radius_mm
        )
        if nearest is None:
            warnings.append(
                "The seed did not reach thresholded material; the largest component was used."
            )
        else:
            selected_label = int(labels[nearest])

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if selected_label <= 0:
        selected_label = int(np.argmax(sizes))
    component_size = int(sizes[selected_label])
    if component_size <= 0:
        raise ValueError("The selected threshold component is empty")
    return labels == selected_label, component_size, warnings


def automatic_tooth_segmentation(
    volume: np.ndarray,
    array_to_world: np.ndarray,
    threshold_method: str = "otsu",
    manual_threshold: Optional[float] = None,
    seed_world: Optional[Sequence[float]] = None,
    roi_world_bounds: Optional[Sequence[float]] = None,
    closing_radius_mm: float = 0.35,
    fill_axis_holes: bool = True,
    seed_search_radius_mm: float = 8.0,
) -> SegmentationResult:
    """Threshold a high-intensity tooth and keep one connected component."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("Input volume must be three-dimensional")
    array_to_world = _as_float_matrix(array_to_world)
    spacing = array_axis_spacings(array_to_world)

    roi_mask: Optional[np.ndarray] = None
    if roi_world_bounds is not None:
        bounds = np.asarray(roi_world_bounds, dtype=float)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            raise ValueError("ROI bounds must contain six finite world coordinates")
        if np.any(bounds[1::2] <= bounds[0::2]):
            raise ValueError("Each ROI maximum must exceed its minimum")
        roi_mask = _mask_from_world_bounds(volume.shape, array_to_world, bounds)
        if not np.any(roi_mask):
            raise ValueError("The ROI does not intersect the input volume")

    method = str(threshold_method).strip().lower()
    if method in {"manual", "fixed"}:
        if manual_threshold is None or not np.isfinite(manual_threshold):
            raise ValueError("A finite manual threshold is required")
        threshold = float(manual_threshold)
        method = "manual"
    elif method in {"otsu", "automatic"}:
        # When an ROI is supplied, estimate the threshold from exactly that
        # world-space box.  Computing Otsu from the entire scan can otherwise
        # be dominated by unrelated anatomy or scan-bed material.
        threshold = otsu_threshold(volume[roi_mask] if roi_mask is not None else volume)
        method = "otsu"
    else:
        raise ValueError("Supported threshold methods are 'otsu' and 'manual'")

    candidate = np.isfinite(volume) & (volume >= threshold)
    if roi_mask is not None:
        candidate &= roi_mask
    if closing_radius_mm > 0.0:
        candidate = ndimage.binary_closing(
            candidate,
            structure=ellipsoid_structure(closing_radius_mm, spacing),
            iterations=1,
        )

    seed_index = None
    if seed_world is not None:
        seed_index = apply_affine(seed_world, np.linalg.inv(array_to_world))

    mask, component_voxels, warnings = select_connected_component(
        candidate,
        spacing,
        seed_index_zyx=seed_index,
        seed_search_radius_mm=seed_search_radius_mm,
    )

    # Keep the anatomical segmentation separate from the solid mask used only
    # for centerline extraction.
    axis_mask = mask.copy()
    if fill_axis_holes:
        axis_mask = ndimage.binary_fill_holes(axis_mask)
    if closing_radius_mm > 0.0:
        axis_mask = ndimage.binary_closing(
            axis_mask,
            structure=ellipsoid_structure(max(closing_radius_mm, float(spacing.min())), spacing),
            iterations=1,
        )
    axis_mask, _, component_warnings = select_connected_component(axis_mask, spacing)
    warnings.extend(component_warnings)

    if component_voxels < 100:
        warnings.append("The selected component contains fewer than 100 voxels.")

    return SegmentationResult(
        mask=np.asarray(mask, dtype=bool),
        axis_mask=np.asarray(axis_mask, dtype=bool),
        threshold=threshold,
        threshold_method=method,
        component_voxels=component_voxels,
        warnings=warnings,
    )


def _mask_from_world_bounds(
    shape_zyx: Sequence[int],
    array_to_world: np.ndarray,
    bounds_ras: Sequence[float],
) -> np.ndarray:
    """Return voxels whose centers lie inside a world-axis-aligned RAS box.

    A transformed volume is generally not axis-aligned in world coordinates.
    Merely transforming the ROI corners back to array coordinates gives a
    conservative array-aligned box and can include substantial material
    outside the requested ROI.  This helper first restricts work to that box,
    then evaluates the exact world-coordinate inequalities voxel by voxel.
    """
    shape = np.asarray(shape_zyx, dtype=int)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("Volume shape must contain three positive dimensions")
    matrix = _as_float_matrix(array_to_world)
    bounds = np.asarray(bounds_ras, dtype=float)
    if bounds.shape != (6,):
        raise ValueError("World bounds must contain six values")

    corners_world = np.asarray(
        [
            [x, y, z]
            for x in (bounds[0], bounds[1])
            for y in (bounds[2], bounds[3])
            for z in (bounds[4], bounds[5])
        ],
        dtype=float,
    )
    corners_array = apply_affine(corners_world, np.linalg.inv(matrix))
    lower = np.maximum(0, np.floor(corners_array.min(axis=0)).astype(int) - 1)
    upper = np.minimum(shape, np.ceil(corners_array.max(axis=0)).astype(int) + 2)
    result = np.zeros(tuple(shape), dtype=bool)
    if np.any(upper <= lower):
        return result

    z = np.arange(lower[0], upper[0], dtype=float)[:, None, None]
    y = np.arange(lower[1], upper[1], dtype=float)[None, :, None]
    x = np.arange(lower[2], upper[2], dtype=float)[None, None, :]
    local = np.ones(tuple((upper - lower).astype(int)), dtype=bool)
    for world_axis in range(3):
        coordinate = (
            matrix[world_axis, 0] * z
            + matrix[world_axis, 1] * y
            + matrix[world_axis, 2] * x
            + matrix[world_axis, 3]
        )
        local &= coordinate >= bounds[2 * world_axis]
        local &= coordinate <= bounds[2 * world_axis + 1]
    result[
        int(lower[0]) : int(upper[0]),
        int(lower[1]) : int(upper[1]),
        int(lower[2]) : int(upper[2]),
    ] = local
    return result


def prepare_axis_mask(
    anatomical_mask: np.ndarray,
    array_to_world: np.ndarray,
    closing_radius_mm: float = 0.35,
    fill_holes: bool = True,
) -> np.ndarray:
    """Create a solid, single-component mask for centerline extraction.

    The input anatomical mask is never modified.  Internal cavities may be
    filled because the medial path should follow the tooth envelope rather
    than branch around pulp or enamel spaces.
    """
    mask = np.asarray(anatomical_mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("Anatomical mask must be a non-empty 3D binary array")
    spacing = array_axis_spacings(array_to_world)
    prepared = mask.copy()
    if fill_holes:
        prepared = ndimage.binary_fill_holes(prepared)
    if closing_radius_mm > 0.0:
        prepared = ndimage.binary_closing(
            prepared,
            structure=ellipsoid_structure(
                max(float(closing_radius_mm), float(spacing.min())), spacing
            ),
            iterations=1,
        )
    prepared, _, _ = select_connected_component(prepared, spacing)
    return np.asarray(prepared, dtype=bool)


def crop_to_mask(
    mask: np.ndarray,
    array_to_world: np.ndarray,
    margin_mm: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("Cannot crop an empty mask")
    spacing = array_axis_spacings(array_to_world)
    margin = np.ceil(float(margin_mm) / spacing).astype(int)
    lower = np.maximum(0, indices.min(axis=0) - margin)
    upper = np.minimum(np.asarray(mask.shape), indices.max(axis=0) + margin + 1)
    slices = tuple(slice(int(lower[d]), int(upper[d])) for d in range(3))
    cropped = mask[slices]
    translation = np.eye(4, dtype=float)
    translation[:3, 3] = lower
    cropped_to_world = _as_float_matrix(array_to_world) @ translation
    return cropped, cropped_to_world, lower


def block_reduce_max(
    mask: np.ndarray,
    array_to_world: np.ndarray,
    target_spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative integer-stride downsampling for binary masks."""
    mask = np.asarray(mask, dtype=bool)
    spacing = array_axis_spacings(array_to_world)
    target = max(float(target_spacing_mm), float(spacing.min()))
    strides = np.maximum(1, np.ceil(target / spacing).astype(int))
    padded_shape = np.ceil(np.asarray(mask.shape) / strides).astype(int) * strides
    pad_width = [(0, int(padded_shape[d] - mask.shape[d])) for d in range(3)]
    padded = np.pad(mask, pad_width, mode="constant", constant_values=False)
    reshaped = padded.reshape(
        padded_shape[0] // strides[0], strides[0],
        padded_shape[1] // strides[1], strides[1],
        padded_shape[2] // strides[2], strides[2],
    )
    reduced = reshaped.max(axis=(1, 3, 5))

    index_transform = np.eye(4, dtype=float)
    index_transform[0, 0] = strides[0]
    index_transform[1, 1] = strides[1]
    index_transform[2, 2] = strides[2]
    index_transform[:3, 3] = 0.5 * (strides - 1)
    reduced_to_world = _as_float_matrix(array_to_world) @ index_transform
    return reduced, reduced_to_world, strides


def principal_axes(points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("At least three 3D points are required for PCA")
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    for column in range(3):
        vectors[:, column] = canonicalize_vector_sign(vectors[:, column])
    # Rebuild a right-handed orthonormal basis after independent sign choices.
    vectors[:, 0] = normalize(vectors[:, 0])
    vectors[:, 1] -= vectors[:, 0] * np.dot(vectors[:, 0], vectors[:, 1])
    vectors[:, 1] = normalize(vectors[:, 1])
    vectors[:, 2] = normalize(np.cross(vectors[:, 0], vectors[:, 1]))
    if np.dot(vectors[:, 2], canonicalize_vector_sign(vectors[:, 2])) < 0.0:
        vectors[:, 1] *= -1.0
        vectors[:, 2] *= -1.0
    return center, vectors, values


def _choose_endpoint(
    coordinates: np.ndarray,
    projections: np.ndarray,
    clearance: np.ndarray,
    low_end: bool,
    cap_fraction: float,
) -> np.ndarray:
    quantile = cap_fraction if low_end else 1.0 - cap_fraction
    boundary = float(np.quantile(projections, quantile))
    candidates = projections <= boundary if low_end else projections >= boundary
    candidate_indices = np.where(candidates)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.arange(coordinates.shape[0])
    p = projections[candidate_indices]
    c = clearance[candidate_indices]
    p_span = max(float(np.ptp(projections)), 1e-6)
    extremity = (projections.max() - p) / p_span if low_end else (p - projections.min()) / p_span
    clearance_scale = max(float(clearance.max()), 1e-6)
    score = c / clearance_scale + 0.20 * extremity
    return coordinates[candidate_indices[int(np.argmax(score))]]


def _neighbor_offsets(spacing_zyx: Sequence[float]) -> List[Tuple[int, int, int, float]]:
    spacing = np.asarray(spacing_zyx, dtype=float)
    offsets: List[Tuple[int, int, int, float]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                step = float(np.linalg.norm(np.array([dz, dy, dx]) * spacing))
                offsets.append((dz, dy, dx, step))
    return offsets


def center_biased_shortest_path(
    mask: np.ndarray,
    clearance_mm: np.ndarray,
    start_zyx: Sequence[int],
    goal_zyx: Sequence[int],
    spacing_zyx: Sequence[float],
    centerline_strength: float = 12.0,
    centerline_exponent: float = 2.0,
) -> np.ndarray:
    """Dijkstra path through a mask, penalizing positions near its surface."""
    mask = np.asarray(mask, dtype=bool)
    clearance = np.asarray(clearance_mm, dtype=float)
    if mask.shape != clearance.shape:
        raise ValueError("Mask and clearance arrays must have identical shapes")
    start = tuple(int(v) for v in start_zyx)
    goal = tuple(int(v) for v in goal_zyx)
    if not mask[start] or not mask[goal]:
        raise ValueError("Both centerline endpoints must lie inside the mask")

    maximum_clearance = max(float(clearance[mask].max()), 1e-6)
    normalized_surface_distance = np.clip(clearance / maximum_clearance, 0.0, 1.0)
    cost_field = 1.0 + float(centerline_strength) * (
        1.0 - normalized_surface_distance
    ) ** float(centerline_exponent)

    shape = mask.shape
    total_size = int(np.prod(shape))
    distances = np.full(total_size, np.inf, dtype=np.float64)
    predecessors = np.full(total_size, -1, dtype=np.int64)
    start_flat = int(np.ravel_multi_index(start, shape))
    goal_flat = int(np.ravel_multi_index(goal, shape))
    distances[start_flat] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start_flat)]
    neighbors = _neighbor_offsets(spacing_zyx)

    while heap:
        distance, flat = heapq.heappop(heap)
        if distance != distances[flat]:
            continue
        if flat == goal_flat:
            break
        z, y, x = np.unravel_index(flat, shape)
        current_cost = cost_field[z, y, x]
        for dz, dy, dx, step in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx
            if nz < 0 or ny < 0 or nx < 0 or nz >= shape[0] or ny >= shape[1] or nx >= shape[2]:
                continue
            if not mask[nz, ny, nx]:
                continue
            neighbor_flat = int(np.ravel_multi_index((nz, ny, nx), shape))
            edge_cost = step * 0.5 * (current_cost + cost_field[nz, ny, nx])
            candidate = distance + edge_cost
            if candidate < distances[neighbor_flat]:
                distances[neighbor_flat] = candidate
                predecessors[neighbor_flat] = flat
                heapq.heappush(heap, (candidate, neighbor_flat))

    if not np.isfinite(distances[goal_flat]):
        raise RuntimeError("No connected path was found between the selected tooth endpoints")

    path_flat: List[int] = [goal_flat]
    current = goal_flat
    while current != start_flat:
        current = int(predecessors[current])
        if current < 0:
            raise RuntimeError("Centerline predecessor chain is incomplete")
        path_flat.append(current)
    path_flat.reverse()
    path = np.column_stack(np.unravel_index(np.asarray(path_flat, dtype=np.int64), shape))
    return path.astype(float)


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Curve points must have shape (N, 3)")
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(lengths)]


def remove_duplicate_points(points: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] <= 1:
        return points.copy()
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance]
    return points[keep]


def resample_polyline(
    points: np.ndarray,
    spacing_mm: float,
    include_endpoint: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    points = remove_duplicate_points(points)
    if points.shape[0] < 2:
        raise ValueError("At least two distinct points are required")
    spacing = float(spacing_mm)
    if spacing <= 0.0:
        raise ValueError("Sampling spacing must be positive")
    arc = cumulative_arc_length(points)
    total = float(arc[-1])
    if total <= 0.0:
        raise ValueError("Curve length is zero")
    sample_arc = np.arange(0.0, total + 0.5 * spacing, spacing)
    if include_endpoint and (sample_arc.size == 0 or total - sample_arc[-1] > 1e-9):
        sample_arc = np.r_[sample_arc, total]
    sample_arc = np.clip(sample_arc, 0.0, total)
    sample_arc = np.unique(sample_arc)
    sampled = np.column_stack([
        np.interp(sample_arc, arc, points[:, dimension]) for dimension in range(3)
    ])
    return sampled, sample_arc


def smooth_curve(
    points: np.ndarray,
    sampling_spacing_mm: float,
    smoothing_mm: float,
) -> np.ndarray:
    points = remove_duplicate_points(points)
    if points.shape[0] < 4 or smoothing_mm <= 0.0:
        return resample_polyline(points, sampling_spacing_mm)[0]
    arc = cumulative_arc_length(points)
    total = float(arc[-1])
    if total <= 0.0:
        return points.copy()
    u = arc / total
    weights = np.ones(points.shape[0], dtype=float)
    weights[0] = weights[-1] = 50.0
    k = min(3, points.shape[0] - 1)
    smoothing_budget = float(smoothing_mm) ** 2 * points.shape[0]
    try:
        spline, _ = splprep(
            points.T,
            u=u,
            w=weights,
            s=smoothing_budget,
            k=k,
            per=False,
        )
        sample_arc = np.arange(0.0, total + 0.5 * sampling_spacing_mm, sampling_spacing_mm)
        if total - sample_arc[-1] > 1e-9:
            sample_arc = np.r_[sample_arc, total]
        sample_u = np.clip(sample_arc / total, 0.0, 1.0)
        smoothed = np.asarray(splev(sample_u, spline)).T
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
        return remove_duplicate_points(smoothed)
    except Exception:
        return resample_polyline(points, sampling_spacing_mm)[0]


def curve_tangents(points: np.ndarray) -> np.ndarray:
    points = remove_duplicate_points(points)
    if points.shape[0] < 2:
        raise ValueError("At least two curve points are required")
    differences = np.empty_like(points)
    differences[0] = points[1] - points[0]
    differences[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        differences[1:-1] = points[2:] - points[:-2]
    tangents = np.empty_like(differences)
    previous = normalize(differences[0], fallback=np.array([0.0, 0.0, 1.0]))
    for index, difference in enumerate(differences):
        tangents[index] = normalize(difference, fallback=previous)
        previous = tangents[index]
    return tangents


def _least_aligned_world_axis(tangent: np.ndarray) -> np.ndarray:
    basis = np.eye(3, dtype=float)
    return basis[int(np.argmin(np.abs(basis @ normalize(tangent))))]


def parallel_transport_frames(
    points_world: np.ndarray,
    initial_normal_world: Optional[Sequence[float]] = None,
) -> FrameResult:
    points = remove_duplicate_points(points_world)
    tangents = curve_tangents(points)
    arc = cumulative_arc_length(points)

    if initial_normal_world is None:
        initial = _least_aligned_world_axis(tangents[0])
    else:
        initial = np.asarray(initial_normal_world, dtype=float)
    initial = initial - tangents[0] * np.dot(initial, tangents[0])
    if np.linalg.norm(initial) <= 1e-10:
        initial = _least_aligned_world_axis(tangents[0])
        initial = initial - tangents[0] * np.dot(initial, tangents[0])
    x_axes = np.empty_like(points)
    y_axes = np.empty_like(points)
    x_axes[0] = normalize(initial)
    y_axes[0] = normalize(np.cross(tangents[0], x_axes[0]))
    x_axes[0] = normalize(np.cross(y_axes[0], tangents[0]))

    for index in range(1, points.shape[0]):
        previous_tangent = tangents[index - 1]
        tangent = tangents[index]
        cross = np.cross(previous_tangent, tangent)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(previous_tangent, tangent), -1.0, 1.0))
        previous_x = x_axes[index - 1]

        if sine <= 1e-10:
            transported = previous_x.copy()
            if cosine < 0.0:
                transported *= -1.0
        else:
            axis = cross / sine
            angle = math.atan2(sine, cosine)
            transported = (
                previous_x * math.cos(angle)
                + np.cross(axis, previous_x) * math.sin(angle)
                + axis * np.dot(axis, previous_x) * (1.0 - math.cos(angle))
            )
        transported -= tangent * np.dot(transported, tangent)
        if np.linalg.norm(transported) <= 1e-10:
            transported = _least_aligned_world_axis(tangent)
            transported -= tangent * np.dot(transported, tangent)
        x_axes[index] = normalize(transported)
        # Prevent a numerical sign flip in nearly degenerate steps.
        if np.dot(x_axes[index], x_axes[index - 1]) < 0.0:
            x_axes[index] *= -1.0
        y_axes[index] = normalize(np.cross(tangent, x_axes[index]))
        x_axes[index] = normalize(np.cross(y_axes[index], tangent))

    return FrameResult(
        centers_world=points,
        x_axes_world=x_axes,
        y_axes_world=y_axes,
        tangents_world=tangents,
        arc_lengths_mm=arc,
    )


def sample_world_points(
    array: np.ndarray,
    array_to_world: np.ndarray,
    points_world: np.ndarray,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    world_to_array = np.linalg.inv(_as_float_matrix(array_to_world))
    array_points = apply_affine(points_world, world_to_array)
    return ndimage.map_coordinates(
        np.asarray(array),
        array_points.T,
        order=int(order),
        mode="constant",
        cval=float(cval),
        prefilter=int(order) > 1,
    )


def _component_near_image_center(binary_slice: np.ndarray) -> Optional[np.ndarray]:
    binary_slice = np.asarray(binary_slice, dtype=bool)
    labels, count = ndimage.label(binary_slice, structure=ndimage.generate_binary_structure(2, 2))
    if count == 0:
        return None
    center = (np.asarray(binary_slice.shape) - 1) / 2.0
    center_index = tuple(np.rint(center).astype(int))
    label_id = int(labels[center_index])
    if label_id == 0:
        foreground = np.argwhere(binary_slice)
        nearest = int(np.argmin(np.linalg.norm(foreground - center, axis=1)))
        label_id = int(labels[tuple(foreground[nearest])])
    return labels == label_id


def refine_centerline_cross_sections(
    mask: np.ndarray,
    array_to_world: np.ndarray,
    points_world: np.ndarray,
    initial_normal_world: np.ndarray,
    fov_mm: float,
    iterations: int = 1,
    pixel_spacing_mm: Optional[float] = None,
    maximum_shift_mm: float = 2.0,
    smoothing_mm: float = 0.75,
) -> np.ndarray:
    """Refine a centerline toward distance-weighted section centroids."""
    points = np.asarray(points_world, dtype=float).copy()
    if iterations <= 0 or points.shape[0] < 5:
        return points
    voxel_spacing = array_axis_spacings(array_to_world)
    pixel = float(pixel_spacing_mm or max(float(voxel_spacing.min()), 0.35))
    side_pixels = int(np.clip(math.ceil(float(fov_mm) / pixel), 17, 257))
    if side_pixels % 2 == 0:
        side_pixels += 1
    offsets = (np.arange(side_pixels, dtype=float) - 0.5 * (side_pixels - 1)) * pixel
    vv, uu = np.meshgrid(offsets, offsets, indexing="ij")
    curve_spacing = max(float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1))), 0.25)

    for _ in range(int(iterations)):
        frames = parallel_transport_frames(points, initial_normal_world)
        updated = points.copy()
        for index in range(2, points.shape[0] - 2):
            world = (
                frames.centers_world[index][None, None, :]
                + uu[..., None] * frames.x_axes_world[index][None, None, :]
                + vv[..., None] * frames.y_axes_world[index][None, None, :]
            )
            section = sample_world_points(
                mask.astype(np.uint8), array_to_world, world.reshape(-1, 3), order=0, cval=0
            ).reshape(side_pixels, side_pixels) > 0
            component = _component_near_image_center(section)
            if component is None or component.sum() < 4:
                continue
            distance = ndimage.distance_transform_edt(component)
            weights = distance ** 2
            total = float(weights.sum())
            if total <= 0.0:
                weights = component.astype(float)
                total = float(weights.sum())
            row = float((weights * np.arange(side_pixels)[:, None]).sum() / total)
            column = float((weights * np.arange(side_pixels)[None, :]).sum() / total)
            shift_u = (column - 0.5 * (side_pixels - 1)) * pixel
            shift_v = (row - 0.5 * (side_pixels - 1)) * pixel
            shift = shift_u * frames.x_axes_world[index] + shift_v * frames.y_axes_world[index]
            shift_norm = float(np.linalg.norm(shift))
            if shift_norm > maximum_shift_mm:
                shift *= maximum_shift_mm / shift_norm
            candidate = points[index] + shift
            inside = sample_world_points(
                mask.astype(np.uint8), array_to_world, candidate[None, :], order=0, cval=0
            )[0]
            if inside > 0:
                updated[index] = candidate
        points = smooth_curve(updated, curve_spacing, smoothing_mm)
    return points


def _recommended_fov(
    mask_world_points: np.ndarray,
    centerline_world: np.ndarray,
    margin_mm: float,
) -> float:
    if mask_world_points.shape[0] > 300_000:
        stride = int(math.ceil(mask_world_points.shape[0] / 300_000.0))
        mask_world_points = mask_world_points[::stride]
    tree = cKDTree(centerline_world)
    distances, _ = tree.query(mask_world_points, k=1)
    radius = float(np.percentile(distances, 99.9)) + float(margin_mm)
    return max(4.0, float(math.ceil(2.0 * radius)))


def recommend_field_of_view(
    mask: np.ndarray,
    array_to_world: np.ndarray,
    centerline_world: np.ndarray,
    margin_mm: float = 3.0,
    maximum_samples: int = 300_000,
) -> float:
    """Recommend a square slice field of view that encloses the tooth mask."""
    mask = np.asarray(mask, dtype=bool)
    flat_indices = np.flatnonzero(mask)
    if flat_indices.size == 0:
        raise ValueError("Cannot estimate field of view from an empty mask")
    if flat_indices.size > int(maximum_samples):
        stride = int(math.ceil(flat_indices.size / float(maximum_samples)))
        flat_indices = flat_indices[::stride]
    coordinates = np.column_stack(np.unravel_index(flat_indices, mask.shape))
    world = apply_affine(coordinates, array_to_world)
    return _recommended_fov(world, np.asarray(centerline_world, dtype=float), margin_mm)


def _control_points_from_curve(points_world: np.ndarray, spacing_mm: float) -> np.ndarray:
    return resample_polyline(points_world, max(float(spacing_mm), 0.1))[0]


def _curve_qc(
    mask: np.ndarray,
    array_to_world: np.ndarray,
    points_world: np.ndarray,
    clearance_mm: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    inside = sample_world_points(
        mask.astype(np.uint8), array_to_world, points_world, order=0, cval=0
    ) > 0
    inside_fraction = float(np.mean(inside))
    arc = cumulative_arc_length(points_world)
    length = float(arc[-1])
    chord = float(np.linalg.norm(points_world[-1] - points_world[0]))
    tangents = curve_tangents(points_world)
    segment_lengths = np.maximum(np.linalg.norm(np.diff(points_world, axis=0), axis=1), 1e-9)
    tangent_angles = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0))
    curvature = tangent_angles / segment_lengths

    qc: Dict[str, object] = {
        "inside_fraction": inside_fraction,
        "curve_length_mm": length,
        "endpoint_distance_mm": chord,
        "tortuosity": length / max(chord, 1e-9),
        "maximum_curvature_per_mm": float(curvature.max()) if curvature.size else 0.0,
        "median_curvature_per_mm": float(np.median(curvature)) if curvature.size else 0.0,
        "sample_count": int(points_world.shape[0]),
    }
    if clearance_mm is not None:
        sampled_clearance = sample_world_points(
            clearance_mm, array_to_world, points_world, order=1, cval=0.0
        )
        qc.update(
            {
                "minimum_clearance_mm": float(sampled_clearance.min()),
                "median_clearance_mm": float(np.median(sampled_clearance)),
                "maximum_clearance_mm": float(sampled_clearance.max()),
            }
        )

    if inside_fraction < 0.98:
        warnings.append(
            f"Only {inside_fraction:.1%} of centerline samples lie inside the solid tooth mask."
        )
    if qc["tortuosity"] > 3.0:
        warnings.append("The fitted centerline is unusually tortuous; inspect it before export.")
    if qc["maximum_curvature_per_mm"] > 1.0:
        warnings.append("A sharp local centerline turn was detected; inspect that location.")
    return qc, warnings


def estimate_tooth_axis(
    axis_mask: np.ndarray,
    array_to_world: np.ndarray,
    coarse_spacing_mm: float = 1.0,
    centerline_strength: float = 12.0,
    centerline_exponent: float = 2.0,
    smoothing_mm: float = 1.0,
    output_spacing_mm: float = 0.5,
    control_point_spacing_mm: float = 3.0,
    endpoint_cap_fraction: float = 0.06,
    cross_section_refinement_iterations: int = 1,
    fov_margin_mm: float = 3.0,
    endpoint_world: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> AxisResult:
    """Estimate a smooth, non-branching tooth centerline.

    The path is computed through a conservative coarse mask with a Dijkstra
    cost that favours voxels far from the surface, then smoothed and optionally
    refined using orthogonal section centroids in the full-resolution mask.
    """
    mask = np.asarray(axis_mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("Axis mask must be a non-empty three-dimensional binary array")
    array_to_world = _as_float_matrix(array_to_world)
    full_spacing = array_axis_spacings(array_to_world)
    if coarse_spacing_mm <= 0.0 or output_spacing_mm <= 0.0 or control_point_spacing_mm <= 0.0:
        raise ValueError("All centerline sampling spacings must be positive")
    if centerline_strength < 0.0:
        raise ValueError("Centerline strength cannot be negative")
    if centerline_exponent <= 0.0:
        raise ValueError("Centerline exponent must be positive")
    if smoothing_mm < 0.0:
        raise ValueError("Smoothing cannot be negative")
    if not 0.0 < endpoint_cap_fraction < 0.5:
        raise ValueError("Endpoint cap fraction must be between 0 and 0.5")
    if cross_section_refinement_iterations < 0:
        raise ValueError("Refinement iteration count cannot be negative")
    if fov_margin_mm < 0.0:
        raise ValueError("Field-of-view margin cannot be negative")

    cropped, cropped_to_world, _ = crop_to_mask(mask, array_to_world, margin_mm=2.0)
    coarse, coarse_to_world, _ = block_reduce_max(
        cropped, cropped_to_world, target_spacing_mm=coarse_spacing_mm
    )
    coarse, _, _ = select_connected_component(coarse, array_axis_spacings(coarse_to_world))
    coarse_spacing = array_axis_spacings(coarse_to_world)
    clearance = ndimage.distance_transform_edt(coarse, sampling=coarse_spacing)
    coordinates = np.argwhere(coarse)
    world_coordinates = apply_affine(coordinates, coarse_to_world)
    pca_center, pca_axes, _ = principal_axes(world_coordinates)
    projections = (world_coordinates - pca_center) @ pca_axes[:, 0]
    clearance_values = clearance[tuple(coordinates.T)]

    if endpoint_world is None:
        start = _choose_endpoint(
            coordinates, projections, clearance_values, True, endpoint_cap_fraction
        )
        goal = _choose_endpoint(
            coordinates, projections, clearance_values, False, endpoint_cap_fraction
        )
    else:
        coarse_world_to_array = np.linalg.inv(coarse_to_world)
        desired_start = apply_affine(endpoint_world[0], coarse_world_to_array)
        desired_goal = apply_affine(endpoint_world[1], coarse_world_to_array)
        start_nearest = _nearest_foreground_index(
            coarse, desired_start, coarse_spacing, maximum_distance_mm=10.0
        )
        goal_nearest = _nearest_foreground_index(
            coarse, desired_goal, coarse_spacing, maximum_distance_mm=10.0
        )
        if start_nearest is None or goal_nearest is None:
            raise ValueError("A supplied endpoint could not be projected into the tooth mask")
        start = np.asarray(start_nearest, dtype=float)
        goal = np.asarray(goal_nearest, dtype=float)

    if np.array_equal(start.astype(int), goal.astype(int)):
        raise ValueError(
            "The two centerline endpoints resolve to the same voxel; use a larger tooth mask or distinct endpoints"
        )

    raw_path_indices = center_biased_shortest_path(
        coarse,
        clearance,
        start.astype(int),
        goal.astype(int),
        coarse_spacing,
        centerline_strength=centerline_strength,
        centerline_exponent=centerline_exponent,
    )
    raw_path_world = apply_affine(raw_path_indices, coarse_to_world)
    preliminary = smooth_curve(
        raw_path_world,
        sampling_spacing_mm=output_spacing_mm,
        smoothing_mm=smoothing_mm,
    )

    # PCA secondary axis gives a deterministic initial in-plane orientation.
    initial_normal = pca_axes[:, 1].copy()
    initial_tangent = normalize(preliminary[1] - preliminary[0])
    initial_normal = initial_normal - initial_tangent * np.dot(initial_normal, initial_tangent)
    if np.linalg.norm(initial_normal) <= 1e-10:
        initial_normal = _least_aligned_world_axis(initial_tangent)
        initial_normal -= initial_tangent * np.dot(initial_normal, initial_tangent)
    initial_normal = canonicalize_vector_sign(initial_normal)

    recommended_fov = _recommended_fov(
        world_coordinates, preliminary, margin_mm=fov_margin_mm
    )
    refined = refine_centerline_cross_sections(
        mask,
        array_to_world,
        preliminary,
        initial_normal,
        fov_mm=recommended_fov,
        iterations=cross_section_refinement_iterations,
        pixel_spacing_mm=max(float(full_spacing.min()), 0.35),
        maximum_shift_mm=max(1.0, 2.0 * float(full_spacing.max())),
        smoothing_mm=max(0.5, smoothing_mm * 0.75),
    )
    refined = smooth_curve(refined, output_spacing_mm, smoothing_mm * 0.5)

    full_clearance = ndimage.distance_transform_edt(mask, sampling=full_spacing)
    qc, warnings = _curve_qc(mask, array_to_world, refined, full_clearance)
    control_points = _control_points_from_curve(refined, control_point_spacing_mm)

    qc.update(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "coarse_shape_zyx": [int(v) for v in coarse.shape],
            "coarse_spacing_zyx_mm": [float(v) for v in coarse_spacing],
            "recommended_fov_mm": recommended_fov,
        }
    )

    return AxisResult(
        points_world=refined,
        raw_path_world=raw_path_world,
        control_points_world=control_points,
        pca_center_world=pca_center,
        pca_axes_world=pca_axes,
        initial_normal_world=initial_normal,
        recommended_fov_mm=recommended_fov,
        qc=qc,
        warnings=warnings,
    )


def sample_volume_along_frames(
    volume: np.ndarray,
    array_to_world: np.ndarray,
    frames: FrameResult,
    pixel_spacing_mm: float,
    field_of_view_mm: float,
    interpolation_order: int = 1,
    outside_value: Optional[float] = None,
    output_dtype: np.dtype = np.float32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a volume on the planes of a rotation-minimizing frame.

    Returns ``(stack, u_coordinates_mm, v_coordinates_mm)``.  The stack shape
    is ``(number_of_slices, rows_v, columns_u)``.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("Input volume must be three-dimensional")
    pixel = float(pixel_spacing_mm)
    fov = float(field_of_view_mm)
    if pixel <= 0.0 or fov <= 0.0:
        raise ValueError("Pixel spacing and field of view must be positive")
    if int(interpolation_order) < 0 or int(interpolation_order) > 5:
        raise ValueError("Interpolation order must be between 0 and 5")
    side = max(3, int(math.ceil(fov / pixel)) + 1)
    if side % 2 == 0:
        side += 1
    coordinates = (np.arange(side, dtype=float) - 0.5 * (side - 1)) * pixel
    vv, uu = np.meshgrid(coordinates, coordinates, indexing="ij")
    output = np.empty(
        (frames.centers_world.shape[0], side, side), dtype=np.dtype(output_dtype)
    )
    world_to_array = np.linalg.inv(_as_float_matrix(array_to_world))
    if outside_value is None:
        finite = _sample_values(volume, maximum_samples=250_000)
        outside_value = float(np.min(finite))

    progress_step = max(1, frames.centers_world.shape[0] // 100)
    for index in range(frames.centers_world.shape[0]):
        world = (
            frames.centers_world[index][None, None, :]
            + uu[..., None] * frames.x_axes_world[index][None, None, :]
            + vv[..., None] * frames.y_axes_world[index][None, None, :]
        )
        array_points = apply_affine(world.reshape(-1, 3), world_to_array)
        sampled = ndimage.map_coordinates(
            volume,
            array_points.T,
            order=int(interpolation_order),
            mode="constant",
            cval=float(outside_value),
            prefilter=int(interpolation_order) > 1,
        ).reshape(side, side)
        output[index] = sampled.astype(output.dtype, copy=False)
        if progress_callback is not None and (
            index % progress_step == 0 or index == frames.centers_world.shape[0] - 1
        ):
            progress_callback(index + 1, frames.centers_world.shape[0])
    return output, coordinates.copy(), coordinates.copy()


def frame_matrices(frames: FrameResult) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=float)[None, :, :], frames.centers_world.shape[0], axis=0)
    matrices[:, :3, 0] = frames.x_axes_world
    matrices[:, :3, 1] = frames.y_axes_world
    matrices[:, :3, 2] = frames.tangents_world
    matrices[:, :3, 3] = frames.centers_world
    return matrices


def validate_frames(frames: FrameResult, tolerance: float = 1e-6) -> Dict[str, float]:
    x = frames.x_axes_world
    y = frames.y_axes_world
    z = frames.tangents_world
    norms = np.r_[np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1), np.linalg.norm(z, axis=1)]
    orthogonality = np.r_[
        np.sum(x * y, axis=1), np.sum(x * z, axis=1), np.sum(y * z, axis=1)
    ]
    handedness = np.sum(np.cross(x, y) * z, axis=1)
    continuity = np.sum(x[:-1] * x[1:], axis=1) if x.shape[0] > 1 else np.array([1.0])
    result = {
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "maximum_orthogonality_error": float(np.max(np.abs(orthogonality))),
        "minimum_handedness": float(np.min(handedness)),
        "minimum_x_axis_continuity": float(np.min(continuity)),
    }
    if result["maximum_norm_error"] > tolerance:
        raise ValueError("Frame axes are not unit length")
    if result["maximum_orthogonality_error"] > tolerance:
        raise ValueError("Frame axes are not orthogonal")
    if result["minimum_handedness"] < 1.0 - tolerance:
        raise ValueError("Frame is not right-handed")
    if result["minimum_x_axis_continuity"] < -tolerance:
        raise ValueError("Frame contains an in-plane sign flip")
    return result
