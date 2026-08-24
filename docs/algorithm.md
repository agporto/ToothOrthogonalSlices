# Algorithm and coordinate conventions

## Terminology

The module estimates a **geometric longitudinal centerline** through a binary tooth envelope. “Axis” in the code and output metadata means this operational geometric centerline. Whether it corresponds to a biological growth trajectory is a separate anatomical hypothesis.

## Coordinate systems

- Input arrays use NumPy `(z, y, x)` or Slicer KJI order.
- All curves, frames, distances, and exported matrices use Slicer world **RAS millimetres**.
- `array_to_world` maps homogeneous `[z, y, x, 1]` coordinates to `[R, A, S, 1]`.
- Each exported plane-to-RAS matrix has columns `[x_axis, y_axis, tangent, center]`.

For pixel coordinates `(column, row)` with centered physical coordinates `(u, v)`, the source point is

\[
p(u,v,s_i) = c(s_i) + u\,e_{x,i} + v\,e_{y,i},
\]

where `c(s_i)` is the curve center and the plane normal is the local tangent `t_i`.

## Segmentation

Automatic segmentation computes a robust Otsu lower threshold over a bounded scalar sample, closes small gaps, and labels 26-connected components. When an ROI is supplied, both threshold estimation and candidate selection are restricted to voxel centers inside its exact world-axis-aligned RAS bounds; this prevents unrelated structures elsewhere in the scan from biasing Otsu. If a valid seed is supplied, the selected component contains the nearest thresholded voxel to that seed; otherwise the largest component is used.

Two masks are maintained:

- **anatomical mask** — thresholded or user-supplied segment, preserved for quantitative reslicing;
- **axis mask** — a copy optionally filled and closed so internal cavities do not induce a branching medial path.

## Centerline path

The solid axis mask is cropped and conservatively downsampled by block maximum. PCA supplies a longitudinal initialization. Endpoint candidates are selected in opposite PCA end caps, favoring voxels with large distance-transform clearance.

For an interior voxel `q`, normalized clearance is

\[
d_n(q) = \frac{D(q)}{\max_{p \in M} D(p)},
\]

where `D` is the Euclidean distance to the mask boundary. The traversal cost field is

\[
w(q) = 1 + \alpha\left(1-d_n(q)\right)^\gamma,
\]

and each 26-neighbor edge uses physical step length multiplied by the mean endpoint cost. Dijkstra’s algorithm returns a single nonbranching minimum-cost path between the selected ends.

The path is converted to world coordinates, spline-smoothed, and optionally refined. During refinement, temporary orthogonal sections of the full-resolution solid mask are sampled; each intermediate point is shifted toward a distance-transform-weighted section centroid, constrained to remain inside the mask, and smoothed again.

## Frame construction

The module does not use a Frenet frame because Frenet normals can be undefined at low curvature and can flip at inflections. It constructs a rotation-minimizing frame by parallel transporting the first in-plane axis between consecutive tangents with the smallest 3-D rotation. Reprojection and Gram–Schmidt cleanup maintain an orthonormal, right-handed basis.

The initial in-plane direction comes from, in priority order:

1. a user orientation line or fiducial;
2. the stored PCA-derived direction from axis estimation;
3. the world basis axis least aligned with the first tangent.

The **Rotate in-plane orientation by 180°** option reverses both in-plane axes while preserving a right-handed frame. The remaining frames inherit orientation by parallel transport. A one-axis image reflection would make the spatial frame left-handed, so the module does not label this operation as a mirror.

## Resampling

- CT: SciPy `map_coordinates`, linear (order 1) or cubic (order 3).
- Mask: nearest-neighbor (order 0).
- CT output: float32. All signed int16 input values are exactly representable before interpolation.
- Individual slices share a fixed square pixel grid and field of view.

## Quality control

Per-slice flags include:

- empty mask;
- multiple 2-D connected components;
- contact with field-of-view boundary;
- mask centroid displaced from the sampled axis;
- high local curvature relative to half the field of view.
- abrupt adjacent cross-sectional area change.

The run summary also reports frame orthonormality diagnostics. These are screening flags, not automatic proof that a curve is anatomically correct.

## Export integrity

Individual TIFF and PNG files, frame metadata, JSON documents, and saved Slicer nodes are written through temporary paths and atomically replaced where the platform permits it. `manifest.json` is written last. A `.export_incomplete.json` marker remains in the output folder if an export terminates before completion, preventing an older manifest from making a partial run appear complete.
