# Synthetic validation

## Purpose

The regression suite verifies geometry and software behavior under known ground truth. It is not a validation of a biological definition of tooth growth axis.

## Phantom

The primary test generates a long curved tube with:

- approximately 62 mm longitudinal extent;
- varying outer radius and a central cavity;
- 0.55, 0.65, and 0.75 mm anisotropic voxel spacing;
- a rotated array-to-world affine;
- high-density tissue and noisy low-density background.

The cavity is retained in the anatomical truth mask and filled in the axis-estimation copy.

## Current regression thresholds

The test requires:

- segmentation intersection-over-union greater than 0.985;
- mean symmetric centerline distance less than 1.25 mm;
- 95th-percentile symmetric distance less than 2.6 mm;
- greater than 98% of centerline samples inside the solid mask;
- orthonormal, right-handed frames with no local sign flips;
- centered source sampling under anisotropic and rotated geometry;
- bit-exact repeatability for repeated axis estimation;
- TIFF re-read values and row orientation equal to the source array;
- ROI-specific Otsu estimation and exact world-bounds enforcement under a rotated affine;
- import and selected logic methods under a minimal mocked Slicer runtime, including transformed KJI-to-RAS geometry and non-displayable table nodes.

On the development run used for version 0.1.0, mean symmetric centerline distance was approximately **0.37 mm** and the 95th percentile approximately **0.60 mm** after excluding a short rounded terminal cap from the ground-truth comparison. The maximum endpoint-directed error is deliberately not used as the main score because a geometric medial path terminates inside rounded end caps.

## Running

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q ToothOrthogonalSlices/Testing/Python
```

A separate in-Slicer generic test checks basic frame construction and resampling. The regular Python suite currently contains eleven tests. Full UI and MRML export behavior should additionally be exercised in the target Slicer release before a production release.
