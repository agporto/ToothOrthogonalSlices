# Changelog

## 0.1.0 — 2026-08-23

- Initial standalone 3D Slicer scripted extension.
- Automatic Otsu or manual threshold segmentation with component seed and ROI support.
- Existing-segmentation workflow for manually corrected or externally generated masks.
- Center-biased, nonbranching path estimation through a solidified tooth envelope.
- Smooth editable Markups curve and optional user-defined endpoints.
- Rotation-minimizing parallel-transport frames for stable in-plane orientation.
- Quantitative CT and mask TIFF export, preview PNG export, and straightened NRRD volumes.
- Per-slice RAS matrices in CSV and compressed NumPy formats.
- Quality-control flags for clipping, empty or disconnected masks, abrupt area changes, centroid offsets, and high-curvature overlap risk.
- Pure-Python synthetic validation suite and GitHub Actions workflow.
- ROI-local Otsu estimation with exact world-space clipping for transformed volumes.
- Atomic quantitative image/metadata output and incomplete-export detection.
- Mocked-Slicer compatibility tests for volume geometry, QC, and table-node handling.
