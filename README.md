# Slicer Tooth Orthogonal Slices

A standalone 3D Slicer scripted extension for producing reproducible cross-sections through long, curved teeth. It can threshold an isolated high-density tooth automatically, accept an existing corrected segmentation, estimate a smooth editable longitudinal centerline, and resample the original CT on planes orthogonal to the local curve tangent.

> **Scientific interpretation:** the automatic curve is a geometric longitudinal/centroidal axis of the segmented tooth envelope. It is not automatically a histological or developmental “growth axis.” Validate the curve against the anatomy and study design before treating it as one.

## Main workflow

1. Select the original scalar CT volume.
2. Either leave **Tooth segmentation** empty for automatic thresholding, or select an existing segmentation and segment.
3. Optionally place a component seed, threshold ROI, or two endpoint fiducials.
4. Click **Estimate segmentation and axis**.
5. Inspect the segmentation and the generated Markups curve in 2D and 3D. Correct the segmentation in Segment Editor or drag curve control points as needed.
6. Choose slice spacing, pixel size, field of view, direction, and export options.
7. Click **Generate and export orthogonal slices**.

The workflow intentionally separates estimation from export so that automatic results can be reviewed rather than silently accepted.

## Outputs

For an input volume named `HorseTooth01`, the selected parent directory receives:

```text
HorseTooth01_orthogonal_slices/
├── manifest.json
├── qc.json
├── slice_frames.csv
├── slice_frames.npz
├── tooth_axis.mrk.json
├── tooth_segmentation.seg.nrrd
├── straightened_ct.nrrd
├── straightened_mask.nrrd
├── ct_slices/
│   ├── slice_0000.tif
│   └── ...
├── mask_slices/
│   ├── mask_0000.tif
│   └── ...
└── preview_slices/
    ├── slice_0000.png
    └── ...
```

The CT TIFFs contain quantitative float32 resampled intensities and are not windowed. Signed 16-bit CT values are represented exactly in float32. Mask TIFFs contain `0` and `1`. Preview PNGs are display-windowed and contain an orange-red mask outline; they are not quantitative.

`slice_frames.csv` and `slice_frames.npz` retain, for every slice:

- center in source Slicer RAS coordinates;
- two transported in-plane unit axes;
- local tangent/plane normal;
- complete 4×4 plane-to-RAS matrix;
- arc-length position;
- mask area and quality-control flags;
- relative image filenames.

The straightened volumes use local rectified coordinates: I and J are transported cross-sectional directions and K follows arc length. Because a curved reconstruction has no single source-space affine transform, use the per-slice matrices to map it back to the original CT.

## Algorithm

The default automatic path is:

1. Robust Otsu thresholding (or a manual lower threshold).
2. Optional world-space ROI restriction.
3. Selection of the seed-containing or largest 26-connected component.
4. Preservation of an anatomical mask plus creation of a hole-filled/closed copy used only for axis estimation.
5. PCA initialization and automatic endpoint selection in opposite longitudinal end caps.
6. A 26-neighbor Dijkstra path whose cost penalizes proximity to the tooth surface using the Euclidean distance transform.
7. Cubic spline smoothing and optional iterative recentering from orthogonal section centroids.
8. User-editable Markups curve creation.
9. Equal arc-length resampling and rotation-minimizing parallel transport to prevent arbitrary slice roll/flips.
10. Linear or cubic CT interpolation and nearest-neighbor mask interpolation.

See [`docs/algorithm.md`](docs/algorithm.md) for definitions and implementation details.

## Installation for development

### Direct scripted-module installation

1. Clone this repository.
2. In 3D Slicer, open **Edit → Application Settings → Modules**.
3. Add the repository's `ToothOrthogonalSlices` subdirectory—the directory containing `ToothOrthogonalSlices.py`—to **Additional module paths**.
4. Restart Slicer and open **Tooth Orthogonal Slices** under the **SlicerMorph** category.

Slicer can also load the module by dragging `ToothOrthogonalSlices.py` into the application window and choosing **Add Python scripted module**.

### Build as a Slicer extension

Use the standard out-of-source Slicer extension build process:

```bash
mkdir build
cd build
cmake -DSlicer_DIR=/path/to/Slicer-build ../SlicerToothOrthogonalSlices
cmake --build . --config Release
```

## Validation and tests

The numerical core has no Slicer dependency and is tested with synthetic curved tubes under anisotropic voxel spacing and a rotated array-to-world affine. Tests cover:

- automatic segmentation and filled axis-mask behavior;
- seed/ROI component restriction;
- curved centerline recovery;
- deterministic results;
- frame orthonormality, right-handedness, and continuity;
- centered orthogonal reslicing;
- quantitative TIFF and PNG orientation round trips;
- ROI-local Otsu thresholding and exact transformed world-ROI clipping;
- selected Slicer-facing logic under a mocked MRML runtime.

Run locally with:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q ToothOrthogonalSlices/Testing/Python
```

The current synthetic benchmark recovers the known curved centerline with approximately **0.37 mm mean symmetric distance** for a 0.55–0.75 mm anisotropic test volume. This is a software regression test, not a biological accuracy claim. Details are in [`docs/validation.md`](docs/validation.md).

## Important limitations

- Thresholding cannot separate a tooth from mineralized anatomy to which it is physically connected. Use a restricted ROI, an existing/manual segmentation, or another segmentation method in that case.
- A branching multi-root tooth does not have a unique single axis. The module generates one nonbranching path; provide endpoints or edit the curve to select the intended route.
- Geometry alone cannot reliably identify root versus crown. Use **Reverse slice order** after anatomical review.
- Strong curvature can make neighboring source-space planes overlap. The module reports a curvature/FOV warning but does not remove duplicated source regions.
- Internal cavities are filled only in the centerline-estimation copy. Quantitative mask exports use the original anatomical segment.
- The module rejects volumes under nonlinear parent transforms. Harden the transform first so that voxel-to-world geometry is unambiguous.
- `manifest.json` is written last. If `.export_incomplete.json` remains in an output folder, that run did not finish and should not be treated as complete.

## Repository structure

```text
ToothOrthogonalSlices/
├── ToothOrthogonalSlices.py   # Slicer UI, MRML integration, export orchestration
├── ToothAxisAlgorithms.py     # tested numerical core, no Slicer dependency
├── ToothSliceIO.py            # tested TIFF/PNG helpers
├── Resources/Icons/
└── Testing/Python/
```

## License

BSD 2-Clause. See [`LICENSE`](LICENSE).
