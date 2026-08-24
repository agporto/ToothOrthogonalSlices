# Contributing

Bug reports should include the 3D Slicer version, operating system, module version, relevant settings, and the full Python console traceback. Do not attach identifiable clinical data.

Before opening a pull request:

1. Run `python -m pytest -q ToothOrthogonalSlices/Testing/Python` in a Python environment containing NumPy, SciPy, VTK, and pytest.
2. Run `python -m compileall -q ToothOrthogonalSlices`.
3. Test the complete two-stage workflow in 3D Slicer on at least one isolated tooth volume and one corrected existing segmentation.
4. Verify that exported TIFF orientation agrees with the straightened NRRD volume and that `slice_frames.csv` maps sampled centers back to the source volume.

Keep numerical algorithms in `ToothAxisAlgorithms.py`, file-format helpers in `ToothSliceIO.py`, and Slicer/MRML integration in `ToothOrthogonalSlices.py`.
