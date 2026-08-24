"""File-output helpers for Tooth Orthogonal Slices.

The functions in this file depend only on NumPy, SciPy, and VTK.  Keeping them
outside the Slicer widget makes TIFF/PNG orientation and scalar preservation
independently testable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import vtk
from scipy import ndimage
from vtk.util.numpy_support import numpy_to_vtk


_MANAGED_SLICE_RE = re.compile(r"^(?:slice|mask)_\d{4,8}\.(?:tif|tiff|png)$", re.IGNORECASE)


def safe_filename(value: str, fallback: str = "tooth") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return cleaned or fallback


PathLike = Union[os.PathLike, str]


def clean_managed_slice_files(folder: PathLike) -> None:
    """Delete only files created by this module, never arbitrary user files."""
    path = Path(folder)
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_file() and _MANAGED_SLICE_RE.match(child.name):
            child.unlink()


def _vtk_scalar_type(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return vtk.VTK_UNSIGNED_CHAR
    if dtype == np.uint16:
        return vtk.VTK_UNSIGNED_SHORT
    if dtype == np.float32:
        return vtk.VTK_FLOAT
    if dtype == np.float64:
        return vtk.VTK_DOUBLE
    raise TypeError(f"Unsupported VTK image scalar dtype: {dtype}")


def _vtk_image_from_array_2d(array: np.ndarray, components: int = 1) -> vtk.vtkImageData:
    array = np.asarray(array)
    if components == 1:
        if array.ndim != 2:
            raise ValueError("Scalar image must have shape (rows, columns)")
        rows, columns = array.shape
    else:
        if array.ndim != 3 or array.shape[2] != components:
            raise ValueError(f"Color image must have shape (rows, columns, {components})")
        rows, columns, _ = array.shape

    # vtkImageWriter uses a lower-left image origin while NumPy image arrays
    # conventionally use upper-left.  Pre-flipping makes a standard TIFF/PNG
    # reader return the same row order supplied by the caller.
    oriented = np.ascontiguousarray(np.flipud(array))
    image = vtk.vtkImageData()
    image.SetDimensions(int(columns), int(rows), 1)
    image.SetExtent(0, int(columns) - 1, 0, int(rows) - 1, 0, 0)
    vtk_array = numpy_to_vtk(
        oriented.reshape(-1, components) if components > 1 else oriented.ravel(),
        deep=True,
        array_type=_vtk_scalar_type(oriented.dtype),
    )
    vtk_array.SetNumberOfComponents(int(components))
    image.GetPointData().SetScalars(vtk_array)
    return image


def _temporary_image_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.writing{path.suffix}")


def write_scalar_tiff(path: PathLike, array: np.ndarray) -> None:
    """Write a quantitative 2-D TIFF without display windowing.

    Signed integer CT arrays are converted to float32, which represents all
    int16 values exactly and is supported consistently by VTK's TIFF writer.
    """
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("TIFF slice must be two-dimensional")
    if np.issubdtype(values.dtype, np.bool_) or values.dtype == np.uint8:
        output = values.astype(np.uint8, copy=False)
    elif values.dtype == np.uint16:
        output = values
    else:
        output = values.astype(np.float32, copy=False)
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_image_path(final_path)
    temporary_path.unlink(missing_ok=True)
    image = _vtk_image_from_array_2d(output, components=1)
    writer = vtk.vtkTIFFWriter()
    writer.SetFileName(str(temporary_path))
    writer.SetInputData(image)
    # PackBits is lossless and broadly available; unlike LZW it does not
    # depend on optional libtiff codec support in a particular Slicer build.
    if hasattr(writer, "SetCompressionToPackBits"):
        writer.SetCompressionToPackBits()
    elif hasattr(writer, "SetCompressionToNoCompression"):
        writer.SetCompressionToNoCompression()
    writer.Write()
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise IOError(f"Failed to write TIFF: {final_path}")
    os.replace(str(temporary_path), str(final_path))


def write_rgb_png(path: PathLike, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb, dtype=np.uint8)
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_image_path(final_path)
    temporary_path.unlink(missing_ok=True)
    image = _vtk_image_from_array_2d(rgb, components=3)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(temporary_path))
    writer.SetInputData(image)
    writer.Write()
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise IOError(f"Failed to write PNG: {final_path}")
    os.replace(str(temporary_path), str(final_path))


def robust_window(
    stack: np.ndarray,
    mask_stack: Optional[np.ndarray] = None,
    percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[float, float]:
    values = np.asarray(stack)
    if mask_stack is not None and np.any(mask_stack):
        selected = values[np.asarray(mask_stack, dtype=bool)]
    else:
        selected = values[np.isfinite(values)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(selected, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(selected))
        high = float(np.max(selected))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def make_preview_rgb(
    scalar_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    window: Sequence[float],
) -> np.ndarray:
    low, high = float(window[0]), float(window[1])
    if high <= low:
        raise ValueError("Preview window maximum must exceed minimum")
    gray = np.clip((np.asarray(scalar_slice, dtype=float) - low) / (high - low), 0.0, 1.0)
    gray = np.rint(gray * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask_slice is not None:
        mask = np.asarray(mask_slice, dtype=bool)
        boundary = mask & ~ndimage.binary_erosion(mask)
        # A high-contrast orange-red boundary is used only in preview PNGs;
        # quantitative TIFF and NRRD data are not altered.
        rgb[boundary, 0] = 255
        rgb[boundary, 1] = 96
        rgb[boundary, 2] = 32
    return rgb
