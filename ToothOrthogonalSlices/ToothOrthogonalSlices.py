"""3D Slicer module for orthogonal cross-sections along a curved tooth axis."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ctk
import numpy as np
import qt
import slicer
import vtk
from scipy import ndimage
from vtk.util.numpy_support import vtk_to_numpy

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from ToothAxisAlgorithms import (
    ALGORITHM_VERSION,
    FrameResult,
    array_axis_spacings,
    automatic_tooth_segmentation,
    estimate_tooth_axis,
    frame_matrices,
    parallel_transport_frames,
    prepare_axis_mask,
    recommend_field_of_view,
    remove_duplicate_points,
    resample_polyline,
    sample_volume_along_frames,
    validate_frames,
)
from ToothSliceIO import (
    clean_managed_slice_files,
    make_preview_rgb,
    robust_window,
    safe_filename,
    write_rgb_png,
    write_scalar_tiff,
)


MODULE_VERSION = "0.1.0"
OUTPUT_ATTRIBUTE = "ToothOrthogonalSlices.OutputRole"


def _value(widget):
    value = getattr(widget, "value")
    return value() if callable(value) else value


def _checked(widget) -> bool:
    method = getattr(widget, "isChecked", None)
    if callable(method):
        return bool(method())
    return bool(getattr(widget, "checked", False))


def _text(widget) -> str:
    value = getattr(widget, "text")
    return str(value() if callable(value) else value).strip()


def _current_index(widget) -> int:
    value = getattr(widget, "currentIndex")
    return int(value() if callable(value) else value)


def _count(widget) -> int:
    value = getattr(widget, "count")
    return int(value() if callable(value) else value)


def _set_status(label, state: str, message: str) -> None:
    colors = {
        "needs_input": "#b94a48",
        "ready": "#31708f",
        "working": "#8a6d3b",
        "complete": "#3c763d",
        "warning": "#8a6d3b",
        "error": "#b94a48",
    }
    titles = {
        "needs_input": "Needs input",
        "ready": "Ready",
        "working": "Working",
        "complete": "Complete",
        "warning": "Review recommended",
        "error": "Error",
    }
    color = colors.get(state, colors["ready"])
    title = titles.get(state, state.replace("_", " ").title())
    safe = str(message)
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br>")
    label.setTextFormat(qt.Qt.RichText)
    label.setText(f'<span style="color:{color}; font-weight:600">{title}</span> — {safe}')
    label.setAccessibleName(f"{title}: {message}")


class ToothOrthogonalSlices(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Tooth Orthogonal Slices"
        self.parent.categories = ["SlicerMorph", "Dental"]
        self.parent.dependencies = ["Segmentations", "Markups"]
        self.parent.contributors = ["Arthur Porto (Florida Museum of Natural History, University of Florida)"]
        self.parent.helpText = (
            "Segments an isolated high-density tooth (or accepts an existing tooth segment), "
            "estimates a smooth geometric longitudinal axis, permits manual curve correction, "
            "and exports quantitative CT sections orthogonal to the local curved axis. "
            "The automatically inferred curve is a geometric centerline and should not be "
            "interpreted as a developmental growth trajectory without anatomical validation."
        )
        self.parent.acknowledgementText = (
            "Developed for reproducible cross-sectional analysis of elongated, curved teeth."
        )


class ToothOrthogonalSlicesWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = ToothOrthogonalSlicesLogic()
        self._build_ui()
        self.layout.addStretch(1)
        self._update_segment_list()
        self._update_enabled_state()

    def _node_selector(
        self,
        node_types: Sequence[str],
        none_enabled: bool = True,
        add_enabled: bool = False,
        rename_enabled: bool = True,
    ):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = list(node_types)
        selector.noneEnabled = bool(none_enabled)
        selector.addEnabled = bool(add_enabled)
        selector.removeEnabled = False
        selector.renameEnabled = bool(rename_enabled)
        selector.selectNodeUponCreation = True
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _double_spin(self, minimum, maximum, value, step, decimals=2, suffix=""):
        widget = qt.QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setValue(float(value))
        widget.setSingleStep(float(step))
        widget.setDecimals(int(decimals))
        if suffix:
            widget.setSuffix(suffix)
        return widget

    def _build_ui(self):
        intro = qt.QLabel(
            "Workflow: select a CT volume; optionally select an existing tooth segment; "
            "estimate and inspect the editable centerline; then generate/export orthogonal slices."
        )
        intro.setWordWrap(True)
        self.layout.addWidget(intro)

        inputs = ctk.ctkCollapsibleButton()
        inputs.text = "1. Inputs"
        inputs.collapsed = False
        self.layout.addWidget(inputs)
        form = qt.QFormLayout(inputs)

        self.inputVolumeSelector = self._node_selector(
            ["vtkMRMLScalarVolumeNode"], none_enabled=False, add_enabled=False
        )
        self.inputVolumeSelector.setToolTip("Original CT volume. Scalar intensities are preserved during export.")
        form.addRow("CT volume:", self.inputVolumeSelector)

        self.segmentationSelector = self._node_selector(
            ["vtkMRMLSegmentationNode"], none_enabled=True, add_enabled=False
        )
        self.segmentationSelector.setToolTip(
            "Optional existing tooth segmentation. Leave empty to threshold automatically."
        )
        form.addRow("Tooth segmentation (optional):", self.segmentationSelector)

        self.segmentCombo = qt.QComboBox()
        self.segmentCombo.setToolTip("Segment used as the tooth. Only needed when a segmentation is selected.")
        form.addRow("Tooth segment:", self.segmentCombo)

        self.seedSelector = self._node_selector(
            ["vtkMRMLMarkupsFiducialNode"], none_enabled=True, add_enabled=True
        )
        self.seedSelector.setToolTip(
            "Optional seed in the tooth. Recommended when multiple high-density objects are present."
        )
        form.addRow("Component seed (optional):", self.seedSelector)

        self.roiSelector = self._node_selector(
            ["vtkMRMLMarkupsROINode"], none_enabled=True, add_enabled=True
        )
        self.roiSelector.setToolTip(
            "Optional ROI limiting automatic thresholding. Its world-axis-aligned bounds are used."
        )
        form.addRow("Threshold ROI (optional):", self.roiSelector)

        self.endpointsSelector = self._node_selector(
            ["vtkMRMLMarkupsFiducialNode"], none_enabled=True, add_enabled=True
        )
        self.endpointsSelector.setToolTip(
            "Optional fiducial list with two points defining the desired axis endpoints and direction."
        )
        form.addRow("Axis endpoints (optional, 2 points):", self.endpointsSelector)

        segmentation_section = ctk.ctkCollapsibleButton()
        segmentation_section.text = "2. Automatic segmentation"
        segmentation_section.collapsed = True
        self.layout.addWidget(segmentation_section)
        form = qt.QFormLayout(segmentation_section)

        self.thresholdMethod = qt.QComboBox()
        self.thresholdMethod.addItem("Automatic Otsu", "otsu")
        self.thresholdMethod.addItem("Manual lower threshold", "manual")
        form.addRow("Threshold method:", self.thresholdMethod)

        self.manualThreshold = self._double_spin(-1_000_000, 1_000_000, 500, 25, 1)
        self.manualThreshold.setEnabled(False)
        form.addRow("Manual threshold:", self.manualThreshold)

        self.closingRadius = self._double_spin(0.0, 10.0, 0.35, 0.05, 2, " mm")
        self.closingRadius.setToolTip("Closes small threshold gaps; avoid values large enough to bridge adjacent anatomy.")
        form.addRow("Closing radius:", self.closingRadius)

        self.fillAxisHoles = qt.QCheckBox("Fill internal cavities only for axis estimation")
        self.fillAxisHoles.setChecked(True)
        self.fillAxisHoles.setToolTip(
            "The exported anatomical mask remains unfilled; a separate solid copy is used for centerline extraction."
        )
        form.addRow(self.fillAxisHoles)

        axis_section = ctk.ctkCollapsibleButton()
        axis_section.text = "3. Axis estimation"
        axis_section.collapsed = False
        self.layout.addWidget(axis_section)
        form = qt.QFormLayout(axis_section)

        self.coarseSpacing = self._double_spin(0.3, 5.0, 1.25, 0.1, 2, " mm")
        form.addRow("Centerline search spacing:", self.coarseSpacing)
        self.centerlineStrength = self._double_spin(0.0, 100.0, 16.0, 1.0, 1)
        self.centerlineStrength.setToolTip("Higher values keep the path farther from the tooth surface.")
        form.addRow("Center preference:", self.centerlineStrength)
        self.centerlineExponent = self._double_spin(0.25, 8.0, 2.0, 0.25, 2)
        form.addRow("Surface penalty exponent:", self.centerlineExponent)
        self.axisSmoothing = self._double_spin(0.0, 10.0, 1.0, 0.1, 2, " mm")
        form.addRow("Axis smoothing:", self.axisSmoothing)
        self.axisSampleSpacing = self._double_spin(0.1, 3.0, 0.5, 0.05, 2, " mm")
        form.addRow("Internal axis sampling:", self.axisSampleSpacing)
        self.controlPointSpacing = self._double_spin(0.5, 25.0, 3.0, 0.25, 2, " mm")
        form.addRow("Editable control-point spacing:", self.controlPointSpacing)
        self.refinementIterations = qt.QSpinBox()
        self.refinementIterations.setRange(0, 5)
        self.refinementIterations.setValue(1)
        self.refinementIterations.setToolTip("Recenters intermediate points using orthogonal mask cross-sections.")
        form.addRow("Cross-section refinement iterations:", self.refinementIterations)
        self.endpointCapFraction = self._double_spin(0.01, 0.20, 0.06, 0.01, 2)
        form.addRow("Automatic endpoint cap fraction:", self.endpointCapFraction)
        self.fovMargin = self._double_spin(0.0, 30.0, 3.0, 0.5, 1, " mm")
        form.addRow("Field-of-view margin:", self.fovMargin)

        self.curveSelector = self._node_selector(
            ["vtkMRMLMarkupsCurveNode"], none_enabled=True, add_enabled=True
        )
        self.curveSelector.setToolTip("Output/editable tooth centerline. Drag control points after estimation if needed.")
        form.addRow("Editable axis curve:", self.curveSelector)

        self.estimateButton = qt.QPushButton("Estimate segmentation and axis")
        self.estimateButton.setMinimumHeight(34)
        form.addRow(self.estimateButton)

        review = qt.QLabel(
            "After estimation, inspect the segmentation and curve in all views. Correct the segmentation in "
            "Segment Editor or drag curve control points before exporting. The two ends are geometric; root/crown "
            "identity is not inferred automatically."
        )
        review.setWordWrap(True)
        form.addRow(review)

        export_section = ctk.ctkCollapsibleButton()
        export_section.text = "4. Orthogonal slicing and export"
        export_section.collapsed = False
        self.layout.addWidget(export_section)
        form = qt.QFormLayout(export_section)

        self.orientationSelector = self._node_selector(
            ["vtkMRMLMarkupsFiducialNode", "vtkMRMLMarkupsLineNode"],
            none_enabled=True,
            add_enabled=True,
        )
        self.orientationSelector.setToolTip(
            "Optional anatomical in-plane orientation. A line uses point 1→2; one fiducial points from the first axis center."
        )
        form.addRow("In-plane orientation landmark (optional):", self.orientationSelector)

        self.sliceSpacing = self._double_spin(0.05, 20.0, 0.5, 0.05, 2, " mm")
        form.addRow("Distance along axis:", self.sliceSpacing)
        self.pixelSpacing = self._double_spin(0.02, 10.0, 0.25, 0.02, 3, " mm")
        self.pixelSpacing.setToolTip("In-plane output pixel size; initialized from the input volume.")
        form.addRow("Output pixel size:", self.pixelSpacing)

        self.autoFov = qt.QCheckBox("Compute field of view from current tooth mask")
        self.autoFov.setChecked(True)
        form.addRow(self.autoFov)
        self.fixedFov = self._double_spin(2.0, 500.0, 30.0, 1.0, 1, " mm")
        self.fixedFov.setEnabled(False)
        form.addRow("Fixed square field of view:", self.fixedFov)

        self.interpolation = qt.QComboBox()
        self.interpolation.addItem("Linear", 1)
        self.interpolation.addItem("Cubic", 3)
        form.addRow("CT interpolation:", self.interpolation)

        self.reverseAxis = qt.QCheckBox("Reverse slice order")
        self.reverseAxis.setToolTip("Use this to switch root-to-crown versus crown-to-root order.")
        form.addRow(self.reverseAxis)
        self.rotateInPlane180 = qt.QCheckBox("Rotate in-plane orientation by 180°")
        self.rotateInPlane180.setToolTip(
            "Reverses both transported in-plane axes while preserving a right-handed spatial frame."
        )
        form.addRow(self.rotateInPlane180)

        self.exportCtTiff = qt.QCheckBox("Quantitative CT TIFF slices (float32)")
        self.exportCtTiff.setChecked(True)
        form.addRow(self.exportCtTiff)
        self.exportMaskTiff = qt.QCheckBox("Binary mask TIFF slices")
        self.exportMaskTiff.setChecked(True)
        form.addRow(self.exportMaskTiff)
        self.exportPreview = qt.QCheckBox("Windowed PNG previews with mask outline")
        self.exportPreview.setChecked(True)
        form.addRow(self.exportPreview)
        self.saveStraightened = qt.QCheckBox("Straightened CT and mask NRRD volumes")
        self.saveStraightened.setChecked(True)
        form.addRow(self.saveStraightened)
        self.saveSegmentation = qt.QCheckBox("Source tooth segmentation (.seg.nrrd)")
        self.saveSegmentation.setChecked(True)
        form.addRow(self.saveSegmentation)
        self.saveCurve = qt.QCheckBox("Editable axis curve (.mrk.json)")
        self.saveCurve.setChecked(True)
        form.addRow(self.saveCurve)
        self.showPlanePreview = qt.QCheckBox("Show representative plane outlines in 3D")
        self.showPlanePreview.setChecked(True)
        form.addRow(self.showPlanePreview)

        self.outputDirectory = qt.QLineEdit()
        browse = qt.QToolButton()
        browse.text = "…"
        browse.setToolTip("Choose output parent folder")
        browse.connect("clicked()", self._browse_output)
        row = qt.QHBoxLayout()
        row.addWidget(self.outputDirectory)
        row.addWidget(browse)
        form.addRow("Output parent folder:", row)

        self.generateButton = qt.QPushButton("Generate and export orthogonal slices")
        self.generateButton.setMinimumHeight(38)
        form.addRow(self.generateButton)

        self.progress = qt.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        form.addRow(self.progress)

        self.statusLabel = qt.QLabel()
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._on_volume_changed)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._on_segmentation_changed)
        self.curveSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._update_enabled_state)
        self.thresholdMethod.connect("currentIndexChanged(int)", self._on_threshold_method_changed)
        self.autoFov.connect("toggled(bool)", lambda enabled: self.fixedFov.setEnabled(not enabled))
        self.outputDirectory.connect("textChanged(QString)", self._update_enabled_state)
        self.estimateButton.connect("clicked()", self.on_estimate)
        self.generateButton.connect("clicked()", self.on_generate)

    def _browse_output(self):
        path = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Select output parent folder")
        if path:
            self.outputDirectory.setText(path)

    def _on_threshold_method_changed(self, index):
        self.manualThreshold.setEnabled(int(index) == 1)

    def _on_volume_changed(self, node=None):
        volume = self.inputVolumeSelector.currentNode()
        if volume is not None:
            try:
                _, matrix = self.logic.volume_array_and_world_matrix(volume, copy_array=False)
                minimum_spacing = float(array_axis_spacings(matrix).min())
                self.pixelSpacing.setValue(max(0.02, minimum_spacing))
                self.axisSampleSpacing.setValue(max(0.1, min(1.0, minimum_spacing)))
            except Exception:
                logging.exception("Could not initialize spacing controls from volume")
        self._update_enabled_state()

    def _on_segmentation_changed(self, node=None):
        self._update_segment_list()
        self._update_enabled_state()

    def _update_segment_list(self):
        selected_id = self.current_segment_id()
        self.segmentCombo.blockSignals(True)
        self.segmentCombo.clear()
        segmentation_node = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if segmentation_node is not None:
            segmentation = segmentation_node.GetSegmentation()
            ids = vtk.vtkStringArray()
            segmentation.GetSegmentIDs(ids)
            for index in range(ids.GetNumberOfValues()):
                segment_id = ids.GetValue(index)
                segment = segmentation.GetSegment(segment_id)
                name = segment.GetName() if segment is not None else segment_id
                self.segmentCombo.addItem(name, segment_id)
                if segment_id == selected_id:
                    self.segmentCombo.setCurrentIndex(_count(self.segmentCombo) - 1)
        self.segmentCombo.blockSignals(False)

    def current_segment_id(self) -> str:
        if not hasattr(self, "segmentCombo") or _count(self.segmentCombo) == 0:
            return ""
        index = _current_index(self.segmentCombo)
        if index < 0:
            return ""
        value = self.segmentCombo.itemData(index)
        return str(value) if value is not None else ""

    def _select_segment_id(self, segment_id: str):
        for index in range(_count(self.segmentCombo)):
            if str(self.segmentCombo.itemData(index)) == str(segment_id):
                self.segmentCombo.setCurrentIndex(index)
                return

    def _update_enabled_state(self, *args):
        has_volume = self.inputVolumeSelector.currentNode() is not None
        segmentation = self.segmentationSelector.currentNode()
        segmentation_ok = segmentation is None or bool(self.current_segment_id())
        self.estimateButton.enabled = bool(has_volume and segmentation_ok)
        has_curve = self.curveSelector.currentNode() is not None
        output_parent = os.path.abspath(os.path.expanduser(_text(self.outputDirectory))) if _text(self.outputDirectory) else ""
        output_ok = bool(output_parent and os.path.isdir(output_parent))
        self.generateButton.enabled = bool(has_volume and segmentation is not None and segmentation_ok and has_curve and output_ok)
        if not has_volume:
            _set_status(self.statusLabel, "needs_input", "Select a scalar CT volume.")
        elif segmentation is not None and not segmentation_ok:
            _set_status(self.statusLabel, "needs_input", "Select a segment from the chosen segmentation.")
        elif not has_curve:
            _set_status(self.statusLabel, "ready", "Estimate an axis, or select an existing Markups curve.")
        elif not output_ok:
            _set_status(self.statusLabel, "needs_input", "Select an existing output parent folder.")
        else:
            _set_status(self.statusLabel, "ready", "Review the curve and segmentation, then generate slices.")

    def _threshold_method_value(self) -> str:
        value = self.thresholdMethod.itemData(_current_index(self.thresholdMethod))
        return str(value)

    def _interpolation_order(self) -> int:
        value = self.interpolation.itemData(_current_index(self.interpolation))
        return int(value)

    def _point_from_node(self, node, index=0) -> Optional[np.ndarray]:
        if node is None or node.GetNumberOfControlPoints() <= int(index):
            return None
        point = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(int(index), point)
        return np.asarray(point, dtype=float)

    def _endpoints(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        node = self.endpointsSelector.currentNode()
        if node is None:
            return None
        if node.GetNumberOfControlPoints() < 2:
            raise ValueError("The optional endpoint node must contain at least two control points.")
        return self._point_from_node(node, 0), self._point_from_node(node, 1)

    def _orientation_vector(self, first_center: np.ndarray) -> Optional[np.ndarray]:
        node = self.orientationSelector.currentNode()
        if node is None or node.GetNumberOfControlPoints() < 1:
            return None
        first = self._point_from_node(node, 0)
        if node.GetNumberOfControlPoints() >= 2:
            second = self._point_from_node(node, 1)
            return second - first
        return first - first_center

    def _roi_bounds(self) -> Optional[List[float]]:
        node = self.roiSelector.currentNode()
        if node is None:
            return None
        bounds = [0.0] * 6
        node.GetRASBounds(bounds)
        return [float(value) for value in bounds]

    def on_estimate(self):
        volume = self.inputVolumeSelector.currentNode()
        if volume is None:
            slicer.util.errorDisplay("Select an input CT volume.")
            return
        self.progress.setVisible(True)
        self.progress.setValue(5)
        self.estimateButton.enabled = False
        _set_status(self.statusLabel, "working", "Segmenting the tooth and estimating its geometric axis…")
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay("Tooth axis estimation failed", waitCursor=True):
                result = self.logic.estimate(
                    volume_node=volume,
                    segmentation_node=self.segmentationSelector.currentNode(),
                    segment_id=self.current_segment_id(),
                    curve_node=self.curveSelector.currentNode(),
                    threshold_method=self._threshold_method_value(),
                    manual_threshold=float(_value(self.manualThreshold)),
                    seed_world=self._point_from_node(self.seedSelector.currentNode(), 0),
                    roi_world_bounds=self._roi_bounds(),
                    endpoint_world=self._endpoints(),
                    closing_radius_mm=float(_value(self.closingRadius)),
                    fill_axis_holes=_checked(self.fillAxisHoles),
                    coarse_spacing_mm=float(_value(self.coarseSpacing)),
                    centerline_strength=float(_value(self.centerlineStrength)),
                    centerline_exponent=float(_value(self.centerlineExponent)),
                    smoothing_mm=float(_value(self.axisSmoothing)),
                    output_spacing_mm=float(_value(self.axisSampleSpacing)),
                    control_point_spacing_mm=float(_value(self.controlPointSpacing)),
                    endpoint_cap_fraction=float(_value(self.endpointCapFraction)),
                    refinement_iterations=int(_value(self.refinementIterations)),
                    fov_margin_mm=float(_value(self.fovMargin)),
                    progress_callback=self._progress_estimate,
                )
            self.segmentationSelector.setCurrentNode(result["segmentation_node"])
            self._update_segment_list()
            self._select_segment_id(result["segment_id"])
            self.curveSelector.setCurrentNode(result["curve_node"])
            self.fixedFov.setValue(float(result["recommended_fov_mm"]))
            self.progress.setValue(100)
            warnings = result.get("warnings", [])
            if warnings:
                _set_status(
                    self.statusLabel,
                    "warning",
                    f"Axis estimated with {len(warnings)} warning(s): " + " ".join(warnings),
                )
            else:
                _set_status(
                    self.statusLabel,
                    "complete",
                    f"Axis estimated ({result['curve_length_mm']:.1f} mm). Inspect and edit the curve before export.",
                )
        except Exception:
            logging.exception("Axis estimation failed")
            _set_status(self.statusLabel, "error", "Axis estimation failed; see the Python console for details.")
        finally:
            self.progress.setVisible(False)
            self.estimateButton.enabled = True
            self._update_enabled_state()

    def _progress_estimate(self, fraction: float, message: str):
        self.progress.setValue(int(np.clip(fraction, 0.0, 1.0) * 100.0))
        self.progress.setFormat(str(message))
        slicer.app.processEvents()

    def on_generate(self):
        volume = self.inputVolumeSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        curve = self.curveSelector.currentNode()
        segment_id = self.current_segment_id()
        parent = os.path.abspath(os.path.expanduser(_text(self.outputDirectory)))
        if volume is None or segmentation is None or curve is None or not segment_id:
            slicer.util.errorDisplay("Select a volume, tooth segment, and axis curve.")
            return
        if not os.path.isdir(parent):
            slicer.util.errorDisplay("The output parent folder does not exist.")
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.generateButton.enabled = False
        _set_status(self.statusLabel, "working", "Resampling CT and mask planes along the edited axis…")
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay("Orthogonal slice generation failed", waitCursor=True):
                reverse_axis = _checked(self.reverseAxis)
                curve_points = self.logic.curve_points_world(curve)
                first_center = curve_points[-1] if reverse_axis else curve_points[0]
                result = self.logic.generate_and_export(
                    volume_node=volume,
                    segmentation_node=segmentation,
                    segment_id=segment_id,
                    curve_node=curve,
                    orientation_vector=self._orientation_vector(first_center),
                    reverse_axis=reverse_axis,
                    rotate_in_plane_180=_checked(self.rotateInPlane180),
                    slice_spacing_mm=float(_value(self.sliceSpacing)),
                    pixel_spacing_mm=float(_value(self.pixelSpacing)),
                    auto_fov=_checked(self.autoFov),
                    fixed_fov_mm=float(_value(self.fixedFov)),
                    fov_margin_mm=float(_value(self.fovMargin)),
                    interpolation_order=self._interpolation_order(),
                    output_parent=parent,
                    export_ct_tiff=_checked(self.exportCtTiff),
                    export_mask_tiff=_checked(self.exportMaskTiff),
                    export_preview_png=_checked(self.exportPreview),
                    save_straightened=_checked(self.saveStraightened),
                    save_segmentation=_checked(self.saveSegmentation),
                    save_curve=_checked(self.saveCurve),
                    show_plane_preview=_checked(self.showPlanePreview),
                    progress_callback=self._progress_generate,
                )
            self.fixedFov.setValue(float(result["field_of_view_mm"]))
            self.progress.setValue(100)
            qc_warnings = result.get("warnings", [])
            state = "warning" if qc_warnings else "complete"
            message = (
                f"Generated {result['number_of_slices']} slices in {result['output_directory']}."
            )
            if qc_warnings:
                message += " Review: " + " ".join(qc_warnings)
            _set_status(self.statusLabel, state, message)
            slicer.util.showStatusMessage(
                f"Tooth Orthogonal Slices: wrote {result['number_of_slices']} slices",
                8000,
            )
        except Exception:
            logging.exception("Slice generation failed")
            _set_status(self.statusLabel, "error", "Slice generation failed; see the Python console for details.")
        finally:
            self.progress.setVisible(False)
            self.generateButton.enabled = True
            self._update_enabled_state()

    def _progress_generate(self, fraction: float, message: str):
        self.progress.setValue(int(np.clip(fraction, 0.0, 1.0) * 100.0))
        self.progress.setFormat(str(message))
        slicer.app.processEvents()


class ToothOrthogonalSlicesLogic(ScriptedLoadableModuleLogic):
    @staticmethod
    def _vtk_matrix_to_array(matrix: vtk.vtkMatrix4x4) -> np.ndarray:
        return np.asarray(
            [[matrix.GetElement(row, column) for column in range(4)] for row in range(4)],
            dtype=float,
        )

    def volume_array_and_world_matrix(self, volume_node, copy_array=True) -> Tuple[np.ndarray, np.ndarray]:
        if volume_node is None or volume_node.GetImageData() is None:
            raise ValueError("Input volume has no image data")
        values = slicer.util.arrayFromVolume(volume_node)
        values = np.array(values, copy=True) if copy_array else values

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = self._vtk_matrix_to_array(ijk_to_ras_vtk)

        # NumPy arrays are KJI (z,y,x), whereas Slicer volume matrices map IJK.
        array_zyx_to_ijk = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        parent_to_world = np.eye(4, dtype=float)
        parent = volume_node.GetParentTransformNode()
        if parent is not None:
            matrix = vtk.vtkMatrix4x4()
            if not parent.GetMatrixTransformToWorld(matrix):
                raise ValueError(
                    "The input volume is under a non-linear transform. Harden the transform before running this module."
                )
            parent_to_world = self._vtk_matrix_to_array(matrix)
        array_to_world = parent_to_world @ ijk_to_ras @ array_zyx_to_ijk
        return values, array_to_world

    def segment_mask(self, segmentation_node, segment_id: str, reference_volume_node) -> np.ndarray:
        if segmentation_node is None or not segment_id:
            raise ValueError("A segmentation and segment must be selected")
        segment = segmentation_node.GetSegmentation().GetSegment(str(segment_id))
        if segment is None:
            raise ValueError(f"Segment ID does not exist: {segment_id}")
        array = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, str(segment_id), reference_volume_node
        )
        mask = np.asarray(array, dtype=bool)
        reference_shape = slicer.util.arrayFromVolume(reference_volume_node).shape
        if mask.shape != reference_shape:
            raise RuntimeError(
                f"Segment geometry {mask.shape} does not match reference volume geometry {reference_shape}"
            )
        if not np.any(mask):
            raise ValueError("The selected segment is empty in the input volume geometry")
        return mask

    def _create_or_update_segmentation(self, volume_node, mask: np.ndarray, existing_node=None):
        node = existing_node
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", f"{volume_node.GetName()} Tooth Segmentation"
            )
            node.CreateDefaultDisplayNodes()
            node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
            if volume_node.GetTransformNodeID():
                node.SetAndObserveTransformNodeID(volume_node.GetTransformNodeID())
        segmentation = node.GetSegmentation()
        segment_id = segmentation.GetSegmentIdBySegmentName("Tooth")
        if not segment_id:
            segment_id = segmentation.AddEmptySegment("Tooth")
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            np.asarray(mask, dtype=np.uint8), node, segment_id, volume_node
        )
        node.CreateClosedSurfaceRepresentation()
        display = node.GetDisplayNode()
        if display is not None:
            display.SetSegmentVisibility3D(segment_id, True)
            display.SetSegmentVisibility2DFill(segment_id, False)
            display.SetSegmentVisibility2DOutline(segment_id, True)
        return node, str(segment_id)

    def _create_or_update_curve(self, volume_node, curve_node, control_points_world: np.ndarray):
        node = curve_node
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", f"{volume_node.GetName()} Tooth Axis"
            )
        node.CreateDefaultDisplayNodes()
        slicer.util.updateMarkupsControlPointsFromArray(node, control_points_world, world=True)
        if hasattr(node, "SetCurveTypeToCardinalSpline"):
            node.SetCurveTypeToCardinalSpline()
        if hasattr(node, "SetNumberOfPointsPerInterpolatingSegment"):
            node.SetNumberOfPointsPerInterpolatingSegment(20)
        display = node.GetDisplayNode()
        if display is not None:
            display.SetLineThickness(0.4)
            display.SetGlyphScale(1.5)
        return node

    def estimate(
        self,
        volume_node,
        segmentation_node=None,
        segment_id="",
        curve_node=None,
        threshold_method="otsu",
        manual_threshold=None,
        seed_world=None,
        roi_world_bounds=None,
        endpoint_world=None,
        closing_radius_mm=0.35,
        fill_axis_holes=True,
        coarse_spacing_mm=1.25,
        centerline_strength=16.0,
        centerline_exponent=2.0,
        smoothing_mm=1.0,
        output_spacing_mm=0.5,
        control_point_spacing_mm=3.0,
        endpoint_cap_fraction=0.06,
        refinement_iterations=1,
        fov_margin_mm=3.0,
        progress_callback=None,
    ) -> Dict[str, object]:
        def progress(fraction, message):
            if progress_callback:
                progress_callback(float(fraction), str(message))

        progress(0.05, "Reading volume")
        # arrayFromVolume returns a view into the MRML volume.  The numerical
        # routines never mutate it, so avoiding a full CT copy substantially
        # reduces peak memory for high-resolution scans.
        volume, array_to_world = self.volume_array_and_world_matrix(volume_node, copy_array=False)
        warnings: List[str] = []
        threshold_value = None
        threshold_used = "existing segmentation"

        if segmentation_node is not None and segment_id:
            progress(0.15, "Reading selected tooth segment")
            anatomical_mask = self.segment_mask(segmentation_node, segment_id, volume_node)
            axis_mask = prepare_axis_mask(
                anatomical_mask,
                array_to_world,
                closing_radius_mm=float(closing_radius_mm),
                fill_holes=bool(fill_axis_holes),
            )
        else:
            progress(0.15, "Thresholding and selecting tooth component")
            segmentation = automatic_tooth_segmentation(
                volume,
                array_to_world,
                threshold_method=threshold_method,
                manual_threshold=manual_threshold,
                seed_world=seed_world,
                roi_world_bounds=roi_world_bounds,
                closing_radius_mm=float(closing_radius_mm),
                fill_axis_holes=bool(fill_axis_holes),
            )
            anatomical_mask = segmentation.mask
            axis_mask = segmentation.axis_mask
            warnings.extend(segmentation.warnings)
            threshold_value = float(segmentation.threshold)
            threshold_used = segmentation.threshold_method
            segmentation_node, segment_id = self._create_or_update_segmentation(
                volume_node, anatomical_mask, existing_node=None
            )

        progress(0.35, "Computing center-biased path")
        axis_result = estimate_tooth_axis(
            axis_mask,
            array_to_world,
            coarse_spacing_mm=float(coarse_spacing_mm),
            centerline_strength=float(centerline_strength),
            centerline_exponent=float(centerline_exponent),
            smoothing_mm=float(smoothing_mm),
            output_spacing_mm=float(output_spacing_mm),
            control_point_spacing_mm=float(control_point_spacing_mm),
            endpoint_cap_fraction=float(endpoint_cap_fraction),
            cross_section_refinement_iterations=int(refinement_iterations),
            fov_margin_mm=float(fov_margin_mm),
            endpoint_world=endpoint_world,
        )
        warnings.extend(axis_result.warnings)
        progress(0.85, "Creating editable Markups curve")
        curve_node = self._create_or_update_curve(
            volume_node, curve_node, axis_result.control_points_world
        )
        curve_node.SetAttribute(
            "ToothOrthogonalSlices.InitialNormalRAS",
            json.dumps([float(value) for value in axis_result.initial_normal_world]),
        )
        curve_node.SetAttribute(
            "ToothOrthogonalSlices.RecommendedFOVmm", str(float(axis_result.recommended_fov_mm))
        )
        curve_node.SetAttribute("ToothOrthogonalSlices.SourceVolumeNodeID", volume_node.GetID())
        curve_node.SetAttribute("ToothOrthogonalSlices.SegmentationNodeID", segmentation_node.GetID())
        curve_node.SetAttribute("ToothOrthogonalSlices.SegmentID", str(segment_id))
        curve_node.SetAttribute("ToothOrthogonalSlices.AxisQC", json.dumps(axis_result.qc, sort_keys=True))
        axis_parameters = {
            "threshold_method": threshold_used,
            "threshold_value": threshold_value,
            "closing_radius_mm": float(closing_radius_mm),
            "fill_axis_holes": bool(fill_axis_holes),
            "coarse_spacing_mm": float(coarse_spacing_mm),
            "centerline_strength": float(centerline_strength),
            "centerline_exponent": float(centerline_exponent),
            "smoothing_mm": float(smoothing_mm),
            "output_spacing_mm": float(output_spacing_mm),
            "control_point_spacing_mm": float(control_point_spacing_mm),
            "endpoint_cap_fraction": float(endpoint_cap_fraction),
            "refinement_iterations": int(refinement_iterations),
            "fov_margin_mm": float(fov_margin_mm),
            "seed_ras": None if seed_world is None else np.asarray(seed_world, dtype=float).tolist(),
            "roi_bounds_ras": (
                None if roi_world_bounds is None else np.asarray(roi_world_bounds, dtype=float).tolist()
            ),
            "endpoint_ras": (
                None
                if endpoint_world is None
                else [
                    np.asarray(endpoint_world[0], dtype=float).tolist(),
                    np.asarray(endpoint_world[1], dtype=float).tolist(),
                ]
            ),
        }
        curve_node.SetAttribute(
            "ToothOrthogonalSlices.AxisParameters", json.dumps(axis_parameters, sort_keys=True)
        )
        curve_node.SetAttribute(
            "ToothOrthogonalSlices.DirectionMeaning",
            "Control-point order only; biological root/crown identity was not inferred.",
        )
        if threshold_value is not None:
            curve_node.SetAttribute("ToothOrthogonalSlices.Threshold", str(threshold_value))
            curve_node.SetAttribute("ToothOrthogonalSlices.ThresholdMethod", threshold_used)

        progress(1.0, "Axis ready for review")
        return {
            "segmentation_node": segmentation_node,
            "segment_id": str(segment_id),
            "curve_node": curve_node,
            "recommended_fov_mm": float(axis_result.recommended_fov_mm),
            "curve_length_mm": float(axis_result.qc["curve_length_mm"]),
            "threshold": threshold_value,
            "threshold_method": threshold_used,
            "axis_qc": axis_result.qc,
            "warnings": warnings,
        }

    def curve_points_world(self, curve_node) -> np.ndarray:
        if curve_node is None:
            raise ValueError("No axis curve is selected")
        curve_points = curve_node.GetCurvePointsWorld()
        if curve_points is not None and curve_points.GetNumberOfPoints() >= 2:
            points = vtk_to_numpy(curve_points.GetData()).astype(float, copy=True)
        else:
            count = curve_node.GetNumberOfControlPoints()
            if count < 2:
                raise ValueError("The axis curve must contain at least two control points")
            points = np.empty((count, 3), dtype=float)
            for index in range(count):
                point = [0.0, 0.0, 0.0]
                curve_node.GetNthControlPointPositionWorld(index, point)
                points[index] = point
        points = remove_duplicate_points(points)
        if points.shape[0] < 2:
            raise ValueError("The axis curve contains fewer than two distinct points")
        return points

    def _initial_normal(self, curve_node, first_tangent: np.ndarray) -> np.ndarray:
        value = curve_node.GetAttribute("ToothOrthogonalSlices.InitialNormalRAS")
        if value:
            try:
                normal = np.asarray(json.loads(value), dtype=float)
                if normal.shape == (3,) and np.all(np.isfinite(normal)):
                    return normal
            except Exception:
                logging.warning("Could not parse stored initial curve normal")
        basis = np.eye(3)
        return basis[int(np.argmin(np.abs(basis @ first_tangent)))]

    def _estimate_memory_bytes(self, number_of_slices, field_of_view_mm, pixel_spacing_mm):
        side = max(3, int(math.ceil(field_of_view_mm / pixel_spacing_mm)) + 1)
        if side % 2 == 0:
            side += 1
        # Conservative peak estimate: NumPy CT/mask stacks, VTK volume copies,
        # and short-lived conversion buffers coexist during scene/output creation.
        return int(number_of_slices) * side * side * 12, side

    def generate_and_export(
        self,
        volume_node,
        segmentation_node,
        segment_id,
        curve_node,
        orientation_vector=None,
        reverse_axis=False,
        rotate_in_plane_180=False,
        slice_spacing_mm=0.5,
        pixel_spacing_mm=0.25,
        auto_fov=True,
        fixed_fov_mm=30.0,
        fov_margin_mm=3.0,
        interpolation_order=1,
        output_parent="",
        export_ct_tiff=True,
        export_mask_tiff=True,
        export_preview_png=True,
        save_straightened=True,
        save_segmentation=True,
        save_curve=True,
        show_plane_preview=True,
        progress_callback=None,
    ) -> Dict[str, object]:
        def progress(fraction, message):
            if progress_callback:
                progress_callback(float(fraction), str(message))

        output_parent_path = Path(output_parent).expanduser().resolve()
        if not output_parent or not output_parent_path.is_dir():
            raise ValueError("Output parent must be an existing directory")
        if not os.access(str(output_parent_path), os.W_OK):
            raise PermissionError(f"Output parent is not writable: {output_parent_path}")
        if fov_margin_mm < 0.0:
            raise ValueError("Field-of-view margin cannot be negative")

        progress(0.02, "Reading input data")
        volume, array_to_world = self.volume_array_and_world_matrix(volume_node, copy_array=False)
        mask = self.segment_mask(segmentation_node, segment_id, volume_node)
        dense_curve = self.curve_points_world(curve_node)
        points, _ = resample_polyline(dense_curve, float(slice_spacing_mm))
        if reverse_axis:
            points = points[::-1].copy()

        tangent = points[1] - points[0]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        initial_normal = (
            np.asarray(orientation_vector, dtype=float)
            if orientation_vector is not None and np.linalg.norm(orientation_vector) > 1e-10
            else self._initial_normal(curve_node, tangent)
        )
        if rotate_in_plane_180:
            initial_normal = -initial_normal
        frames = parallel_transport_frames(points, initial_normal)
        frame_validation = validate_frames(frames, tolerance=5e-6)

        progress(0.08, "Determining field of view")
        if auto_fov:
            field_of_view_mm = recommend_field_of_view(
                mask,
                array_to_world,
                points,
                margin_mm=float(fov_margin_mm),
            )
        else:
            field_of_view_mm = float(fixed_fov_mm)
        if field_of_view_mm <= 0.0 or pixel_spacing_mm <= 0.0:
            raise ValueError("Field of view and pixel spacing must be positive")

        estimated_bytes, side = self._estimate_memory_bytes(
            points.shape[0], field_of_view_mm, float(pixel_spacing_mm)
        )
        if estimated_bytes > 3 * 1024**3:
            raise MemoryError(
                f"Requested stack is approximately {estimated_bytes / 1024**3:.1f} GiB "
                f"({points.shape[0]} slices × {side} × {side}). Increase slice/pixel spacing or reduce field of view."
            )

        progress(0.12, "Resampling CT planes")
        ct_stack, u_coordinates, v_coordinates = sample_volume_along_frames(
            volume,
            array_to_world,
            frames,
            pixel_spacing_mm=float(pixel_spacing_mm),
            field_of_view_mm=float(field_of_view_mm),
            interpolation_order=int(interpolation_order),
            output_dtype=np.float32,
            progress_callback=(
                lambda done, total: progress(0.12 + 0.30 * done / max(total, 1), "Resampling CT planes")
            ),
        )
        actual_field_of_view_mm = float(u_coordinates[-1] - u_coordinates[0])
        progress(0.43, "Resampling segmentation planes")
        mask_stack, _, _ = sample_volume_along_frames(
            mask.astype(np.uint8),
            array_to_world,
            frames,
            pixel_spacing_mm=float(pixel_spacing_mm),
            field_of_view_mm=float(field_of_view_mm),
            interpolation_order=0,
            outside_value=0.0,
            output_dtype=np.uint8,
            progress_callback=(
                lambda done, total: progress(0.43 + 0.18 * done / max(total, 1), "Resampling mask planes")
            ),
        )
        mask_stack = mask_stack > 0

        progress(0.62, "Computing quality-control measurements")
        slice_records, qc, qc_warnings = self._slice_quality_control(
            frames, mask_stack, float(pixel_spacing_mm), actual_field_of_view_mm
        )
        qc["frame_validation"] = frame_validation

        source_role_suffix = str(volume_node.GetID())
        ct_node = self._update_rectified_volume(
            role=f"StraightenedCT:{source_role_suffix}",
            class_name="vtkMRMLScalarVolumeNode",
            name=f"{volume_node.GetName()} Straightened CT",
            array=ct_stack,
            pixel_spacing_mm=float(pixel_spacing_mm),
            slice_spacing_mm=float(slice_spacing_mm),
            u0=float(u_coordinates[0]),
            v0=float(v_coordinates[0]),
        )
        mask_node = self._update_rectified_volume(
            role=f"StraightenedMask:{source_role_suffix}",
            class_name="vtkMRMLLabelMapVolumeNode",
            name=f"{volume_node.GetName()} Straightened Tooth Mask",
            array=mask_stack.astype(np.uint8),
            pixel_spacing_mm=float(pixel_spacing_mm),
            slice_spacing_mm=float(slice_spacing_mm),
            u0=float(u_coordinates[0]),
            v0=float(v_coordinates[0]),
        )
        frame_table_node = self._update_frame_table(
            frames, slice_records, volume_node.GetName(), source_role_suffix
        )
        plane_model_node = self._update_plane_preview(
            frames,
            field_of_view_mm=actual_field_of_view_mm,
            name=f"{volume_node.GetName()} Orthogonal Plane Preview",
            visible=bool(show_plane_preview),
            source_role_suffix=source_role_suffix,
        )

        progress(0.67, "Writing output files")
        output_directory = self._export_results(
            volume_node=volume_node,
            segmentation_node=segmentation_node,
            segment_id=str(segment_id),
            curve_node=curve_node,
            ct_node=ct_node,
            mask_node=mask_node,
            ct_stack=ct_stack,
            mask_stack=mask_stack,
            frames=frames,
            slice_records=slice_records,
            qc=qc,
            qc_warnings=qc_warnings,
            array_to_world=array_to_world,
            u_coordinates=u_coordinates,
            v_coordinates=v_coordinates,
            output_parent=output_parent,
            pixel_spacing_mm=float(pixel_spacing_mm),
            slice_spacing_mm=float(slice_spacing_mm),
            field_of_view_mm=actual_field_of_view_mm,
            interpolation_order=int(interpolation_order),
            reverse_axis=bool(reverse_axis),
            rotate_in_plane_180=bool(rotate_in_plane_180),
            export_ct_tiff=bool(export_ct_tiff),
            export_mask_tiff=bool(export_mask_tiff),
            export_preview_png=bool(export_preview_png),
            save_straightened=bool(save_straightened),
            save_segmentation=bool(save_segmentation),
            save_curve=bool(save_curve),
            progress_callback=lambda fraction, message: progress(0.67 + 0.32 * fraction, message),
        )
        progress(1.0, "Export complete")
        return {
            "output_directory": str(output_directory),
            "number_of_slices": int(frames.centers_world.shape[0]),
            "field_of_view_mm": actual_field_of_view_mm,
            "ct_node": ct_node,
            "mask_node": mask_node,
            "frame_table_node": frame_table_node,
            "plane_model_node": plane_model_node,
            "qc": qc,
            "warnings": qc_warnings,
        }

    def _get_or_create_output_node(self, class_name: str, role: str, name: str):
        node = None
        for candidate in slicer.util.getNodesByClass(class_name):
            if candidate.GetAttribute(OUTPUT_ATTRIBUTE) == role:
                node = candidate
                break
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass(class_name, name)
            node.SetAttribute(OUTPUT_ATTRIBUTE, role)
        else:
            node.SetName(name)
        if hasattr(node, "CreateDefaultDisplayNodes"):
            node.CreateDefaultDisplayNodes()
        return node

    def _update_rectified_volume(
        self,
        role,
        class_name,
        name,
        array,
        pixel_spacing_mm,
        slice_spacing_mm,
        u0,
        v0,
    ):
        node = self._get_or_create_output_node(class_name, role, name)
        node.SetAndObserveTransformNodeID(None)
        slicer.util.updateVolumeFromArray(node, np.asarray(array))
        ijk_to_ras = np.eye(4, dtype=float)
        ijk_to_ras[0, 0] = float(pixel_spacing_mm)
        ijk_to_ras[1, 1] = float(pixel_spacing_mm)
        ijk_to_ras[2, 2] = float(slice_spacing_mm)
        ijk_to_ras[:3, 3] = [float(u0), float(v0), 0.0]
        node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(ijk_to_ras))
        node.SetAttribute(
            "ToothOrthogonalSlices.GeometryNote",
            "Rectified coordinates: I/J are transported in-plane axes; K is arc-length order. "
            "Use slice_frames.csv for mapping to source RAS.",
        )
        display = node.GetDisplayNode()
        if display is not None and class_name == "vtkMRMLScalarVolumeNode":
            display.AutoWindowLevelOn()
        return node

    def _slice_quality_control(
        self,
        frames: FrameResult,
        mask_stack: np.ndarray,
        pixel_spacing_mm: float,
        field_of_view_mm: float,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object], List[str]]:
        records: List[Dict[str, object]] = []
        areas = np.zeros(mask_stack.shape[0], dtype=float)
        component_counts = np.zeros(mask_stack.shape[0], dtype=int)
        border_touching = np.zeros(mask_stack.shape[0], dtype=bool)
        centroid_offsets = np.full(mask_stack.shape[0], np.nan, dtype=float)
        center = 0.5 * (np.asarray(mask_stack.shape[1:]) - 1)

        tangents = frames.tangents_world
        curvature = np.zeros(mask_stack.shape[0], dtype=float)
        if mask_stack.shape[0] > 1:
            angles = np.arccos(
                np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
            )
            ds = np.maximum(np.diff(frames.arc_lengths_mm), 1e-9)
            segment_curvature = angles / ds
            curvature[:-1] = segment_curvature
            curvature[-1] = segment_curvature[-1]

        previous_x = frames.x_axes_world[0]
        for index, section in enumerate(mask_stack):
            _, count = ndimage.label(
                section, structure=ndimage.generate_binary_structure(2, 2)
            )
            component_counts[index] = int(count)
            areas[index] = float(np.count_nonzero(section)) * pixel_spacing_mm**2
            border_touching[index] = bool(
                np.any(section[0, :])
                or np.any(section[-1, :])
                or np.any(section[:, 0])
                or np.any(section[:, -1])
            )
            foreground = np.argwhere(section)
            if foreground.size:
                centroid_offsets[index] = float(
                    np.linalg.norm((foreground.mean(axis=0) - center) * pixel_spacing_mm)
                )
            rotation_deg = 0.0
            if index > 0:
                rotation_deg = math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(previous_x, frames.x_axes_world[index]), -1.0, 1.0
                            )
                        )
                    )
                )
                previous_x = frames.x_axes_world[index]
            flags: List[str] = []
            if count == 0:
                flags.append("empty_mask")
            if count > 1:
                flags.append("multiple_components")
            if border_touching[index]:
                flags.append("touches_field_boundary")
            if np.isfinite(centroid_offsets[index]) and centroid_offsets[index] > 2.0 * pixel_spacing_mm:
                flags.append("mask_centroid_offset")
            if curvature[index] * 0.5 * field_of_view_mm > 0.8:
                flags.append("high_curvature_overlap_risk")
            records.append(
                {
                    "index": int(index),
                    "arc_length_mm": float(frames.arc_lengths_mm[index]),
                    "area_mm2": float(areas[index]),
                    "component_count": int(count),
                    "touches_field_boundary": bool(border_touching[index]),
                    "centroid_offset_mm": (
                        float(centroid_offsets[index]) if np.isfinite(centroid_offsets[index]) else None
                    ),
                    "curvature_per_mm": float(curvature[index]),
                    "frame_rotation_deg": float(rotation_deg),
                    "flags": ";".join(flags),
                }
            )

        positive = areas[areas > 0]
        abrupt = []
        if areas.size > 1:
            denominator = np.maximum(np.minimum(areas[:-1], areas[1:]), pixel_spacing_mm**2)
            relative = np.abs(np.diff(areas)) / denominator
            abrupt = (np.where(relative > 0.75)[0] + 1).astype(int).tolist()
            for index in abrupt:
                existing = str(records[index]["flags"])
                records[index]["flags"] = (
                    f"{existing};abrupt_area_change" if existing else "abrupt_area_change"
                )
        centroid_offset_indices = np.where(
            np.isfinite(centroid_offsets) & (centroid_offsets > 2.0 * pixel_spacing_mm)
        )[0].astype(int).tolist()
        qc: Dict[str, object] = {
            "number_of_slices": int(mask_stack.shape[0]),
            "curve_length_mm": float(frames.arc_lengths_mm[-1]),
            "empty_slice_indices": np.where(areas == 0)[0].astype(int).tolist(),
            "multiple_component_slice_indices": np.where(component_counts > 1)[0].astype(int).tolist(),
            "field_boundary_slice_indices": np.where(border_touching)[0].astype(int).tolist(),
            "abrupt_area_change_slice_indices": abrupt,
            "mask_centroid_offset_slice_indices": centroid_offset_indices,
            "high_curvature_overlap_risk_indices": np.where(
                curvature * 0.5 * field_of_view_mm > 0.8
            )[0].astype(int).tolist(),
            "area_mm2": {
                "minimum_nonzero": float(positive.min()) if positive.size else 0.0,
                "median_nonzero": float(np.median(positive)) if positive.size else 0.0,
                "maximum": float(areas.max()) if areas.size else 0.0,
            },
            "maximum_centroid_offset_mm": (
                float(np.nanmax(centroid_offsets)) if np.any(np.isfinite(centroid_offsets)) else None
            ),
            "maximum_curvature_per_mm": float(curvature.max()) if curvature.size else 0.0,
        }
        warnings: List[str] = []
        if qc["empty_slice_indices"]:
            warnings.append(f"{len(qc['empty_slice_indices'])} slices have an empty tooth mask.")
        if qc["field_boundary_slice_indices"]:
            warnings.append(
                f"The tooth touches the output boundary in {len(qc['field_boundary_slice_indices'])} slices; increase field of view."
            )
        if qc["multiple_component_slice_indices"]:
            warnings.append(
                f"{len(qc['multiple_component_slice_indices'])} slices contain multiple mask components."
            )
        if qc["high_curvature_overlap_risk_indices"]:
            warnings.append(
                "Strong curvature may cause neighboring planes to overlap in the source volume."
            )
        if abrupt:
            warnings.append(f"Abrupt area changes were flagged in {len(abrupt)} slices.")
        return records, qc, warnings

    def _update_frame_table(self, frames, records, source_name, source_role_suffix):
        node = self._get_or_create_output_node(
            "vtkMRMLTableNode", f"FrameTable:{source_role_suffix}", f"{source_name} Slice Frames"
        )
        table = vtk.vtkTable()

        def add_numeric(name, values, integer=False):
            array = vtk.vtkIntArray() if integer else vtk.vtkDoubleArray()
            array.SetName(name)
            array.SetNumberOfValues(len(values))
            for index, value in enumerate(values):
                array.SetValue(index, int(value) if integer else float(value))
            table.AddColumn(array)

        add_numeric("slice_index", range(len(records)), integer=True)
        add_numeric("arc_length_mm", frames.arc_lengths_mm)
        for axis, data in (
            ("center_ras", frames.centers_world),
            ("x_axis_ras", frames.x_axes_world),
            ("y_axis_ras", frames.y_axes_world),
            ("tangent_ras", frames.tangents_world),
        ):
            add_numeric(f"{axis}_x", data[:, 0])
            add_numeric(f"{axis}_y", data[:, 1])
            add_numeric(f"{axis}_z", data[:, 2])
        add_numeric("area_mm2", [record["area_mm2"] for record in records])
        add_numeric(
            "component_count", [record["component_count"] for record in records], integer=True
        )
        flags = vtk.vtkStringArray()
        flags.SetName("qc_flags")
        for record in records:
            flags.InsertNextValue(str(record["flags"]))
        table.AddColumn(flags)
        node.SetAndObserveTable(table)
        return node

    def _update_plane_preview(
        self, frames, field_of_view_mm, name, visible=True, source_role_suffix=""
    ):
        node = self._get_or_create_output_node(
            "vtkMRMLModelNode", f"PlanePreview:{source_role_suffix}", name
        )
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        half = 0.5 * float(field_of_view_mm)
        count = frames.centers_world.shape[0]
        step = max(1, int(math.ceil(count / 20.0)))
        indices = list(range(0, count, step))
        if indices[-1] != count - 1:
            indices.append(count - 1)
        for index in indices:
            center = frames.centers_world[index]
            x = frames.x_axes_world[index]
            y = frames.y_axes_world[index]
            corners = [
                center - half * x - half * y,
                center + half * x - half * y,
                center + half * x + half * y,
                center - half * x + half * y,
            ]
            ids = [points.InsertNextPoint(*corner) for corner in corners]
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(5)
            for position, point_id in enumerate(ids + [ids[0]]):
                polyline.GetPointIds().SetId(position, point_id)
            lines.InsertNextCell(polyline)
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        node.SetAndObservePolyData(polydata)
        display = node.GetDisplayNode()
        if display is not None:
            display.SetVisibility(bool(visible))
            display.SetLineWidth(2.0)
            display.SetOpacity(0.65)
        return node

    @staticmethod
    def _write_json_atomic(path: Path, value: object):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(str(temporary), str(path))

    @staticmethod
    def _save_node_atomic(node, path: Path):
        """Save a Slicer node without exposing a partially written final file."""
        temporary_directory = path.parent / ".writing"
        temporary_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = temporary_directory / path.name
        temporary_path.unlink(missing_ok=True)
        if not slicer.util.saveNode(node, str(temporary_path)):
            temporary_path.unlink(missing_ok=True)
            raise IOError(f"Could not save {path}")
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            temporary_path.unlink(missing_ok=True)
            raise IOError(f"Slicer reported success but produced no data for {path}")
        os.replace(str(temporary_path), str(path))
        try:
            temporary_directory.rmdir()
        except OSError:
            pass

    @staticmethod
    def _json_node_attribute(node, name: str):
        value = node.GetAttribute(name) if node is not None else None
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            logging.warning("Could not parse JSON node attribute %s", name)
            return value

    def _export_results(
        self,
        volume_node,
        segmentation_node,
        segment_id,
        curve_node,
        ct_node,
        mask_node,
        ct_stack,
        mask_stack,
        frames,
        slice_records,
        qc,
        qc_warnings,
        array_to_world,
        u_coordinates,
        v_coordinates,
        output_parent,
        pixel_spacing_mm,
        slice_spacing_mm,
        field_of_view_mm,
        interpolation_order,
        reverse_axis,
        rotate_in_plane_180,
        export_ct_tiff,
        export_mask_tiff,
        export_preview_png,
        save_straightened,
        save_segmentation,
        save_curve,
        progress_callback=None,
    ) -> Path:
        def progress(fraction, message):
            if progress_callback:
                progress_callback(float(fraction), str(message))

        parent = Path(output_parent).expanduser().resolve()
        if not output_parent or not parent.is_dir():
            raise ValueError("Output parent must be an existing directory")
        if not os.access(str(parent), os.W_OK):
            raise PermissionError(f"Output parent is not writable: {parent}")

        base_name = safe_filename(volume_node.GetName())
        root = parent / f"{base_name}_orthogonal_slices"
        root.mkdir(parents=True, exist_ok=True)
        incomplete_path = root / ".export_incomplete.json"
        self._write_json_atomic(
            incomplete_path,
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "module_version": MODULE_VERSION,
                "message": "This marker is removed only after a complete export.",
            },
        )
        # A previous manifest must never make a partially overwritten run look
        # complete.  Other files are replaced atomically below.
        (root / "manifest.json").unlink(missing_ok=True)

        optional_outputs = {
            "straightened_ct.nrrd": bool(save_straightened),
            "straightened_mask.nrrd": bool(save_straightened),
            "tooth_segmentation.seg.nrrd": bool(save_segmentation),
            "tooth_axis.mrk.json": bool(save_curve),
        }
        for filename, enabled in optional_outputs.items():
            if not enabled:
                (root / filename).unlink(missing_ok=True)
        ct_directory = root / "ct_slices"
        mask_directory = root / "mask_slices"
        preview_directory = root / "preview_slices"
        for directory in (ct_directory, mask_directory, preview_directory):
            directory.mkdir(parents=True, exist_ok=True)
            clean_managed_slice_files(directory)

        number = ct_stack.shape[0]
        digits = max(4, len(str(max(0, number - 1))))
        ct_paths = [""] * number
        mask_paths = [""] * number
        preview_paths = [""] * number
        window = robust_window(ct_stack, mask_stack)

        for index in range(number):
            if export_ct_tiff:
                relative = Path("ct_slices") / f"slice_{index:0{digits}d}.tif"
                write_scalar_tiff(root / relative, ct_stack[index])
                ct_paths[index] = relative.as_posix()
            if export_mask_tiff:
                relative = Path("mask_slices") / f"mask_{index:0{digits}d}.tif"
                write_scalar_tiff(root / relative, mask_stack[index].astype(np.uint8))
                mask_paths[index] = relative.as_posix()
            if export_preview_png:
                relative = Path("preview_slices") / f"slice_{index:0{digits}d}.png"
                preview = make_preview_rgb(ct_stack[index], mask_stack[index], window)
                write_rgb_png(root / relative, preview)
                preview_paths[index] = relative.as_posix()
            if index % max(1, number // 100) == 0 or index == number - 1:
                progress(0.70 * (index + 1) / max(number, 1), "Writing individual slice images")

        matrices = frame_matrices(frames)
        npz_path = root / "slice_frames.npz"
        temporary_npz = root / ".slice_frames.writing.npz"
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(
                stream,
                centers_ras=frames.centers_world,
                x_axes_ras=frames.x_axes_world,
                y_axes_ras=frames.y_axes_world,
                tangents_ras=frames.tangents_world,
                arc_lengths_mm=frames.arc_lengths_mm,
                plane_to_ras=matrices,
                u_coordinates_mm=np.asarray(u_coordinates),
                v_coordinates_mm=np.asarray(v_coordinates),
            )
        os.replace(str(temporary_npz), str(npz_path))

        csv_path = root / "slice_frames.csv"
        matrix_fields = [f"m{row}{column}" for row in range(4) for column in range(4)]
        fieldnames = [
            "slice_index",
            "arc_length_mm",
            "center_ras_x",
            "center_ras_y",
            "center_ras_z",
            "x_axis_ras_x",
            "x_axis_ras_y",
            "x_axis_ras_z",
            "y_axis_ras_x",
            "y_axis_ras_y",
            "y_axis_ras_z",
            "tangent_ras_x",
            "tangent_ras_y",
            "tangent_ras_z",
            *matrix_fields,
            "area_mm2",
            "component_count",
            "touches_field_boundary",
            "centroid_offset_mm",
            "curvature_per_mm",
            "frame_rotation_deg",
            "qc_flags",
            "ct_tiff",
            "mask_tiff",
            "preview_png",
        ]
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index, record in enumerate(slice_records):
                row = {
                    "slice_index": index,
                    "arc_length_mm": frames.arc_lengths_mm[index],
                    "center_ras_x": frames.centers_world[index, 0],
                    "center_ras_y": frames.centers_world[index, 1],
                    "center_ras_z": frames.centers_world[index, 2],
                    "x_axis_ras_x": frames.x_axes_world[index, 0],
                    "x_axis_ras_y": frames.x_axes_world[index, 1],
                    "x_axis_ras_z": frames.x_axes_world[index, 2],
                    "y_axis_ras_x": frames.y_axes_world[index, 0],
                    "y_axis_ras_y": frames.y_axes_world[index, 1],
                    "y_axis_ras_z": frames.y_axes_world[index, 2],
                    "tangent_ras_x": frames.tangents_world[index, 0],
                    "tangent_ras_y": frames.tangents_world[index, 1],
                    "tangent_ras_z": frames.tangents_world[index, 2],
                    "area_mm2": record["area_mm2"],
                    "component_count": record["component_count"],
                    "touches_field_boundary": record["touches_field_boundary"],
                    "centroid_offset_mm": record["centroid_offset_mm"],
                    "curvature_per_mm": record["curvature_per_mm"],
                    "frame_rotation_deg": record["frame_rotation_deg"],
                    "qc_flags": record["flags"],
                    "ct_tiff": ct_paths[index],
                    "mask_tiff": mask_paths[index],
                    "preview_png": preview_paths[index],
                }
                for field, value in zip(matrix_fields, matrices[index].reshape(-1)):
                    row[field] = value
                writer.writerow(row)
        os.replace(str(temporary_csv), str(csv_path))
        progress(0.78, "Writing frame metadata")

        saved_files = {
            "frame_csv": "slice_frames.csv",
            "frame_npz": "slice_frames.npz",
            "quality_control": "qc.json",
            "manifest": "manifest.json",
        }
        if save_straightened:
            ct_path = root / "straightened_ct.nrrd"
            mask_path = root / "straightened_mask.nrrd"
            self._save_node_atomic(ct_node, ct_path)
            self._save_node_atomic(mask_node, mask_path)
            saved_files["straightened_ct"] = ct_path.name
            saved_files["straightened_mask"] = mask_path.name
        if save_segmentation:
            segmentation_path = root / "tooth_segmentation.seg.nrrd"
            self._save_node_atomic(segmentation_node, segmentation_path)
            saved_files["segmentation"] = segmentation_path.name
        if save_curve:
            curve_path = root / "tooth_axis.mrk.json"
            self._save_node_atomic(curve_node, curve_path)
            saved_files["axis_curve"] = curve_path.name
        progress(0.90, "Saving Slicer data nodes")

        qc_document = dict(qc)
        qc_document["warnings"] = list(qc_warnings)
        self._write_json_atomic(root / "qc.json", qc_document)
        manifest = {
            "format": "ToothOrthogonalSlices",
            "format_version": 1,
            "module_version": MODULE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": "RAS (millimetres)",
            "source": {
                "volume_name": volume_node.GetName(),
                "volume_node_id": volume_node.GetID(),
                "segmentation_name": segmentation_node.GetName(),
                "segmentation_node_id": segmentation_node.GetID(),
                "segment_id": str(segment_id),
                "segment_name": segmentation_node.GetSegmentation().GetSegment(segment_id).GetName(),
                "curve_name": curve_node.GetName(),
                "curve_node_id": curve_node.GetID(),
                "array_zyx_to_world_ras": np.asarray(array_to_world).tolist(),
                "axis_estimation_parameters": self._json_node_attribute(
                    curve_node, "ToothOrthogonalSlices.AxisParameters"
                ),
                "axis_estimation_qc": self._json_node_attribute(
                    curve_node, "ToothOrthogonalSlices.AxisQC"
                ),
            },
            "sampling": {
                "number_of_slices": int(number),
                "slice_spacing_mm": float(slice_spacing_mm),
                "terminal_spacing_note": (
                    "The final interval may be shorter than the nominal spacing so that the curve endpoint is included."
                ),
                "pixel_spacing_mm": float(pixel_spacing_mm),
                "field_of_view_mm": float(field_of_view_mm),
                "image_shape_rows_columns": [int(ct_stack.shape[1]), int(ct_stack.shape[2])],
                "ct_interpolation_order": int(interpolation_order),
                "mask_interpolation": "nearest-neighbor",
                "reverse_axis": bool(reverse_axis),
                "rotate_in_plane_180": bool(rotate_in_plane_180),
                "direction_note": (
                    "Slice order follows the selected Markups curve control-point order after optional reversal; "
                    "root/crown identity is not inferred."
                ),
                "frame_method": "rotation-minimizing parallel transport",
            },
            "image_encoding": {
                "ct_tiff": "float32 quantitative resampled scalar values; no display windowing",
                "mask_tiff": "uint8 binary values 0/1",
                "preview_png_window": [float(window[0]), float(window[1])],
                "straightened_volume_note": (
                    "The straightened NRRD uses rectified local coordinates. A single affine cannot encode the "
                    "curved mapping; use slice_frames.csv or slice_frames.npz for source RAS coordinates."
                ),
            },
            "files": saved_files,
            "quality_control_summary": qc_document,
        }
        self._write_json_atomic(root / "manifest.json", manifest)
        incomplete_path.unlink(missing_ok=True)
        progress(1.0, "Finalizing manifest")
        return root


class ToothOrthogonalSlicesTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_frames_and_resampling()

    def setUp(self):
        pass

    def test_frames_and_resampling(self):
        t = np.linspace(0.0, 1.0, 101)
        curve = np.column_stack([5.0 * np.sin(np.pi * t), np.zeros_like(t), 40.0 * t])
        sampled, _ = resample_polyline(curve, 0.5)
        frames = parallel_transport_frames(sampled, [0.0, 1.0, 0.0])
        metrics = validate_frames(frames)
        self.assertLess(metrics["maximum_orthogonality_error"], 1e-6)

        shape = (60, 40, 40)
        volume = np.zeros(shape, dtype=np.float32)
        volume[:, 17:23, 17:23] = 100.0
        matrix = np.eye(4)
        matrix[:3, 3] = [-10.0, -20.0, -20.0]
        straight_curve = np.column_stack(
            [np.linspace(-9.0, 49.0, 60), np.zeros(60), np.zeros(60)]
        )
        straight_frames = parallel_transport_frames(straight_curve, [0.0, 1.0, 0.0])
        stack, _, _ = sample_volume_along_frames(
            volume,
            matrix,
            straight_frames,
            pixel_spacing_mm=1.0,
            field_of_view_mm=20.0,
            interpolation_order=1,
            outside_value=0.0,
        )
        self.assertEqual(stack.shape[0], straight_curve.shape[0])
