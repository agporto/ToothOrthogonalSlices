from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from ToothAxisAlgorithms import parallel_transport_frames


def _load_slicer_module(monkeypatch):
    qt = types.ModuleType("qt")
    qt.Qt = types.SimpleNamespace(RichText=1)
    ctk = types.ModuleType("ctk")

    slicer = types.ModuleType("slicer")
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")

    class Base:
        def __init__(self, *args, **kwargs):
            pass

    for name in (
        "ScriptedLoadableModule",
        "ScriptedLoadableModuleLogic",
        "ScriptedLoadableModuleTest",
        "ScriptedLoadableModuleWidget",
    ):
        setattr(scripted, name, Base)

    slicer.ScriptedLoadableModule = scripted
    slicer.util = types.SimpleNamespace()
    slicer.mrmlScene = None
    monkeypatch.setitem(sys.modules, "qt", qt)
    monkeypatch.setitem(sys.modules, "ctk", ctk)
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    monkeypatch.setitem(sys.modules, "slicer.ScriptedLoadableModule", scripted)

    path = MODULE_DIR / "ToothOrthogonalSlices.py"
    spec = importlib.util.spec_from_file_location("ToothOrthogonalSlicesSlicerSmoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, slicer


def test_logic_import_memory_and_quality_control(monkeypatch):
    module, _ = _load_slicer_module(monkeypatch)
    logic = module.ToothOrthogonalSlicesLogic()

    estimated, side = logic._estimate_memory_bytes(100, 20.0, 0.25)
    assert side == 81
    assert estimated == 100 * 81 * 81 * 12

    points = np.column_stack(
        [np.zeros(12), np.zeros(12), np.linspace(0.0, 11.0, 12)]
    )
    frames = parallel_transport_frames(points, [1.0, 0.0, 0.0])
    mask = np.zeros((12, 41, 41), dtype=bool)
    yy, xx = np.ogrid[:41, :41]
    radii = [8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8]
    for index, radius in enumerate(radii):
        mask[index] = (xx - 20) ** 2 + (yy - 20) ** 2 <= radius**2

    records, qc, warnings = logic._slice_quality_control(frames, mask, 0.25, 10.0)
    assert len(records) == 12
    assert 3 in qc["abrupt_area_change_slice_indices"]
    assert "abrupt_area_change" in records[3]["flags"]
    assert any("Abrupt area" in warning for warning in warnings)


def test_volume_array_world_matrix_and_non_displayable_output_node(monkeypatch):
    module, slicer = _load_slicer_module(monkeypatch)
    logic = module.ToothOrthogonalSlicesLogic()

    array = np.zeros((3, 4, 5), dtype=np.int16)
    slicer.util.arrayFromVolume = lambda node: array

    class ParentTransform:
        def GetMatrixTransformToWorld(self, matrix):
            matrix.Identity()
            matrix.SetElement(0, 3, 10.0)
            matrix.SetElement(1, 3, -2.0)
            return True

    class Volume:
        def GetImageData(self):
            return object()

        def GetIJKToRASMatrix(self, matrix):
            matrix.Identity()
            matrix.SetElement(0, 0, 0.5)
            matrix.SetElement(1, 1, 0.6)
            matrix.SetElement(2, 2, 0.7)

        def GetParentTransformNode(self):
            return ParentTransform()

    values, array_to_world = logic.volume_array_and_world_matrix(Volume(), copy_array=False)
    assert values is array
    # NumPy z maps through K (IJK z), y through J, and x through I.
    assert np.allclose(array_to_world[:3, 0], [0.0, 0.0, 0.7])
    assert np.allclose(array_to_world[:3, 1], [0.0, 0.6, 0.0])
    assert np.allclose(array_to_world[:3, 2], [0.5, 0.0, 0.0])
    assert np.allclose(array_to_world[:3, 3], [10.0, -2.0, 0.0])

    class TableLikeNode:
        def __init__(self):
            self.attributes = {}
            self.name = ""

        def SetAttribute(self, key, value):
            self.attributes[key] = value

        def GetAttribute(self, key):
            return self.attributes.get(key)

        def SetName(self, name):
            self.name = name

    created = TableLikeNode()
    slicer.util.getNodesByClass = lambda class_name: []
    slicer.mrmlScene = types.SimpleNamespace(
        AddNewNodeByClass=lambda class_name, name: created
    )
    output = logic._get_or_create_output_node("vtkMRMLTableNode", "frames", "Frames")
    assert output is created
    assert created.attributes[module.OUTPUT_ATTRIBUTE] == "frames"
    # The node deliberately has no CreateDefaultDisplayNodes method.

    created.SetAttribute("json", json.dumps({"value": 4}))
    assert logic._json_node_attribute(created, "json") == {"value": 4}


def test_export_bundle_is_complete_and_self_describing(monkeypatch, tmp_path):
    module, slicer = _load_slicer_module(monkeypatch)
    logic = module.ToothOrthogonalSlicesLogic()

    def save_node(node, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mock-slicer-node\n")
        return True

    slicer.util.saveNode = save_node

    class NamedNode:
        def __init__(self, name, node_id):
            self.name = name
            self.node_id = node_id
            self.attributes = {}

        def GetName(self):
            return self.name

        def GetID(self):
            return self.node_id

        def GetAttribute(self, key):
            return self.attributes.get(key)

    class Segment:
        def GetName(self):
            return "Tooth"

    class Segmentation:
        def GetSegment(self, segment_id):
            return Segment() if segment_id == "tooth" else None

    class SegmentationNode(NamedNode):
        def GetSegmentation(self):
            return Segmentation()

    volume_node = NamedNode("Horse Tooth 01", "vtkMRMLScalarVolumeNode1")
    segmentation_node = SegmentationNode(
        "Horse Tooth 01 Segmentation", "vtkMRMLSegmentationNode1"
    )
    curve_node = NamedNode("Horse Tooth 01 Axis", "vtkMRMLMarkupsCurveNode1")
    curve_node.attributes["ToothOrthogonalSlices.AxisParameters"] = json.dumps(
        {"threshold_method": "otsu", "coarse_spacing_mm": 1.25}
    )
    curve_node.attributes["ToothOrthogonalSlices.AxisQC"] = json.dumps(
        {"inside_fraction": 1.0}
    )

    points = np.column_stack(
        [np.zeros(4), np.zeros(4), np.linspace(0.0, 3.0, 4)]
    )
    frames = parallel_transport_frames(points, [1.0, 0.0, 0.0])
    ct_stack = np.arange(4 * 9 * 9, dtype=np.float32).reshape(4, 9, 9)
    mask_stack = np.zeros((4, 9, 9), dtype=bool)
    mask_stack[:, 2:7, 2:7] = True
    records, qc, warnings = logic._slice_quality_control(
        frames, mask_stack, pixel_spacing_mm=0.5, field_of_view_mm=4.0
    )
    coordinates = (np.arange(9) - 4) * 0.5

    root = logic._export_results(
        volume_node=volume_node,
        segmentation_node=segmentation_node,
        segment_id="tooth",
        curve_node=curve_node,
        ct_node=NamedNode("straight ct", "ct"),
        mask_node=NamedNode("straight mask", "mask"),
        ct_stack=ct_stack,
        mask_stack=mask_stack,
        frames=frames,
        slice_records=records,
        qc=qc,
        qc_warnings=warnings,
        array_to_world=np.eye(4),
        u_coordinates=coordinates,
        v_coordinates=coordinates,
        output_parent=str(tmp_path),
        pixel_spacing_mm=0.5,
        slice_spacing_mm=1.0,
        field_of_view_mm=4.0,
        interpolation_order=1,
        reverse_axis=False,
        rotate_in_plane_180=False,
        export_ct_tiff=True,
        export_mask_tiff=True,
        export_preview_png=True,
        save_straightened=True,
        save_segmentation=True,
        save_curve=True,
    )

    assert root == tmp_path / "Horse_Tooth_01_orthogonal_slices"
    assert not (root / ".export_incomplete.json").exists()
    assert not (root / ".writing").exists()
    assert len(list((root / "ct_slices").glob("*.tif"))) == 4
    assert len(list((root / "mask_slices").glob("*.tif"))) == 4
    assert len(list((root / "preview_slices").glob("*.png"))) == 4
    assert (root / "straightened_ct.nrrd").is_file()
    assert (root / "straightened_mask.nrrd").is_file()
    assert (root / "tooth_segmentation.seg.nrrd").is_file()
    assert (root / "tooth_axis.mrk.json").is_file()

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format"] == "ToothOrthogonalSlices"
    assert manifest["sampling"]["number_of_slices"] == 4
    assert manifest["source"]["axis_estimation_parameters"]["threshold_method"] == "otsu"
    assert manifest["source"]["axis_estimation_qc"]["inside_fraction"] == 1.0
    with np.load(root / "slice_frames.npz") as bundle:
        assert bundle["plane_to_ras"].shape == (4, 4, 4)
    assert len((root / "slice_frames.csv").read_text().splitlines()) == 5
