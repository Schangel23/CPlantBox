"""Native LeafShape grids realize their requested mature dimensions."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import plantbox as pb

from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.nurbs_blade import loft_leaf_nurbs


REPO = Path(__file__).resolve().parents[3]


def _dimensions(shape, n_u: int, n_v: int, length: float, width: float):
    grid = np.asarray(
        shape.sampleCanonicalGrid(n_u, n_v, length, width), dtype=float
    ).reshape(n_u, n_v, 3)
    midrib = grid[:, n_v // 2]
    arc = np.linalg.norm(np.diff(midrib, axis=0), axis=1).sum()
    max_width = np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1).max()
    return arc, max_width


def test_supported_leaf_shapes_materialize_requested_mature_dimensions():
    n_u, n_v = 11, 5
    z = np.linspace(0.0, 10.0, n_u)
    x = np.linspace(-1.0, 1.0, n_v)
    median = pb.MedianLeafShape(
        [pb.Vector3d(xv, 0.0, zv) for zv in z for xv in x], n_u, n_v
    )
    distribution = pb.LeafShapeDistribution.load(
        str(REPO / "dart/coupling/data/m07_leaf_shape_dist.json")
    )
    parametric = distribution.makeShape(4, 0.0, 42)

    for shape in (median, parametric):
        arc, max_width = _dimensions(shape, n_u, n_v, 24.0, 6.0)
        np.testing.assert_allclose((arc, max_width), (24.0, 6.0), atol=1e-9)


def test_leaf_effective_grid_uses_specific_size_and_current_length(tmp_path):
    tree = ET.parse(REPO / "dart/coupling/data/maize_calibrated.xml")
    leaf_rp = next(leaf for leaf in tree.getroot().iter("leaf")
                   if leaf.get("subType") == "7")
    width_rp = next(p for p in leaf_rp if p.get("name") == "Width_blade")
    width_rp.set("dev", "0.75")
    xml = tmp_path / "maize_width_dev.xml"
    tree.write(xml)

    plant = grow_plant(
        str(xml), simulation_time=80, seed=42, enable_photosynthesis=False
    )
    leaf = next(
        organ for organ in plant.getOrgans(4)
        if organ.getNumberOfNodes() >= 2 and organ.getParameter("subType") == 7
    )
    current_length = float(leaf.getLength(True))
    mature_length = float(leaf.getParameter("k"))
    assert current_length / mature_length > 0.999

    lrp = leaf.getLeafRandomParameter()
    grid = np.asarray(leaf.getEffectiveSurfaceCPs(), dtype=float).reshape(
        int(lrp.surface_n_u), int(lrp.surface_n_v), 3
    )
    midrib = grid[:, grid.shape[1] // 2]
    arc = np.linalg.norm(np.diff(midrib, axis=0), axis=1).sum()
    max_width = np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1).max()
    np.testing.assert_allclose(
        (arc, max_width),
        (current_length, float(leaf.getParameter("width_blade"))),
        atol=1e-6,
    )

    extracted = extract_organs_for_lofter(plant, species="maize")
    node_ids = [int(node_id) for node_id in leaf.getNodeIds()]
    organ = next(
        item for item in extracted
        if item.get("type") == "leaf" and item.get("node_ids") == node_ids
    )
    np.testing.assert_allclose(organ["surface_cps_local"], grid, atol=0.0, rtol=0.0)


def test_lofter_tessellates_physical_surface_without_rescaling():
    n_u, n_v = 11, 5
    z = np.linspace(0.0, 10.0, n_u)
    x = np.linspace(-1.0, 1.0, n_v)
    grid = np.asarray([[(xv, 0.0, zv) for xv in x] for zv in z])
    organ = {
        "organ_id": 1,
        "surface_cps_local": grid,
        "raw_donor": True,
        "mature_length": 20.0,
        "current_length": 5.0,
        "collar_pos": np.zeros(3),
        "collar_tangent": np.array([0.0, 0.0, 1.0]),
        "skeleton": grid[:, n_v // 2],
    }

    tessellated = loft_leaf_nurbs(organ)[8]
    midrib = tessellated[:, n_v // 2]
    arc = np.linalg.norm(np.diff(midrib, axis=0), axis=1).sum()
    max_width = np.linalg.norm(tessellated[:, -1] - tessellated[:, 0], axis=1).max()
    np.testing.assert_allclose((arc, max_width), (10.0, 2.0), atol=1e-9)
