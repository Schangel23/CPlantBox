from pathlib import Path
import re

import plantbox as pb


DATA = Path(__file__).resolve().parents[1] / "data"
MAIZE_XML = DATA / "maize_calibrated.xml"


def _patch_scalar(src: Path, dst: Path, name: str, value: float) -> None:
    text = src.read_text()
    text = re.sub(
        rf'(<parameter name="{name}" value=")[^"]+(" */>)',
        rf"\g<1>{value}\g<2>",
        text,
    )
    dst.write_text(text)


def _canonical_arch_span(scale: float) -> float:
    dist = pb.LeafShapeDistribution.load(str(DATA / "plant01_leaf_shape_dist.json"))
    shape = dist.makeShape(4, 0.0, 7, scale)
    n_u = dist.numCpsU()
    n_v = dist.numCpsV()
    cps = shape.sampleCanonicalGrid(n_u, n_v, 1.0, 1.0)
    mid = n_v // 2
    ys = [cps[i * n_v + mid].y for i in range(n_u)]
    return max(abs(y) for y in ys)


def test_maize_parametric_path_live_and_curvature_knob(tmp_path):
    lrp = pb.LeafRandomParameter(pb.MappedPlant())
    assert lrp.midrib_curvature_scale == 1.0
    assert lrp.young_fade_end == 0.7
    assert lrp.young_template_curvature_scale == 0.0

    text = MAIZE_XML.read_text()
    assert 'name="surface_cp"' not in text
    scales = [float(v) for v in re.findall(r'name="shape_variation_scale" value="([^"]+)"', text)]
    assert scales and all(v > 0.0 for v in scales)

    patched = tmp_path / "maize_curvature.xml"
    _patch_scalar(MAIZE_XML, patched, "midrib_curvature_scale", 0.5)
    plant = pb.MappedPlant()
    plant.readParameters(str(patched))
    assert plant.getOrganRandomParameter(4, 2).midrib_curvature_scale == 0.5

    assert _canonical_arch_span(1.5) > _canonical_arch_span(0.5) * 1.15
