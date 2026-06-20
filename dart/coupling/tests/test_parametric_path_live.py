from pathlib import Path
import re

import numpy as np

from dart.coupling.growth.grow import grow_plant


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


def _arch_span(xml: Path) -> float:
    plant = grow_plant(str(xml), simulation_time=45, seed=7)
    spans = []
    for leaf in plant.getOrgans():
        if leaf.organType() != 4:
            continue
        cps = leaf.getEffectiveSurfaceCPs()
        if not cps:
            continue
        arr = np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float)
        spans.append(float(arr[:, 1].max() - arr[:, 1].min()))
    assert spans
    return max(spans)


def test_maize_parametric_path_live_and_curvature_knob(tmp_path):
    text = MAIZE_XML.read_text()
    assert 'name="surface_cp"' not in text
    scales = [float(v) for v in re.findall(r'name="shape_variation_scale" value="([^"]+)"', text)]
    assert scales and all(v > 0.0 for v in scales)

    low = tmp_path / "maize_low_curvature.xml"
    high = tmp_path / "maize_high_curvature.xml"
    _patch_scalar(MAIZE_XML, low, "midrib_curvature_scale", 0.5)
    _patch_scalar(MAIZE_XML, high, "midrib_curvature_scale", 1.5)

    assert _arch_span(high) > _arch_span(low) * 1.15
