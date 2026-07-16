"""Regression coverage for maturity-delayed FA internode cessation."""

from pathlib import Path

from dart.coupling.growth.grow import grow_plant


XML = (Path(__file__).parents[1] / "data" / "maize_mirza_fused_train.xml")
SAMPLE_TIMES = (68.5, 82.9)


def _collar_delta(tmp_path, extend_cessation):
    text = XML.read_text()
    anchor = '        <parameter name="internode_maturity_span" value="756.222578284797" />'
    replacement = (
        f'{anchor}\n        <parameter name="internode_maturity_extends_cessation" '
        f'value="{int(extend_cessation)}" />'
    )
    assert anchor in text
    xml = tmp_path / f"maturity_cessation_{int(extend_cessation)}.xml"
    xml.write_text(text.replace(anchor, replacement, 1))

    collars = {}

    def capture(plant, time):
        leaf = next(
            organ for organ in plant.getOrgans()
            if organ.organType() == 4 and int(organ.getParameter("subType")) == 5
        )
        parent = leaf.getParent()
        assert int(leaf.getNodeId(0)) == int(parent.getNodeId(leaf.parentNI))
        assert abs(float(leaf.getNode(0).z) - float(parent.getNode(leaf.parentNI).z)) < 1e-12
        collars[float(time)] = float(leaf.getNode(0).z)

    grow_plant(
        str(xml), simulation_time=max(SAMPLE_TIMES), seed=42, daily_met={},
        snapshot_times=SAMPLE_TIMES, snapshot_callback=capture,
    )
    return collars[SAMPLE_TIMES[1]] - collars[SAMPLE_TIMES[0]]


def test_maturity_window_can_extend_operational_cessation(tmp_path):
    """Opt-in keeps a delayed internode moving after bare Phase IV completion."""
    assert abs(_collar_delta(tmp_path, extend_cessation=False)) < 1e-12
    assert _collar_delta(tmp_path, extend_cessation=True) > 0.1
