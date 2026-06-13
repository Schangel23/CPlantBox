import math

import pytest

from dart.coupling.prospect_params import (
    cab_from_vcmax25,
    get_prospect_params_per_relative_height,
    relative_heights_from_leaf_organs,
    vcmax25_from_cab,
    wang2026_vcmax_multiplier,
)


def test_wang2026_vcmax_multiplier_is_top_anchored_and_declines_downward():
    assert wang2026_vcmax_multiplier(1.0) == pytest.approx(1.0)
    assert wang2026_vcmax_multiplier(0.0) == pytest.approx(math.exp(-0.64))
    assert wang2026_vcmax_multiplier(0.5) > wang2026_vcmax_multiplier(0.0)
    assert wang2026_vcmax_multiplier(0.5) < wang2026_vcmax_multiplier(1.0)


def test_lops_profiles_can_be_interpolated_by_relative_height(monkeypatch):
    monkeypatch.setenv("COUPLING_SPECIES", "maize")

    params = get_prospect_params_per_relative_height(
        55.0, [0.0, 0.5, 1.0], apply_wang_vcmax_profile=False)

    assert [p["Cab"] for p in params] == pytest.approx([15.0, 47.0, 25.0])
    assert [p["N"] for p in params] == pytest.approx([1.20, 1.60, 1.35])


def test_leaf_organs_are_mapped_by_attachment_height():
    class Node:
        def __init__(self, z):
            self.z = z

    class Leaf:
        def __init__(self, z):
            self._nodes = [Node(z)]

        def getNodes(self):
            return self._nodes

    leaves = [Leaf(40.0), Leaf(10.0), Leaf(70.0)]

    assert relative_heights_from_leaf_organs(leaves) == pytest.approx(
        [0.5, 0.0, 1.0])


def test_wang2026_profile_sets_effective_chl_from_top_canopy_vcmax(monkeypatch):
    monkeypatch.setenv("COUPLING_SPECIES", "maize")

    params = get_prospect_params_per_relative_height(
        55.0, [0.0, 1.0], apply_wang_vcmax_profile=True)

    top_vcmax = vcmax25_from_cab(25.0)
    expected_bottom_vcmax = top_vcmax * math.exp(-0.64)

    assert params[1]["Vcmax25"] == pytest.approx(top_vcmax)
    assert params[1]["Cab"] == pytest.approx(25.0)
    assert params[0]["Vcmax25"] == pytest.approx(expected_bottom_vcmax)
    assert params[0]["Cab"] == pytest.approx(cab_from_vcmax25(expected_bottom_vcmax))
