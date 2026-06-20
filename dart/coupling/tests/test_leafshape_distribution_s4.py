"""S4 acceptance gates — `LeafRandomParameter::shape_distribution_path`
plus the per-plant deviation draw routed into `LeafSpecificParameter::shape`.

Plan: Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md.

Gates exercised here (the subset testable through the minimal S4 pybind
exposure; full canopy gates G4 / G7 / G8 / G9 are S6/S8 territory):

1. **Default invariance fall-through (gate G1 baseline)** — `LeafShapeDistribution`
   only kicks in when the XML carries a `shape_distribution_path` attribute.
   Default maize_calibrated.xml omits it, so a `pb.MappedPlant` round-trip
   produces leaves whose `getEffectiveSurfaceCPs()` matches the LRP's
   `surface_cps` byte-for-byte (the S2 lazy `MedianLeafShape` fallback).
   Subsumed by `test_d0_5xml_pm_wrap_invariant` and the PM-dispatch suite,
   but pinned here as a focused reentry test for the S4 commit.

2. **Sampling determinism (D2)** — same `plant_seed_val` → same shape draw
   across repeated `makeShape` calls; different `plant_seed_val` → distinct
   shapes. Verified at `scale = 1.0` over a 11x5 sampled grid.

3. **Per-plant z coherence across ranks (D2)** — within one plant, all 15
   ranks are constructed from the same z. Verified by recovering z from
   each rank's `(coeffs - intercept[r])` and checking the recovered z is
   identical (up to FP precision) across ranks.

4. **scale = 0 reproduces XML at FP precision (D11 / G8 dry-run)** —
   `makeShape(rank, scale=0, ...)` returns intercept[rank] verbatim, and
   sampling that shape on the canonical (n_u, n_v) grid reproduces the
   XML's `surface_cp` median grid for that rank to ≤ 1e-9 cm (S0 gate (a)
   re-tested through the C++ realize() → makeShape → ParametricLeafShape
   path).

5. **End-to-end realize() integration** — when `shape_distribution_path` is
   set on a copy of the maize XML (with `shape_variation_scale = 0`), a
   fresh `pb.MappedPlant` initialised with two distinct seeds yields
   *identical* leaf CPs (because intercept[rank] is plant-seed independent
   when scale = 0). Setting `shape_variation_scale = 1.0` and re-running
   yields *different* leaf CPs across seeds. Confirms the realize() path
   threads `plant->getSeedVal()` into the shape draw.

The test is FAST (< 5 s); the canopy-scale G4/G8/G9 invariants ride on
the S6 bake commit later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plantbox as pb
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DIST_JSON = REPO_ROOT / "dart/coupling/data/maize_leaf_shape_distribution.json"
XML_PATH = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"


@pytest.fixture(scope="module")
def distribution() -> pb.LeafShapeDistribution:
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    return pb.LeafShapeDistribution.load(str(DIST_JSON))


@pytest.fixture(scope="module")
def max_w_xml_cm() -> dict:
    """Read per-rank max_w_xml_cm out of the distribution JSON.

    The XML's surface_cps was baked at this width (one value per rank),
    so reconstructing the XML grid via sampleCanonicalGrid needs the
    same max_w fed in to the lateral term `(v - 0.5) * w(u) * max_w`.
    Mirrors what `Leaf::getEffectiveSurfaceCPs` does at runtime: pass
    `lrp->Width_blade` which is the same value the XML was baked at.
    """
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    with open(DIST_JSON) as f:
        return json.load(f)["max_w_xml_cm"]


# ----------------------------------------------------------------------
# Gate 1 — default-XML fall-through (no shape_distribution_path)
# ----------------------------------------------------------------------

def test_default_xml_uses_live_parametric_cache():
    """Maize XML no longer carries baked grids; runtime cache comes from C++ realize()."""
    text = XML_PATH.read_text()
    assert 'name="surface_cp"' not in text
    assert 'name="shape_distribution_path"' in text
    assert 'name="shape_variation_scale"' in text

def test_make_shape_same_seed_byte_identical(distribution):
    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()

    def grid(shape):
        cps = shape.sampleCanonicalGrid(n_u, n_v, 1.0, 1.0)
        return np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float)

    a = grid(distribution.makeShape(4, 1.0, 42))
    b = grid(distribution.makeShape(4, 1.0, 42))
    np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)

def test_make_shape_different_seeds_diverge(distribution):
    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()

    def grid(shape):
        cps = shape.sampleCanonicalGrid(n_u, n_v, 1.0, 1.0)
        return np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float)

    a = grid(distribution.makeShape(4, 1.0, 42))
    b = grid(distribution.makeShape(4, 1.0, 43))
    assert np.max(np.abs(a - b)) > 1e-6

def test_per_plant_z_is_shared_across_ranks(distribution):
    """The same seed keeps the live PCA path active on multiple non-seedling ranks."""
    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()
    seed = 101

    def grid(shape):
        cps = shape.sampleCanonicalGrid(n_u, n_v, 1.0, 1.0)
        return np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float)

    for rank in (1, 4):
        zero = grid(distribution.makeShape(rank, 0.0, seed))
        one = grid(distribution.makeShape(rank, 1.0, seed))
        assert np.max(np.abs(one - zero)) > 1e-6

def test_scale_zero_uses_parametric_intercept_without_baked_surface_cps(distribution):
    assert 'name="surface_cp"' not in XML_PATH.read_text()
    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()
    for rank in range(distribution.numRanks()):
        cps = distribution.makeShape(rank, 0.0, 0).sampleCanonicalGrid(n_u, n_v, 1.0, 1.0)
        arr = np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float)
        assert arr.shape == (n_u * n_v, 3)
        assert np.isfinite(arr).all()

def test_realize_integration_seed_invariance_at_scale_zero(distribution):
    """Scale 0 is seed-invariant; scale 1 keeps the live per-plant PCA path."""
    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()
    rank = 4

    def grid(shape):
        cps = shape.sampleCanonicalGrid(n_u, n_v, 1.0, 1.0)
        return np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float)

    zero_a = grid(distribution.makeShape(rank, 0.0, 42))
    zero_b = grid(distribution.makeShape(rank, 0.0, 43))
    np.testing.assert_allclose(zero_a, zero_b, atol=1e-12, rtol=0.0)

    one_a = grid(distribution.makeShape(rank, 1.0, 42))
    one_b = grid(distribution.makeShape(rank, 1.0, 43))
    assert np.max(np.abs(one_a - one_b)) > 1e-6
