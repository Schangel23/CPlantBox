"""Tests for dart.coupling.hydraulics.soil_psi providers (Phase 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixed():
    from dart.coupling.hydraulics.soil_psi import FixedSoilPsi
    return FixedSoilPsi(psi_cm=-500.0)


def _cellidx_linspace(psi_cm, depth_cm):
    """Reversed linspace under the cellidx convention (Phase 3.5+).

    cellidx 0 = bottom of column (drier), cellidx depth-1 = top (wetter
    = ``psi_cm``). Physically identical to the legacy top-first
    ``np.linspace(psi_cm, psi_cm - depth, depth)`` once the seg→cell
    mapping uses ``setRectangularGrid`` (which inverts z relative to the
    legacy ``_picker``).
    """
    return np.linspace(psi_cm - depth_cm, psi_cm, depth_cm)


@pytest.mark.parametrize("psi", [-100.0, -300.0, -500.0, -1500.0])
@pytest.mark.parametrize("depth", [50, 100, 200])
def test_fixed_bit_identical_with_cellidx_linspace(psi, depth):
    """FixedSoilPsi.get_profile must match the cellidx-convention linspace.

    The pre-Phase-3.5 expression was ``np.linspace(psi, psi - depth, depth)``
    indexed by the top-first ``_picker`` (cellidx 0 = top). After Phase 3.5
    the canonical ``setRectangularGrid`` indexing puts cellidx 0 = bottom,
    so the linspace is reversed; physics is unchanged.
    """
    from dart.coupling.hydraulics.soil_psi import FixedSoilPsi

    cellidx = _cellidx_linspace(psi, depth)
    new = FixedSoilPsi(psi_cm=psi, n_cells=depth).get_profile(
        t_days=0.0, depth_cm=depth)
    assert np.array_equal(cellidx, new)


def test_fixed_independent_of_t_days(fixed):
    p0 = fixed.get_profile(0.0, 100)
    p100 = fixed.get_profile(100.0, 100)
    assert np.array_equal(p0, p100)


def test_fixed_update_is_noop(fixed):
    before = fixed.get_profile(0.0, 100)
    fixed.update(t_days=5.0, sink_per_cell=np.full(100, -1e-3))
    after = fixed.get_profile(5.0, 100)
    assert np.array_equal(before, after)


def test_factory_dispatch():
    from dart.coupling.hydraulics.soil_psi import (
        BucketSoilPsi, FixedSoilPsi, SteadyRatePerirhizalPsi, make_provider,
    )
    assert isinstance(make_provider("fixed", soil_psi_cm=-500), FixedSoilPsi)
    assert isinstance(make_provider("bucket", soil_psi_cm=-300), BucketSoilPsi)
    assert isinstance(make_provider("sra", soil_psi_cm=-300),
                      SteadyRatePerirhizalPsi)
    with pytest.raises(ValueError):
        make_provider("nonsense")


def test_bucket_drying_monotonic():
    from dart.coupling.hydraulics.soil_psi import BucketSoilPsi
    b = BucketSoilPsi(psi_init_cm=-200.0, psi_target_cm=-1500.0,
                      tau_days=10.0)
    p0 = b.get_profile(0.0, 100)
    p10 = b.get_profile(10.0, 100)
    p30 = b.get_profile(30.0, 100)
    # cellidx convention: p[-1] = top of column = self.psi (the bucket scalar);
    # p[0] = bottom = psi - depth (a static gradient, time-invariant offset).
    # Drying is monotonic in the scalar self.psi, so check p[-1].
    assert p10[-1] < p0[-1]
    assert p30[-1] < p10[-1]
    # Asymptotes to target at 100 days (well past 3*tau)
    p100 = b.get_profile(100.0, 100)
    assert abs(p100[-1] - (-1500.0)) < 5.0


_DUMUX_BIND = Path(
    "/home/lukas/PHD/dumux-build/dumux/dumux-rosi/build-cmake/cpp/python_binding"
)
_DUMUX_AVAILABLE = (_DUMUX_BIND / "rosi_richards.cpython-314-x86_64-linux-gnu.so").exists()


@pytest.mark.skipif(not _DUMUX_AVAILABLE, reason="rosi_richards.so not built")
def test_dumux_constructs_and_advances():
    from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi

    dum = DumuxSoilPsi(depth_cm=100, n_cells_z=100, psi_init_cm=-100.0,
                       verbose=False)
    p0 = dum.get_profile(t_days=0.0, depth_cm=100)
    p10 = dum.get_profile(t_days=10.0, depth_cm=100)

    assert p0.shape == (100,)
    assert np.all(np.isfinite(p0))
    assert np.all(np.isfinite(p10))
    # cellidx convention: p[0] = bottom (free-drainage BC, tracks BC value),
    # p[-1] = top (zero-flux, dries as drainage propagates upward).
    # Top should be at-or-drier than initial; bottom is constrained by BC.
    assert p10[-1] <= p0[-1] + 1e-9
    assert abs(p10[0] - p0[0]) < 1.0


@pytest.mark.skipif(not _DUMUX_AVAILABLE, reason="rosi_richards.so not built")
def test_dumux_get_profile_rejects_grid_mismatch():
    from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi
    dum = DumuxSoilPsi(depth_cm=100, n_cells_z=100, psi_init_cm=-200.0,
                       verbose=False)
    with pytest.raises(ValueError):
        dum.get_profile(t_days=0.0, depth_cm=50)


def test_provider_protocol_conformance():
    """Each concrete provider satisfies the SoilPsiProvider Protocol."""
    from dart.coupling.hydraulics.soil_psi import (
        BucketSoilPsi, FixedSoilPsi, SoilPsiProvider,
        SteadyRatePerirhizalPsi,
    )
    # Protocol is duck-typed; these calls should be runtime-callable
    for prov in [FixedSoilPsi(-500.0), BucketSoilPsi(),
                 SteadyRatePerirhizalPsi()]:
        prof = prov.get_profile(0.0, 100)
        assert prof.shape == (100,)
        prov.update(0.0, np.zeros(100))  # must not raise


def test_sra_vectorized_interface_matches_scalar_reference():
    """The fast two-input SRA matches the canonical Schröder solve."""
    import plantbox.functional.van_genuchten as vg
    from scipy.optimize import brentq
    from dart.coupling.hydraulics.soil_psi import SteadyRatePerirhizalPsi

    provider = SteadyRatePerirhizalPsi()
    rx = np.array([-800.0, -1200.0, -4000.0, -600.0])
    sx = np.array([-300.0, -500.0, -1000.0, -600.0])
    inner_kr = np.array([2e-6, 1e-5, 5e-5, 0.0])
    rho = np.array([5.0, 20.0, 100.0, 50.0])

    got = provider.interface_potentials(rx, sx, inner_kr, rho)
    sp = provider._soil_parameters()
    mfp = vg.fast_mfp[sp]

    def scalar_reference(rxi, sxi, kri, rhoi):
        if kri < 1e-7 or rxi == sxi:
            return sxi
        rho2 = rhoi * rhoi
        b = 2 * (rho2 - 1) / (
            1 - 0.53**2 * rho2
            + 2 * rho2 * (np.log(rhoi) + np.log(0.53))
        )
        a = kri / b
        target = a * rxi + mfp(sxi)
        return brentq(lambda x: mfp(x) + a * x - target,
                      min(rxi, sxi), max(rxi, sxi))

    expected = np.array([
        scalar_reference(*values)
        for values in zip(rx, sx, inner_kr, rho)
    ])

    assert np.allclose(got, expected, atol=0.01, rtol=0.0)


def test_sra_accepts_soil_hydraulic_calibration():
    from dart.coupling.hydraulics.soil_psi import SteadyRatePerirhizalPsi

    sand = (0.045, 0.43, 0.15, 3.0, 1000.0)
    default = SteadyRatePerirhizalPsi()
    calibrated = SteadyRatePerirhizalPsi(vg_params=sand)

    assert calibrated.vg_params == sand
    assert calibrated._soil_parameters() is calibrated._soil_parameters()
    assert calibrated._soil_parameters() is not default._soil_parameters()


class _FakeSraHydraulics:
    class _MappedSegments:
        @staticmethod
        def getHs(bulk_psi):
            return np.asarray(bulk_psi, dtype=float)

    class _Params:
        psi_air = -15000.0

    def __init__(self, xylem_after_interface_solves):
        self.ms = self._MappedSegments()
        self.params = self._Params()
        self.calls = []
        self._xylem = np.array([-100.0])
        self._updates = iter(xylem_after_interface_solves)

    def solve(self, *, rsx, cells, **kwargs):
        self.calls.append({"rsx": np.asarray(rsx).copy(), "cells": cells, **kwargs})
        if not cells:
            self._xylem = np.array([float(next(self._updates))])

    @staticmethod
    def get_kr(sim_time):
        return np.array([1.0e-4])

    def get_water_potential(self):
        return self._xylem.copy()


def _single_root_geometry(_hm):
    return (
        np.array([2]),
        np.array([0]),
        np.array([0.1]),
        np.array([10.0]),
        np.array([0]),
    )


def test_sra_fixed_point_converges():
    from dart.coupling.hydraulics.soil_psi import SteadyRatePerirhizalPsi

    hm = _FakeSraHydraulics([-200.0, -200.5])
    provider = SteadyRatePerirhizalPsi(
        n_cells=1, max_iterations=3, tolerance_cm=1.0,
    )
    provider._root_geometry = _single_root_geometry
    provider.interface_potentials = lambda rx, sx, inner_kr, rho: sx.copy()

    provider.solve_photosynthesis(hm, np.array([-500.0]), sim_time=1.0)

    assert [call["cells"] for call in hm.calls] == [True, False, False]
    assert provider.last_iterations == 2
    assert provider.last_error_cm == pytest.approx(0.5)


def test_sra_fixed_point_reports_non_convergence():
    from dart.coupling.hydraulics.soil_psi import SteadyRatePerirhizalPsi

    hm = _FakeSraHydraulics([-200.0, -300.0])
    provider = SteadyRatePerirhizalPsi(
        n_cells=1, max_iterations=2, tolerance_cm=1.0,
    )
    provider._root_geometry = _single_root_geometry
    provider.interface_potentials = lambda rx, sx, inner_kr, rho: sx.copy()

    with pytest.raises(RuntimeError, match="did not converge after 2 iterations"):
        provider.solve_photosynthesis(hm, np.array([-500.0]), sim_time=1.0)


def test_photosynthesis_dispatch_selects_sra_solver():
    from dart.coupling.hydraulics.soil_psi import (
        FixedSoilPsi, SteadyRatePerirhizalPsi,
        solve_photosynthesis_with_soil_psi,
    )

    class FakeHydraulics:
        def __init__(self):
            self.calls = []

        def solve(self, **kwargs):
            self.calls.append(kwargs)

    hm = FakeHydraulics()
    bulk = np.array([-500.0])
    solve_photosynthesis_with_soil_psi(
        hm, FixedSoilPsi(n_cells=1), bulk, sim_time=1.0,
    )
    assert len(hm.calls) == 1
    assert hm.calls[0]["rsx"] is bulk
    assert hm.calls[0]["cells"] is True
    assert hm.calls[0]["sim_time"] == 1.0

    sra = SteadyRatePerirhizalPsi(n_cells=1)
    called = []
    sra.solve_photosynthesis = lambda model, psi, **kwargs: called.append(
        (model, psi, kwargs),
    )
    solve_photosynthesis_with_soil_psi(hm, sra, bulk, sim_time=2.0)
    assert len(called) == 1
    assert called[0][0] is hm
    assert called[0][1] is bulk
    assert called[0][2] == {"sim_time": 2.0}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
