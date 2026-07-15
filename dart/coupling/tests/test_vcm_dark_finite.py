"""Dark-state regression: at zero light the fluorescence outputs (eta, Ja) must stay finite
and bounded for both c4Model=1 (two-cell C4) and c3Model=1 (FvCB C3).

Zero / near-zero light is a normal diurnal state (night, deep shade). The fluorescence code
clamps Qlight to 1e-9 before the ps = Ja/(beta*Q) division (Photosynthesis.cpp:705 and :837),
so no 0/0 occurs; this test locks that in so a future edit removing the clamp is caught.
Expected dark behaviour: Ja -> 0, eta -> 0 (both finite, non-negative, bounded).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve

XML = os.path.join(os.path.dirname(__file__), "..", "data", "maize_calibrated.xml")


def _check(label, **kwargs):
    plant = grow_plant(os.path.normpath(XML), simulation_time=50, seed=42)
    res = run_photosynthesis_solve(plant, sim_time=1, par=0.0, tleaf=20.0,
                                   label=label, **kwargs)
    eta = np.asarray(res["eta"], dtype=float)
    Ja = np.asarray(res["Ja"], dtype=float)
    assert len(eta) > 0, f"{label}: no leaf segments"
    print(f"{label}: n={len(eta)} eta[min={eta.min():.3g},max={eta.max():.3g}] "
          f"Ja[min={Ja.min():.3g},max={Ja.max():.3g}]")
    assert np.all(np.isfinite(eta)), f"{label}: non-finite eta at zero light"
    assert np.all(np.isfinite(Ja)), f"{label}: non-finite Ja at zero light"
    assert np.all((eta >= 0.0) & (eta < 5.0)), f"{label}: eta out of [0,5) at zero light"
    assert np.all(Ja >= -1e-12), f"{label}: negative Ja at zero light"


def main():
    _check("C4-dark", c4_model=1)            # maize is C4
    _check("C3-dark", c3_model=1, photo_type=0)  # force C3 pathway
    print("OK: dark-state fluorescence finite and bounded for C3 and C4")


def test_vcm_dark_finite():
    main()


if __name__ == "__main__":
    main()
