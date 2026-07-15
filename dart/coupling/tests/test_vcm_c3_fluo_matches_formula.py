"""Validate the C++ C3 fluorescence (Photosynthesis::photoC3_fluo, c3Model=1) against an
independent Python restatement of the bolt-on formula.

C3 keeps CPlantBox's FvCB carbon untouched; the fluorescence is added on top:
    Ja  = J * (An + Rd) / Vj   (clipped to [0, min(J, Jmax*qLs)])
    eta = MD12(Ja, Jms=Jmax*qLs) / fo0     with beta_c3 = 0.507
This test grows maize, forces the C3 pathway (PhotoType=0) so the C3 branch executes, then
recomputes Ja/eta in Python from the exact per-segment state the C++ used and asserts a match.
This guards the unit handling and the MD12 wiring on the C3 path.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve

XML = os.path.join(os.path.dirname(__file__), "..", "data", "maize_calibrated.xml")

# must match Photosynthesis.h defaults
KF, KD, KD_DARK, PO0MAX, BETA_C3 = 3.0e7, 1.0e8, 1.95e8, 0.88, 0.507


def md12_fs(ps, Ja, Jms, kps, kf, kds, kDs):
    fs1 = ps * (kf / kps) / (1.0 - Ja / Jms)
    par1 = kps / (kps - kds)
    par2 = par1 * (kf + kDs + kds) / kf
    fs2 = (par1 - ps) / par2
    return np.minimum(fs1, fs2)


def main():
    plant = grow_plant(os.path.normpath(XML), simulation_time=55, seed=42)
    res = run_photosynthesis_solve(plant, sim_time=1, par=1200.0, tleaf=25.0,
                                   label="c3-fluo-check", c3_model=1, photo_type=0)

    eta_cpp = np.asarray(res["eta"], dtype=float)
    Ja_cpp = np.asarray(res["Ja"], dtype=float) * 1e6   # mol -> umol
    vin = res.get("_vcm_inputs")
    assert vin is not None, "C3 branch not exercised"
    n = len(eta_cpp)
    assert n > 0

    J = np.asarray(vin["J"], dtype=float) * 1e6
    Vj = np.asarray(vin["Vj"], dtype=float) * 1e6
    An = np.asarray(vin["An"], dtype=float) * 1e6
    Rd = np.asarray(vin["Rd"], dtype=float) * 1e6
    Jmax = np.asarray(vin["Jmax"], dtype=float) * 1e6
    Ql = np.asarray(vin["Qlight"], dtype=float)
    Q = (Ql if len(Ql) == n else np.full(n, Ql[0])) * 1e6
    Q = np.where(Q <= 0, 1e-9, Q)

    qLs, po0m = 1.0, PO0MAX
    kPSII = (KD + KF) * po0m / (1.0 - po0m)
    fo0 = KF / (KF + kPSII + KD)
    kps, kds, kDs = kPSII * qLs, KD_DARK * qLs, KD + 0.0
    Jms = np.maximum(Jmax * qLs, 1e-9)
    frac = np.where(Vj > 1e-12, np.clip((An + Rd) / np.where(Vj > 1e-12, Vj, 1.0), 0.0, 1.0), 1.0)
    Ja_ref = np.clip(J * frac, 0.0, np.minimum(J, Jms))
    ps = Ja_ref / (BETA_C3 * Q)
    eta_ref = md12_fs(ps, Ja_ref, Jms, kps, KF, kds, kDs) / fo0

    print(f"segments: {n}")
    print(f"eta C++ [{eta_cpp.min():.3f},{eta_cpp.max():.3f}]  ref [{eta_ref.min():.3f},{eta_ref.max():.3f}]")
    print(f"Ja  C++ [{Ja_cpp.min():.1f},{Ja_cpp.max():.1f}]  ref [{Ja_ref.min():.1f},{Ja_ref.max():.1f}]")

    assert np.all(np.isfinite(eta_cpp)) and np.all((eta_cpp > 0) & (eta_cpp < 5)), "C3 eta out of range"
    assert np.all(Ja_cpp <= J + 1e-6), "Ja exceeds potential J"
    for name, a, b in [("Ja", Ja_cpp, Ja_ref), ("eta", eta_cpp, eta_ref)]:
        dmax = np.abs(a - b).max()
        rmax = (np.abs(a - b) / (np.abs(b) + 1e-9)).max()
        print(f"{name}: max|abs|={dmax:.3e}  max|rel|={rmax:.3e}")
        assert np.allclose(a, b, rtol=2e-3, atol=1e-4), f"{name} C++ vs reference mismatch"

    print("OK: C++ C3 fluorescence matches the Python restatement (Ja, eta)")


def test_vcm_c3_fluo_matches_formula():
    main()


if __name__ == "__main__":
    main()
