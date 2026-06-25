"""Cross-validate the C++ TWO_CELL_VCM branch (Photosynthesis::photoC4_loop_vcm) against
the Python reference oracle (vcm_c4_reference.vcm_c4) on a real grown maize plant.

Both implement the same von Caemmerer (2000) two-cell C4 + Magnani-Difazio fluorescence.
We grow maize, run the coupled photosynthesis solve with c4_model=1, then feed the exact
per-segment inputs the C++ used (ci, Qlight, TleafK, Vcrefmax) into the oracle and assert
An / Ja / eta agree. This is the bit-for-bit guard on the port's unit conversions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve
from dart.coupling.tests.vcm_c4_reference import vcm_c4

PATM_HPA = 1013.15
XML = os.path.join(os.path.dirname(__file__), "..", "data", "maize_calibrated.xml")


def main():
    plant = grow_plant(os.path.normpath(XML), simulation_time=55, seed=42)
    res = run_photosynthesis_solve(plant, sim_time=1, par=1500.0, tleaf=30.0,
                                   label="vcm-cross-check", c4_model=1)

    eta_cpp = np.asarray(res["eta"], dtype=float)
    Ja_cpp = np.asarray(res["Ja"], dtype=float) * 1e6      # mol -> umol
    vin = res.get("_vcm_inputs")
    assert vin is not None, "VCM branch not exercised - is maize PhotoType C4?"
    n = len(eta_cpp)
    assert n > 0, "no leaf segments"

    ci = np.asarray(vin["ci"], dtype=float)
    An_cpp = np.asarray(vin["An"], dtype=float) * 1e6      # mol -> umol
    Vcref = np.asarray(vin["Vcrefmax"], dtype=float)
    Ql = np.asarray(vin["Qlight"], dtype=float)
    Tl = np.asarray(vin["TleafK"], dtype=float)

    def mors(d):  # replicate getMeanOrSegData
        return d if len(d) == n else np.full(n, d[0])

    Q = mors(Ql) * 1e6                                     # mol -> umol photons
    T = mors(Tl)
    Ci_bar = ci * PATM_HPA / 1000.0                        # mol/mol -> bar
    Vcmax25 = Vcref * 1e6                                  # mol -> umol
    p_Pa = PATM_HPA * 100.0
    O = 210e-3 * PATM_HPA / 1000.0                         # oi (mol/mol) -> bar

    orc = vcm_c4(Ci_bar, Q, T, Vcmax25, p_Pa, O=O)
    An_or = np.asarray(orc["A"], dtype=float)
    Ja_or = np.asarray(orc["Ja"], dtype=float)
    eta_or = np.asarray(orc["eta"], dtype=float)

    print(f"segments: {n}")
    print(f"An  C++ [{An_cpp.min():.2f},{An_cpp.max():.2f}]  oracle [{An_or.min():.2f},{An_or.max():.2f}]")
    print(f"Ja  C++ [{Ja_cpp.min():.1f},{Ja_cpp.max():.1f}]  oracle [{Ja_or.min():.1f},{Ja_or.max():.1f}]")
    print(f"eta C++ [{eta_cpp.min():.3f},{eta_cpp.max():.3f}]  oracle [{eta_or.min():.3f},{eta_or.max():.3f}]")

    assert np.all(np.isfinite(eta_cpp)), "C++ eta not finite"
    assert np.all((eta_cpp > 0) & (eta_cpp < 5)), "C++ eta out of physical range"

    def worst(a, b):
        d = np.abs(a - b)
        rel = d / (np.abs(b) + 1e-9)
        return d.max(), rel.max()

    for name, a, b in [("An", An_cpp, An_or), ("Ja", Ja_cpp, Ja_or), ("eta", eta_cpp, eta_or)]:
        dmax, rmax = worst(a, b)
        print(f"{name}: max|abs|={dmax:.3e}  max|rel|={rmax:.3e}")
        assert np.allclose(a, b, rtol=2e-3, atol=1e-4), \
            f"{name} C++ vs oracle mismatch: max abs {dmax:.3e}, max rel {rmax:.3e}"

    print("OK: C++ TWO_CELL_VCM matches the Python oracle (An, Ja, eta)")


if __name__ == "__main__":
    main()
