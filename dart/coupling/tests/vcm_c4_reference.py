"""Clean reference port of the von Caemmerer (2000) two-cell C4 model + Magnani-Difazio
(2012) fluorescence, used as the validation ORACLE for the CPlantBox C++ port.

Source: DART/.../BALENO/plugins/PhotosynthesisCM/vegetation_model_CM.py (C4 branch).
That file's enzyme-limited electron-transport (Ja) branch (lines 258-268) is dead MATLAB
paren-indexing (`A(ind)`, `gbs(ind)`...) that raises TypeError in Python and was never
exercised. Here it is corrected to numpy indexing/algebra so we have a runnable oracle.

Goal of the coupling fix: CPlantBox becomes the single authority for An, Tuzet stomata,
actual electron transport Ja (with bundle-sheath leakiness gbs -- no fixed 1/6 cost), and
fluorescence yield eta. This module computes A, Ja, eta from a *given* Ci (as CPlantBox's
C++ solve will), so it both (a) sanity-checks the algorithm on maize and (b) is the
bit-for-bit oracle the C++ TWO_CELL_VCM branch is validated against.

Units: Ci, Cm, Kc, Ko, Kp, O are partial pressures in bar; rates in umol m-2 s-1.
CPlantBox Ci[mol/mol] -> bar via Ci * p[Pa] * 1e-5.
"""
import numpy as np

# PSII rate constants (Magnani-Difazio 2012 / van der Tol 2014); the qLs/NPQs stress knobs
# are the per-pathway calibration handles (caveat #5: C3-origin, fit to C4 SIF data later).
KF = 3.0e7
KD = 1.0e8
KD_DARK = 1.95e8       # 'kd' in the reference (constitutive thermal dissipation)
PO0MAX = 0.88
BETA_C4 = 0.4          # PSII fraction of absorbed light reaching photochemistry (C4)
R_GAS = 8.314          # J mol-1 K-1


def _md12_fs(ps, Ja, Jms, kps, kf, kds, kDs):
    """Magnani-Difazio (2012) steady-state fluorescence (vegetation_model_CM.py:294-305)."""
    fs1 = ps * (kf / kps) / (1.0 - Ja / Jms)
    par1 = kps / (kps - kds)
    par2 = par1 * (kf + kDs + kds) / kf
    fs2 = (par1 - ps) / par2
    return np.minimum(fs1, fs2)


def vcm_c4(Ci, Q, T, Vcmax25, p,
           O=0.21, qLs=1.0, NPQs=0.0, Rdparam=0.025):
    """Two-cell C4 assimilation + fluorescence.

    Parameters (scalar or numpy array, broadcast):
      Ci       intercellular CO2 [bar]
      Q        absorbed PAR [umol photons m-2 s-1]
      T        leaf temperature [K]
      Vcmax25  max Rubisco carboxylation at 25C [umol m-2 s-1]
      p        air pressure [Pa]
      O        O2 partial pressure [bar]
      qLs      functional-PSII fraction (1.0 unstressed)   -- calibration knob
      NPQs     sustained non-photochemical quenching (0.0) -- calibration knob
      Rdparam  Rd / Vcmax25
    Returns dict with A (net, umol m-2 s-1), Ja (umol e- m-2 s-1), eta, and intermediates.
    """
    Ci = np.asarray(Ci, dtype=float)
    Q = np.where(np.asarray(Q, dtype=float) <= 0, 1e-9, np.asarray(Q, dtype=float))
    T = np.asarray(T, dtype=float)

    TREF = 25.0 + 273.15
    dum1 = R_GAS / 1000.0 * T
    dum2 = R_GAS / 1000.0 * TREF

    # --- reference-temperature parameters (C4) ---
    SCOOP = 2862.0
    Rdopt = Rdparam * Vcmax25
    Jmo = Vcmax25 * 40.0 / 6.0
    Vpmo = Vcmax25 * 2.33
    Vpr = 80.0
    gbs = (0.0207 * Vcmax25 + 0.4806) * 1000.0
    x = 0.4
    alpha = 0.0

    # --- temperature-correction constants (C4) ---
    HARD = 46.39;    CRD = 1000.0 * HARD / (R_GAS * TREF)
    HAGSTAR = 37.83; CGSTAR = 1000.0 * HAGSTAR / (R_GAS * TREF)
    HAJ = 77.9; HDJ = 191.9; DELTASJ = 0.627
    HAVCM = 67.29; HDVC = 144.57; DELTASVC = 0.472
    HAVPM = 70.37; HDVP = 117.93; DELTASVP = 0.376
    KCOP = 944.0; Q10KC = 2.1
    KOOP = 633.0; Q10KO = 1.2
    KPOP = 82.0;  Q10KP = 2.1

    # --- temperature corrections ---
    Rd = Rdopt * np.exp(CRD - HARD / dum1)
    SCO = SCOOP / np.exp(CGSTAR - HAGSTAR / dum1)
    Jmax = Jmo * np.exp(HAJ * (T - TREF) / (TREF * dum1))
    Jmax = Jmax * (1.0 + np.exp((TREF * DELTASJ - HDJ) / dum2))
    Jmax = Jmax / (1.0 + np.exp((T * DELTASJ - HDJ) / dum1))
    Vcmax = Vcmax25 * np.exp(HAVCM * (T - TREF) / (TREF * dum1))
    Vcmax = Vcmax * (1.0 + np.exp((TREF * DELTASVC - HDVC) / dum2))
    Vcmax = Vcmax / (1.0 + np.exp((T * DELTASVC - HDVC) / dum1))
    Vpmax = Vpmo * np.exp(HAVPM * (T - TREF) / (TREF * dum1))
    Vpmax = Vpmax * (1.0 + np.exp((TREF * DELTASVP - HDVP) / dum2))
    Vpmax = Vpmax / (1.0 + np.exp((T * DELTASVP - HDVP) / dum1))
    Kc = KCOP * Q10KC ** ((T - TREF) / 10.0) * 1e-11 * p
    Ko = KOOP * Q10KO ** ((T - TREF) / 10.0) * 1e-08 * p
    Kp = KPOP * Q10KP ** ((T - TREF) / 10.0) * 1e-11 * p

    # --- electron transport (with qLs/NPQs stress) ---
    kPSII = (KD + KF) * PO0MAX / (1.0 - PO0MAX)
    fo0 = KF / (KF + kPSII + KD)
    kps = kPSII * qLs
    kNPQs = NPQs * (KF + KD)
    kds = KD_DARK * qLs
    kDs = KD + kNPQs
    Jms = Jmax * qLs
    po0 = kps / (kps + KF + kDs)
    THETA = (kps - kds) / (kps + KF + kDs)
    Q2 = BETA_C4 * Q * po0
    J = (Q2 + Jms - np.sqrt((Q2 + Jms) ** 2 - 4 * THETA * Q2 * Jms)) / (2 * THETA)

    # --- two-cell C4 assimilation (von Caemmerer 2000) ---
    Cm = Ci
    Rm = 0.5 * Rd
    gam = 0.5 / SCO
    Vpc = Vpmax * Cm / (Cm + Kp)
    Vp = np.minimum(Vpc, Vpr)

    d1 = alpha / 0.047
    d2 = Kc / Ko
    dum3 = Vp - Rm + gbs * Cm
    dum4 = Vcmax - Rd
    dum5 = gbs * Kc * (1.0 + O / Ko)
    dum6 = gam * Vcmax
    dum7 = x * J / 2.0 - Rm + gbs * Cm
    dum8 = (1.0 - x) * J / 3.0
    dum9 = dum8 - Rd
    dum10 = dum8 + Rd * 7.0 / 3.0

    # CO2/enzyme-limited Ac (vegetation_model_CM.py:247-250)
    a_c = 1.0 - d1 * d2
    b_c = -(dum3 + dum4 + dum5 + d1 * (dum6 + Rd * d2))
    c_c = dum4 * dum3 - dum6 * gbs * O + Rd * dum5
    Ac = (-b_c - np.sqrt(b_c ** 2 - 4.0 * a_c * c_c)) / (2.0 * a_c)

    # light/electron-limited Aj (vegetation_model_CM.py:251-255)
    a_j = 1.0 - 7.0 / 3.0 * gam * d1
    b_j = -(dum7 + dum9 + gbs * gam * O * 7.0 / 3.0 + d1 * gam * dum10)
    c_j = dum7 * dum9 - gbs * gam * O * dum10
    Aj = (-b_j - np.sqrt(b_j ** 2 - 4.0 * a_j * c_j)) / (2.0 * a_j)

    A = np.minimum(Ac, Aj)

    # actual electron transport Ja: J when light-limited; else invert the enzyme-limited
    # quadratic (von Caemmerer 2000; corrected from the reference's dead A(ind) branch).
    Asafe = np.where(np.abs(A) < 1e-9, 1e-9, A)
    a_e = x * (1.0 - x) / 6.0 / Asafe
    b_e = (1.0 - x) / 3.0 * (gbs / Asafe * (Cm - Rm / gbs - gam * O) - 1.0 - d1) \
        - x / 2.0 * (1.0 + Rd / Asafe)
    c_e = (1.0 + Rd / Asafe) * (Rm - gbs * Cm - 7.0 * gbs * gam * O / 3.0) \
        + (Rd + A) * (1.0 - 7.0 * alpha * gam / 3.0 / 0.047)
    disc = np.maximum(b_e ** 2 - 4.0 * a_e * c_e, 0.0)
    Ja_enz = (-b_e + np.sqrt(disc)) / (2.0 * a_e)
    Ja = np.where(Ac <= Aj, Ja_enz, J)
    Ja = np.clip(Ja, 0.0, J)

    # --- PSII yield + fluorescence (vegetation_model_CM.py:272-276) ---
    ps = Ja / (BETA_C4 * Q)
    fs = _md12_fs(ps, Ja, Jms, kps, KF, kds, kDs)
    eta = fs / fo0

    return dict(A=A, Ac=Ac, Aj=Aj, Ja=Ja, J=J, eta=eta,
                Vcmax=Vcmax, Vpmax=Vpmax, Jmax=Jmax, Rd=Rd, Vp=Vp)


def _selfcheck():
    """Maize C4 sanity: saturating light + ambient CO2 -> An ~ 30-45 umol, eta bounded."""
    p = 1.013e5
    Vcmax25 = 40.0                  # typical field maize
    Ca = 400e-6 * p * 1e-5          # 400 ppm -> bar
    Ci = 0.4 * Ca                   # C4 Ci/Ca ~ 0.4
    O = 0.21

    sat = vcm_c4(Ci, Q=1800.0, T=303.15, Vcmax25=Vcmax25, p=p, O=O)
    An = float(sat["A"])
    eta = float(sat["eta"])
    print(f"saturating: An={An:.1f} umol/m2/s  Ja={float(sat['Ja']):.0f}  "
          f"J={float(sat['J']):.0f}  eta={eta:.3f}  (Ac={float(sat['Ac']):.1f}, Aj={float(sat['Aj']):.1f})")
    assert 20.0 < An < 50.0, f"maize saturating An out of range: {An}"
    assert 0.0 < eta < 5.0, f"eta out of range: {eta}"

    # light response: An and eta must respond to PAR, An monotone up to saturation
    Qs = np.array([0.0, 100.0, 400.0, 900.0, 1800.0])
    out = vcm_c4(Ci, Q=Qs, T=303.15, Vcmax25=Vcmax25, p=p, O=O)
    Avec = np.asarray(out["A"])
    print("light response An:", np.round(Avec, 1))
    assert np.all(np.diff(Avec) >= -1e-6), f"An not monotone in PAR: {Avec}"
    assert Avec[-1] > Avec[1] + 5.0, "An barely responds to light"
    assert np.all(np.isfinite(out["eta"])), "eta not finite across light"

    # dark: net assimilation negative (respiration dominates)
    dark = vcm_c4(Ci, Q=0.0, T=303.15, Vcmax25=Vcmax25, p=p, O=O)
    print(f"dark: An={float(dark['A']):.2f}")
    assert float(dark["A"]) < 0.0, "dark An should be negative"

    print("OK: vcm_c4 reference passes maize sanity checks")


if __name__ == "__main__":
    _selfcheck()
