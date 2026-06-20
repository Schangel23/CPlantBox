#!/usr/bin/env python3
"""Refit per-rank ParametricLeafShape intercepts to plant_01 RECON midribs.

The native PrescribedMidribTropism (tropismT=8) steers each leaf along the
EFFECTIVE surface_cps midrib, which at maturity is the ParametricLeafShape
sampleCanonicalGrid(lmax, Width_blade) = evaluate_symmetric(intercept) + asym
residual. The deployed intercepts were fit to donor/median shapes and then
rank-shifted (+3) + stub-copied, so per-rank they no longer match plant_01
RECON. This rewrites intercepts[r] (and its asym residual + per-rank reference
scales) from RECON cps_{r}[last_reliable] for every RECON-reliable rank, giving
PARAM rank r == RECON rank r in SHAPE (midrib arch). The SIZE knobs that scale
at runtime are the XML leaf lmax / Width_blade — untouched here — so plant area
is preserved (grown-size + RECON-shape, exactly the 1:1-proof combination).

Intercept coeffs are dimensionless (droop/along ÷ lmax, width ÷ max_w), so
midrib turning is scale-invariant: the RECON arch survives the XML size scaling.

Run from CPlantBox root with cpbenv. Idempotent; writes the JSON in place.
Self-check (assert) compares refit-intercept turning to raw RECON turning.
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home/lukas/PHD/CPlantBox")
sys.path.insert(0, str(ROOT))
JSON = ROOT / "dart/coupling/data/plant01_leaf_shape_dist.json"
NPZ = Path("/home/lukas/pointr/fp4d_plants_basecarry/Plot03/plant_01/plant01_stage_library.npz")

# load the fitter as a module (needs registration in sys.modules for dataclasses)
_spec = importlib.util.spec_from_file_location(
    "fitmod", str(ROOT / "dart/coupling/scripts/fit_parametric_leaf_shape.py"))
fit = importlib.util.module_from_spec(_spec)
sys.modules["fitmod"] = fit
_spec.loader.exec_module(fit)


def regrid(grid, nu, nv):
    """Bilinear regrid an (Nu,Nv,3) CP grid to (nu,nv,3) via per-axis linear interp."""
    su = np.linspace(0, 1, grid.shape[0]); tu = np.linspace(0, 1, nu)
    g = np.stack([np.array([np.interp(tu, su, grid[:, j, k]) for k in range(3)]).T
                  for j in range(grid.shape[1])], axis=1)
    sv = np.linspace(0, 1, g.shape[1]); tv = np.linspace(0, 1, nv)
    g = np.stack([np.array([np.interp(tv, sv, g[i, :, k]) for k in range(3)]).T
                  for i in range(g.shape[0])], axis=0)
    return g


def midrib_turning(mid, n=20):
    mid = np.asarray(mid, float)
    if len(mid) < 3:
        return np.nan
    seg = np.linalg.norm(np.diff(mid, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    if arc[-1] < 1e-9:
        return np.nan
    s = np.linspace(0, arc[-1], n + 1)
    m = np.column_stack([np.interp(s, arc, mid[:, k]) for k in range(3)])
    sg = np.diff(m, axis=0)
    sg /= np.linalg.norm(sg, axis=1, keepdims=True) + 1e-12
    return float(sum(np.degrees(np.arccos(np.clip(np.dot(sg[i], sg[i - 1]), -1, 1)))
                     for i in range(1, len(sg))))


def main():
    J = json.loads(JSON.read_text())
    nu, nv = J["n_u"], J["n_v"]
    d = np.load(NPZ, allow_pickle=True)
    refit, worst = [], 0.0
    for r in range(J["n_ranks"]):
        cps, rel = d.get(f"cps_{r}"), d.get(f"reliable_{r}")
        if cps is None or rel is None or not np.asarray(rel).any():
            continue  # no RECON for this rank -> keep deployed (stub) intercept
        grid = regrid(np.asarray(cps[np.where(rel)[0][-1]], float), nu, nv)
        sym = fit.extract_symmetric(grid)
        coeffs = fit.fit_intercept(sym)
        asym = fit.compute_asym_residual(grid, coeffs, sym.max_w, sym.lmax_self)
        J["intercepts"][str(r)] = np.asarray(coeffs).tolist()
        J["asym_residual_grids_cm"][str(r)] = np.asarray(asym).tolist()
        J["lmax_intercept_cm"][str(r)] = float(sym.lmax_self)
        J["max_w_xml_cm"][str(r)] = float(sym.max_w)
        # self-check: the refit shape's midrib turning vs raw RECON
        rec = fit.evaluate_symmetric(coeffs, sym.max_w, sym.lmax_self)
        t_fit = midrib_turning(rec[:, rec.shape[1] // 2, :])
        t_raw = midrib_turning(grid[:, nv // 2, :])
        worst = max(worst, abs(t_fit - t_raw))
        refit.append(r)
        print(f"  rank {r:>2}: turning RECON {t_raw:6.1f}  -> intercept {t_fit:6.1f}  "
              f"(d {t_fit - t_raw:+5.1f})  lmax {sym.lmax_self:5.1f} max_w {sym.max_w:5.1f}")
    J.setdefault("shift_history", []).append({
        "op": "refit_intercepts_to_recon",
        "ranks": refit,
        "note": "intercepts overwritten 1:1 from plant01 RECON midribs (shape only; "
                "size via XML lmax/Width_blade unchanged). Undoes the +3 shift for "
                "these ranks so PARAM rank r == RECON rank r.",
    })
    JSON.write_text(json.dumps(J, indent=2))
    print(f"refit {len(refit)} ranks: {refit}")
    print(f"worst intercept-vs-RECON turning gap: {worst:.1f} deg (spline-basis fidelity)")
    # the parametric basis smooths sharp curvature; demand it stays in a sane band
    assert worst < 45.0, f"refit turning gap {worst:.1f} too large — frame/regrid bug?"


if __name__ == "__main__":
    main()
