"""Shared kappa(s) computation for CurvatureProfileTropism calibration + validation.

Curvature profile kappa(s) of a 3D midrib polyline, computed exactly the way the
success metric is defined: smoothing cubic spline by arc length, then
kappa = |r' x r''| / |r'|^3 (physical 1/cm), sampled at normalized-arc knots.

Used by both fit_curvature_profile_to_recon.py (writes the XML) and the
validation (measures the grown getNodes midrib), so the two are guaranteed to
use the identical definition.
"""
import numpy as np
from scipy.interpolate import splprep, splev


def kappa_profile(pts, n_knots=12, sf=1e-5, n_resample=40):
    """kappa(s) of a midrib polyline.

    pts : (M,3) ordered midrib points (base->tip), physical units (cm).
    Returns (phi, kappa): phi = normalized arc fraction knots in [0,1],
    kappa = curvature magnitude [1/cm] at each knot (piecewise-linear ready).

    The polyline is first resampled to n_resample points by arc length so the
    smoothing s = sf*L^2 acts identically regardless of input density (RECON's
    ~18 CPs vs a grown skeleton's hundreds of nodes), making the fit-side and
    validation-side kappa(s) directly comparable.
    """
    pts = np.asarray(pts, float)
    # drop consecutive duplicates (splprep chokes on zero-length segments)
    d = np.r_[1.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)]
    pts = pts[d > 1e-9]
    if len(pts) < 4:
        return np.linspace(0, 1, n_knots), np.zeros(n_knots)
    # resample to a fixed point count by arc length (density-independent smoothing)
    seg0 = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc0 = np.concatenate([[0.0], np.cumsum(seg0)])
    u0 = arc0 / arc0[-1]
    ur = np.linspace(0.0, 1.0, n_resample)
    pts = np.column_stack([np.interp(ur, u0, pts[:, k]) for k in range(3)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum())
    s = sf * L * L
    tck, _ = splprep(pts.T, s=s, k=3)
    uu = np.linspace(0.0, 1.0, 200)
    r1 = np.array(splev(uu, tck, der=1))   # (3, N) d r / d u
    r2 = np.array(splev(uu, tck, der=2))   # (3, N)
    cross = np.cross(r1.T, r2.T)           # (N,3)
    num = np.linalg.norm(cross, axis=1)
    den = np.linalg.norm(r1.T, axis=1) ** 3 + 1e-12
    kap_u = num / den                      # kappa as function of spline param u
    # convert spline-param u to physical-arc fraction: integrate |r'(u)| du
    speed = np.linalg.norm(r1.T, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(uu))])
    frac = arc / (arc[-1] + 1e-12)
    phi = np.linspace(0.0, 1.0, n_knots)
    kappa = np.interp(phi, frac, kap_u)
    return phi, kappa


if __name__ == "__main__":
    # self-check: a circular arc of radius R has constant kappa = 1/R
    R = 5.0
    th = np.linspace(0, 1.2, 40)
    arc = np.c_[R * np.cos(th), R * np.sin(th), np.zeros_like(th)]
    phi, kap = kappa_profile(arc, n_knots=10)
    # ~11% spline-edge inflation at the endpoints on a perfect circle; interior is tight
    assert np.allclose(kap, 1.0 / R, atol=0.03), kap
    print("kappa_profile self-check ok: const kappa", kap.mean(), "expected", 1.0 / R)
