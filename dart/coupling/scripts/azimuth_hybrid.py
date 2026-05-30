"""Unsupervised azimuth split refinement for maize leaf point labels.

This module deliberately uses only geometry and the baseline prediction.  It
keeps the baseline leaf/non-leaf decision fixed, then splits individual
baseline leaf clusters when their distal/high-radius points show separated
cylindrical azimuth modes around the fitted pseudostem axis.
"""
from __future__ import annotations

import numpy as np


def fit_axis(pts, nonleaf):
    """PCA axis of the non-leaf core, with bottom-40% fallback."""
    pts = np.asarray(pts, dtype=np.float64)
    nl = pts[np.asarray(nonleaf, dtype=bool)]
    if len(nl) < 10:
        zhi = np.quantile(pts[:, 2], 0.4)
        nl = pts[pts[:, 2] <= zhi]
    if len(nl) < 10:
        nl = pts

    zhi = np.quantile(nl[:, 2], 0.6)
    core = nl[nl[:, 2] <= zhi]
    if len(core) < 10:
        core = nl

    c = core.mean(0)
    _, s, vt = np.linalg.svd(core - c, full_matrices=False)
    a = vt[0]
    if a[2] < 0:
        a = -a

    z = np.array([0.0, 0.0, 1.0])
    tilt = np.degrees(np.arccos(np.clip(abs(a[2]), 0, 1)))
    gap = s[0] / max(s[1], 1e-12) if len(s) > 1 else np.inf
    if tilt < 3.0 or gap < 1.15:
        w = np.clip(tilt / 3.0, 0.0, 1.0) if tilt < 3.0 else 0.5
        a = w * a + (1.0 - w) * z
        a /= max(np.linalg.norm(a), 1e-12)

    return c, a


def cylindrical(pts, c, a):
    d = pts - c
    h = d @ a
    w = d - np.outer(h, a)
    r = np.linalg.norm(w, axis=1)
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = tmp - (tmp @ a) * a
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    phi = np.arctan2(w @ v, w @ u)
    return h, r, phi, u, v


def angdist(p, q):
    """Circular distance in radians."""
    return np.abs(np.angle(np.exp(1j * (p - q))))


def _circular_peaks(phi, weights, min_sep, nbins=72):
    if len(phi) < 12 or np.sum(weights) <= 0:
        return []

    hist, edges = np.histogram(phi, bins=nbins, range=(-np.pi, np.pi),
                               weights=weights)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    smooth = sum(kernel[i] * np.roll(hist, i - 2) for i in range(len(kernel)))
    smooth /= kernel.sum()
    if smooth.max() <= 0:
        return []

    left = np.roll(smooth, 1)
    right = np.roll(smooth, -1)
    floor = max(0.18 * float(smooth.max()), float(np.median(smooth)))
    cand = np.flatnonzero((smooth >= left) & (smooth >= right) & (smooth >= floor))
    if len(cand) == 0:
        return []

    centers = 0.5 * (edges[:-1] + edges[1:])
    order = cand[np.argsort(smooth[cand])[::-1]]
    modes = []
    for b in order:
        p = centers[b]
        if all(angdist(p, q) >= min_sep for q in modes):
            modes.append(float(p))

    if len(modes) <= 1:
        return modes
    modes.sort()
    return modes


def _gap_modes(phi, weights, min_sep, min_group_weight_frac=0.08):
    """Modes from separated occupied arcs after cutting the largest wrap gap."""
    if len(phi) < 12 or np.sum(weights) <= 0:
        return []

    order = np.argsort(phi)
    ps = phi[order]
    ws = weights[order]
    gaps = np.diff(np.r_[ps, ps[0] + 2.0 * np.pi])
    cut = int(np.argmax(gaps))
    unwrapped = np.r_[ps[cut + 1:], ps[:cut + 1] + 2.0 * np.pi]
    w_unwrapped = np.r_[ws[cut + 1:], ws[:cut + 1]]
    internal = np.flatnonzero(np.diff(unwrapped) >= min_sep)
    if len(internal) == 0:
        return []

    starts = np.r_[0, internal + 1]
    stops = np.r_[internal + 1, len(unwrapped)]
    min_w = min_group_weight_frac * float(np.sum(w_unwrapped))
    modes = []
    for lo, hi in zip(starts, stops):
        if np.sum(w_unwrapped[lo:hi]) < min_w:
            continue
        z = np.sum(w_unwrapped[lo:hi] * np.exp(1j * unwrapped[lo:hi]))
        if abs(z) > 0:
            modes.append(float(np.angle(z)))
    return modes


def _reindex(pred):
    out = np.zeros(len(pred), dtype=np.int64)
    next_id = 1
    for lab in sorted(int(x) for x in np.unique(pred) if x != 0):
        out[pred == lab] = next_id
        next_id += 1
    return out


def azimuth_refine(pts, base_pred, tau_deg=50, min_phi_sep_deg=40,
                   high_r_frac=0.5):
    """Split baseline leaf clusters by unsupervised distal azimuth modes.

    Parameters
    ----------
    pts : (N, 3) array
        Point cloud coordinates.
    base_pred : (N,) array
        Baseline per-point labels, with 0 reserved for non-leaf.
    tau_deg, min_phi_sep_deg, high_r_frac : float
        Split trigger separation, peak de-duplication separation, and global
        high-radius quantile used for azimuth estimation.
    """
    pts = np.asarray(pts, dtype=np.float64)
    base_pred = np.asarray(base_pred, dtype=np.int64)
    if len(pts) == 0 or len(base_pred) != len(pts):
        return np.asarray(base_pred, dtype=np.int64).copy()

    leaf_mask = base_pred != 0
    if np.count_nonzero(leaf_mask) == 0:
        return np.zeros(len(pts), dtype=np.int64)

    c, a = fit_axis(pts, base_pred == 0)
    _, r, phi, _, _ = cylindrical(pts, c, a)

    high_r_frac = float(np.clip(high_r_frac, 0.0, 1.0))
    r_thr = np.quantile(r[leaf_mask], high_r_frac)
    tau = np.radians(tau_deg)
    min_sep = np.radians(min_phi_sep_deg)

    refined = np.zeros(len(pts), dtype=np.int64)
    next_id = 1
    for lab in sorted(int(x) for x in np.unique(base_pred) if x != 0):
        idx = np.flatnonzero(base_pred == lab)
        if len(idx) == 0:
            continue

        hi_idx = idx[r[idx] >= r_thr]
        if len(hi_idx) < max(12, min(40, int(0.05 * len(idx)))):
            refined[idx] = next_id
            next_id += 1
            continue

        weights = np.maximum(r[hi_idx], 1e-6)
        modes = _gap_modes(phi[hi_idx], weights, min_sep)
        if len(modes) < 2:
            modes = _circular_peaks(phi[hi_idx], weights, min_sep)
        if len(modes) < 2:
            refined[idx] = next_id
            next_id += 1
            continue

        modes_arr = np.array(modes, dtype=np.float64)
        d_modes = angdist(modes_arr[:, None], modes_arr[None, :])
        if float(d_modes.max()) < tau:
            refined[idx] = next_id
            next_id += 1
            continue

        assign = np.argmin(angdist(phi[idx][:, None], modes_arr[None, :]), axis=1)
        counts = np.bincount(assign, minlength=len(modes_arr))
        keep = counts >= max(20, int(0.03 * len(idx)))
        if np.count_nonzero(keep) < 2:
            refined[idx] = next_id
            next_id += 1
            continue

        kept_modes = modes_arr[keep]
        assign = np.argmin(angdist(phi[idx][:, None], kept_modes[None, :]), axis=1)
        for k in range(len(kept_modes)):
            sub = idx[assign == k]
            if len(sub):
                refined[sub] = next_id
                next_id += 1

    return _reindex(refined)
