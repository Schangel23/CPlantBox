"""Measure physical contacts between GT leaf instances in a synthetic cache."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from viz_cylindrical_unwrap import fit_axis, cylindrical


def _angdist(a, b):
    return abs(float(np.angle(np.exp(1j * (a - b)))))


def _circmean(phi):
    return float(np.angle(np.mean(np.exp(1j * phi))))


def _leaf_azimuths(phi, r, gt):
    out = {}
    for g in np.unique(gt):
        if g == 0:
            continue
        m = gt == g
        rg = r[m]
        sel = m.copy()
        sel[m] = rg >= np.quantile(rg, 0.5)
        out[int(g)] = _circmean(phi[sel])
    return out


def _collar_heights(h, r, gt):
    out = {}
    for g in np.unique(gt):
        if g == 0:
            continue
        m = gt == g
        rg = r[m]
        sel = m.copy()
        sel[m] = rg <= np.quantile(rg, 0.15)
        out[int(g)] = float(np.median(h[sel]))
    return out


def _touching_pairs(pts, gt, threshold):
    ids = [int(i) for i in np.unique(gt) if i != 0]
    trees = {i: cKDTree(pts[gt == i]) for i in ids}
    pairs = []
    total_pairs = 0
    for a_i, ga in enumerate(ids):
        pa = pts[gt == ga]
        for gb in ids[a_i + 1:]:
            total_pairs += 1
            d, _ = trees[gb].query(pa, k=1, distance_upper_bound=threshold)
            if np.any(np.isfinite(d)):
                pairs.append((ga, gb))
    return pairs, total_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.cache, "cloud_*.npz")))
    if not files:
        raise SystemExit(f"no cloud_*.npz files in {a.cache}")

    total_touch = 0
    total_pairs = 0
    adjacent_touch = 0
    touch_per_plant = []
    heights = []
    collar_spacings = []

    print(f"=== leaf contacts | cache {a.cache} | {len(files)} plants | threshold={a.threshold:g} cm ===")
    for i, f in enumerate(files):
        d = np.load(f)
        pts, gt = d["pts"], d["gt"]
        c, axis, _ = fit_axis(pts, gt == 0)
        h, r, phi, _, _ = cylindrical(pts, c, axis)
        az = _leaf_azimuths(phi, r, gt)
        collars = _collar_heights(h, r, gt)
        pairs, n_pairs = _touching_pairs(pts, gt, a.threshold)
        adj = sum(1 for ga, gb in pairs if _angdist(az[ga], az[gb]) < np.radians(60.0))
        total_touch += len(pairs)
        total_pairs += n_pairs
        adjacent_touch += adj
        touch_per_plant.append(len(pairs))
        heights.append(float(pts[:, 2].max() - pts[:, 2].min()))
        ch = np.array([collars[k] for k in sorted(collars)], dtype=float)
        if len(ch) > 1:
            collar_spacings.append(float(np.median(np.diff(np.sort(ch)))))
        print(
            f"[{i+1:2d}/{len(files)}] seed{int(d['seed']):3d} d{int(d['day']):3d} "
            f"L={len(collars):2d} touching={len(pairs):3d}/{n_pairs:3d} adjacent={adj:3d} "
            f"height={heights[-1]:.2f}cm collar_med={collar_spacings[-1] if collar_spacings else float('nan'):.2f}cm"
        )

    frac_touch = total_touch / max(total_pairs, 1)
    frac_adj = adjacent_touch / max(total_touch, 1)
    print("\n--- CONTACT SUMMARY ---")
    print(f"mean touching pairs per plant {np.mean(touch_per_plant):.3f}")
    print(f"total touching pairs across cache {total_touch}/{total_pairs} ({100*frac_touch:.1f}%)")
    print(f"adjacent-azimuth touching fraction {adjacent_touch}/{total_touch} ({100*frac_adj:.1f}%)")
    print(f"plant height mean cm {np.mean(heights):.3f}")
    print(f"median collar spacing cm {np.median(collar_spacings):.3f}")


if __name__ == "__main__":
    main()
