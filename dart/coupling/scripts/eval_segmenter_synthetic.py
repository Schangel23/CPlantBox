"""Quantitative segmenter judge on LABELLED synthetic maize -- no hand labels.

The synthetic pipeline grows each plant from known organs, so every surface
point carries its true leaf id for free. We sample a labelled complete cloud,
apply the SAME FP4D-style occlusion the completion generator uses (carrying
labels through every mask), then run the MonGraphSeg pseudostem segmenter and
score it against ground truth:

  * leaf-count error      (predicted vs true leaf count)
  * mean per-leaf IoU     (Hungarian-matched predicted<->true leaf clusters)
  * point accuracy        (fraction of leaf points given the right instance)

Optionally runs a pre-segmentation fill first (``--fill directional``) so the
SAME metric tells us whether a fill actually helps segmentation, instead of
eyeballing PNGs. Run from CPlantBox root with cpbenv:

    cpbenv/bin/python dart/coupling/scripts/eval_segmenter_synthetic.py \
        --n 8 --fill none
    cpbenv/bin/python dart/coupling/scripts/eval_segmenter_synthetic.py \
        --n 8 --fill directional
"""
from __future__ import annotations
import sys
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "/home/lukas/pointr")          # directional_fill (pure numpy)

from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs
try:
    from gen_synth_completion_pairs import (occlude_depthbuffer, scanner_viewpoint,
                                            ground_up_dropout, sector_loss,
                                            within_leaf_holes)
except ModuleNotFoundError:
    from dart.coupling.scripts.gen_synth_completion_pairs import (
        occlude_depthbuffer, scanner_viewpoint, ground_up_dropout, sector_loss,
        within_leaf_holes)
import mongraphseg_graph as mg

DEFAULT_XML = "dart/coupling/data/maize_calibrated.xml"


def sample_labelled(mesh, n, rng):
    """Area-weighted surface sampling that also returns each point's leaf label.

    Label 0 = stem/sheath/tassel (non-leaf); 1..L = leaf instances by organ_id.
    """
    v = mesh.vertices[mesh.indices]                       # (T,3,3)
    a = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) * 0.5
    p = a / a.sum()
    tri = rng.choice(len(mesh.indices), n, p=p)
    u = rng.random((n, 1)); w = rng.random((n, 1))
    over = (u + w > 1).ravel(); u[over] = 1 - u[over]; w[over] = 1 - w[over]
    A, B, C = v[tri, 0], v[tri, 1], v[tri, 2]
    pts = A + u * (B - A) + w * (C - A)
    type_by_id = {m["organ_id"]: m["type"] for m in mesh.organ_meta}
    tri_oid = mesh.organ_ids[tri]
    leaf_ids = sorted({oid for oid in np.unique(tri_oid)
                       if str(type_by_id.get(oid, "")).startswith("leaf")})
    remap = {oid: i + 1 for i, oid in enumerate(leaf_ids)}
    lab = np.array([remap.get(oid, 0) for oid in tri_oid], dtype=np.int64)
    return pts.astype(np.float64), lab


def occlude_labelled(pts, lab, rng):
    """FP4D-style self-occlusion (depth buffer + ground-up + sector) keeping
    labels aligned. Mirrors gen_synth_completion_pairs.make_partial masks."""
    idx = occlude_depthbuffer(pts, scanner_viewpoint(pts, rng),
                              ang_res_deg=rng.uniform(0.4, 0.9),
                              depth_tol_cm=rng.uniform(1.0, 2.5))
    pts, lab = pts[idx], lab[idx]
    keep = ground_up_dropout(pts, rng, strength=rng.uniform(0.3, 0.8))
    pts, lab = pts[keep], lab[keep]
    if rng.random() < 0.4 and len(pts) > 500:
        keep = sector_loss(pts, rng, frac=rng.uniform(0.15, 0.35))
        pts, lab = pts[keep], lab[keep]
    kept = within_leaf_holes(
        pts, rng,
        n_holes_per_kpt=rng.uniform(8.0, 14.0),
        r_lo_cm=rng.uniform(0.5, 0.8),
        r_hi_cm=rng.uniform(1.5, 2.5))
    pts, lab = pts[kept], lab[kept]
    # size-invariant voxel, label = first point in each voxel
    vox = float(np.clip(0.012 * np.ptp(pts[:, 2]), 0.1, 0.6))
    mn = pts.min(0); ijk = np.floor((pts - mn) / vox).astype(np.int64)
    _, u = np.unique(ijk, axis=0, return_index=True)
    pts, lab = pts[u], lab[u]
    pts = pts + rng.normal(0, 0.1, pts.shape)              # 1 mm
    return pts, lab


def predicted_labels(organs, pts):
    """Map segmenter organ dict -> per-point predicted leaf id (0 = non-leaf)."""
    pred = np.zeros(len(pts), dtype=np.int64)
    tree = cKDTree(pts)
    lid = 0
    for k in sorted(organs):
        if not k.startswith("leaf"):
            continue
        v = organs[k]
        if len(v) == 0:
            continue
        lid += 1
        _, idx = tree.query(v)
        pred[idx] = lid
    return pred


def score(gt, pred):
    """Leaf-count error, Hungarian mean IoU, leaf-point accuracy."""
    gt_ids = [i for i in np.unique(gt) if i != 0]
    pr_ids = [i for i in np.unique(pred) if i != 0]
    if not gt_ids:
        return None
    iou = np.zeros((len(gt_ids), len(pr_ids)))
    gmask = {g: gt == g for g in gt_ids}
    pmask = {p: pred == p for p in pr_ids}
    for a, g in enumerate(gt_ids):
        for b, p in enumerate(pr_ids):
            inter = np.count_nonzero(gmask[g] & pmask[p])
            if inter == 0:
                continue
            union = np.count_nonzero(gmask[g] | pmask[p])
            iou[a, b] = inter / union
    matched = np.zeros(0)
    r = c = np.zeros(0, dtype=int)
    if len(pr_ids):
        r, c = linear_sum_assignment(-iou)
        matched = iou[r, c]
    leafpts = gt != 0
    # point accuracy: among GT leaf points, fraction whose pred matches the
    # Hungarian-assigned instance for their GT leaf
    acc = np.nan
    if len(pr_ids):
        assign = {gt_ids[a]: pr_ids[b] for a, b in zip(r, c) if iou[a, b] > 0}
        ok = np.zeros(len(gt), bool)
        for g, p in assign.items():
            ok |= (gt == g) & (pred == p)
        acc = ok[leafpts].mean()
    return {"n_gt": len(gt_ids), "n_pred": len(pr_ids),
            "count_err": len(pr_ids) - len(gt_ids),
            "mean_iou": float(np.mean(matched)) if len(pr_ids) else 0.0,
            "iou_ge50": int(np.count_nonzero(matched >= 0.5)),
            "pt_acc": float(acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--day_lo", type=int, default=40)
    ap.add_argument("--day_hi", type=int, default=92)
    ap.add_argument("--n_sample", type=int, default=16384)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--fill", choices=["none", "directional", "combined", "maize"], default="none")
    ap.add_argument("--nsk", type=int, default=400, help="skeleton nodes")
    ap.add_argument("--min_leaf", type=float, default=3.0, help="min leaf/branch len cm")
    ap.add_argument("--prune", choices=["length", "support"], default="length")
    ap.add_argument("--support_frac", type=float, default=0.02)
    ap.add_argument("--assign", choices=["segment", "geodesic"], default="segment")
    ap.add_argument("--geodesic_k", type=int, default=10)
    ap.add_argument("--geodesic_max_edge_cm", type=float, default=3.0)
    ap.add_argument("--distal_seed_start", type=float, default=0.5)
    ap.add_argument("--distal_seed_end", type=float, default=1.0)
    ap.add_argument("--pseudostem_basal_only", action="store_true",
                    help="confine pseudostem to base->lowest collar (anti-theft)")
    ap.add_argument("--pseudostem_top_margin_cm", type=float, default=2.0)
    a = ap.parse_args()

    rows = []
    for i in range(a.n):
        seed = a.seed0 + i
        # per-plant RNG: plant + occlusion identical across --fill modes (clean A/B)
        rng = np.random.default_rng(seed)
        frng = np.random.default_rng(seed + 7)          # independent fill stream
        day = int(rng.integers(a.day_lo, a.day_hi + 1))
        plant = grow_plant(a.xml, simulation_time=day, seed=seed)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, comp_lab = sample_labelled(mesh, a.n_sample, rng)
        pts, gt = occlude_labelled(comp, comp_lab, rng)
        if a.fill == "directional":
            import directional_fill as df
            sp = df.local_spacing(pts)
            lab_f, fids = df.fragments(pts, 2.5 * sp, 8)
            eps = df.endpoints(pts, lab_f, fids, 4.0 * sp, 2.0)
            pairs = df.stitch(eps, 8.0, 30.0)
            seeds = [df.seed_tube(p0, p1, sp, 0.5 * sp, frng) for p0, p1, _ in pairs]
            if seeds:
                add = np.vstack(seeds)
                # seeds inherit the GT label of their nearest real point (for scoring)
                _, ni = cKDTree(pts).query(add)
                pts = np.vstack([pts, add]); gt = np.concatenate([gt, gt[ni]])
        if a.fill == "combined":
            import combined_fill as cf
            orig_pts = pts.copy()
            add = cf.combined_fill(pts, frng, return_added_only=True)
            if len(add):
                # added points inherit the GT label of their nearest ORIGINAL point
                _, ni = cKDTree(orig_pts).query(add)
                pts = np.vstack([pts, add]); gt = np.concatenate([gt, gt[ni]])
        if a.fill == "maize":
            import maize_leaf_fill as mlf
            orig_pts = pts.copy()
            add = mlf.maize_leaf_fill(pts, frng, return_added_only=True)
            if len(add):
                # within-footprint fill: nearest original is the same leaf -> clean label
                _, ni = cKDTree(orig_pts).query(add)
                pts = np.vstack([pts, add]); gt = np.concatenate([gt, gt[ni]])
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=a.nsk,
                                              min_leaf_len_cm=a.min_leaf,
                                              prune=a.prune,
                                              support_frac=a.support_frac,
                                              assign=a.assign,
                                              geodesic_k=a.geodesic_k,
                                              geodesic_max_edge_cm=a.geodesic_max_edge_cm,
                                              distal_seed_start=a.distal_seed_start,
                                              distal_seed_end=a.distal_seed_end,
                                              pseudostem_basal_only=a.pseudostem_basal_only,
                                              pseudostem_top_margin_cm=a.pseudostem_top_margin_cm)
        s = score(gt, predicted_labels(organs, pts))
        if s is None:
            continue
        s.update(seed=seed, day=day, npts=len(pts))
        rows.append(s)
        print(f"[{i+1}/{a.n}] seed {seed:3d} d{day:3d} pts {len(pts):5d} | "
              f"GT {s['n_gt']:2d} pred {s['n_pred']:2d} (err {s['count_err']:+d}) | "
              f"meanIoU {s['mean_iou']:.2f} IoU>=.5 {s['iou_ge50']:2d} | ptAcc {s['pt_acc']:.2f}")

    if rows:
        ce = np.array([r["count_err"] for r in rows])
        mi = np.array([r["mean_iou"] for r in rows])
        pa = np.array([r["pt_acc"] for r in rows])
        g5 = np.array([r["iou_ge50"] for r in rows])
        ng = np.array([r["n_gt"] for r in rows])
        print(f"\n=== fill={a.fill} assign={a.assign}  N={len(rows)} ===")
        print(f"count-err   mean {ce.mean():+.2f}  abs {np.abs(ce).mean():.2f}")
        print(f"mean IoU    {mi.mean():.3f}")
        print(f"recall@.5   {g5.sum()}/{ng.sum()} leaves ({100*g5.sum()/ng.sum():.0f}%)")
        print(f"point acc   {np.nanmean(pa):.3f}")


if __name__ == "__main__":
    main()
