"""Skeleton-graph metric judge -- validates the CPlantBox-tuning targets.

Per-point IoU recall is capped ~39% by pseudostem theft of leaf-blade bases, but
that is NOT what the project needs. CPlantBox calibration uses the SKELETON-GRAPH
metrics: leaf COUNT, per-leaf INSERTION HEIGHT + ANGLE, and INTERNODE spacing.
The skeleton has ~correct terminal count (count-err -0.08), so these may be usable
even though per-point IoU is not. This judge measures them directly against
synthetic ground truth.

GT per leaf (from the COMPLETE pre-occlusion cloud, so it is the TRUE value the
occluded scan must recover):
  * insertion height  = min-z of the complete leaf points
  * insertion angle   = angle of the leaf principal axis from vertical (deg)
Predicted per leaf (from segment_plant_pseudostem on the OCCLUDED cloud):
  * insertion height  = min-z of the predicted leaf cluster
  * insertion angle   = principal-axis angle of the predicted cluster
GT<->pred matched by Hungarian on |insertion-height difference|. Reports
leaf-count error, insertion-height MAE, insertion-angle MAE, internode MAE.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/eval_skeleton_metrics.py --n 12
"""
import argparse
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


def axis_angle_from_vertical(P):
    """Angle (deg) of a point set's principal axis from the +z vertical."""
    if len(P) < 3:
        return np.nan
    c = P - P.mean(0)
    _, V = np.linalg.eigh(c.T @ c)
    axis = V[:, 2]
    cosang = abs(axis[2]) / (np.linalg.norm(axis) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))


def leaf_metrics(P):
    """(insertion_z, angle_from_vertical_deg, length_cm) for a leaf point set."""
    z0 = float(P[:, 2].min())
    ang = axis_angle_from_vertical(P)
    c = P - P.mean(0)
    _, V = np.linalg.eigh(c.T @ c)
    t = c @ V[:, 2]
    length = float(t.max() - t.min())
    return z0, ang, length


def gt_leaf_sets(comp, comp_lab):
    return [comp[comp_lab == g] for g in np.unique(comp_lab) if g != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--assign", default="segment")
    ap.add_argument("--nsk", type=int, default=400)
    ap.add_argument("--gt", choices=["complete", "visible"], default="complete",
                    help="complete=true leaf count (incl occlusion-erased leaves); "
                         "visible=only leaves surviving occlusion (fair segmenter test)")
    a = ap.parse_args()

    cnt_err, z_mae, ang_mae, inode_mae = [], [], [], []
    matched_frac = []
    for i in range(a.n):
        rng = np.random.default_rng(i)
        day = int(rng.integers(40, 93))
        plant = grow_plant(J.DEFAULT_XML, simulation_time=day, seed=i)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, clab = J.sample_labelled(mesh, 16384, rng)
        pts, gt = J.occlude_labelled(comp, clab, rng)

        if a.gt == "visible":
            gt_sets = [P for P in gt_leaf_sets(pts, gt) if len(P) >= 10]
        else:
            gt_sets = [P for P in gt_leaf_sets(comp, clab) if len(P) >= 10]
        gt_m = np.array([leaf_metrics(P) for P in gt_sets])           # (G,3)
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=a.nsk, assign=a.assign)
        pr_sets = [organs[k] for k in sorted(organs)
                   if k.startswith("leaf") and len(organs[k]) >= 10]
        if not pr_sets:
            cnt_err.append(-len(gt_sets)); continue
        pr_m = np.array([leaf_metrics(P) for P in pr_sets])           # (P,3)

        cnt_err.append(len(pr_sets) - len(gt_sets))
        # match by insertion-height distance
        C = np.abs(gt_m[:, None, 0] - pr_m[None, :, 0])
        r, c = linear_sum_assignment(C)
        # only accept matches within 8 cm insertion height (else it's a miss)
        good = C[r, c] <= 8.0
        rr, cc = r[good], c[good]
        matched_frac.append(len(rr) / max(len(gt_sets), 1))
        if len(rr):
            z_mae.append(np.mean(np.abs(gt_m[rr, 0] - pr_m[cc, 0])))
            am = np.abs(gt_m[rr, 1] - pr_m[cc, 1])
            am = am[np.isfinite(am)]
            if len(am):
                ang_mae.append(np.mean(am))
        # internode: sorted insertion-height gaps, GT vs pred (matched count)
        gz = np.sort(gt_m[:, 0]); pz = np.sort(pr_m[:, 0])
        kk = min(len(gz), len(pz))
        if kk >= 2:
            inode_mae.append(np.mean(np.abs(np.diff(gz[:kk]) - np.diff(pz[:kk]))))
        print(f"[{i+1}/{a.n}] d{day:3d} GT {len(gt_sets):2d} pred {len(pr_sets):2d} "
              f"matched {len(rr):2d} | z-MAE {z_mae[-1] if z_mae else np.nan:.1f}cm")

    ce = np.array(cnt_err)
    print(f"\n=== SKELETON-GRAPH METRICS (assign={a.assign}, nsk={a.nsk}, N={a.n}) ===")
    print(f"leaf-count error    mean {ce.mean():+.2f}   abs {np.abs(ce).mean():.2f}")
    print(f"matched leaf frac   {np.mean(matched_frac)*100:.0f}%  (pred within 8cm of a GT insertion)")
    print(f"insertion-height MAE {np.mean(z_mae):.2f} cm" if z_mae else "insertion-height MAE n/a")
    print(f"insertion-angle  MAE {np.mean(ang_mae):.1f} deg" if ang_mae else "insertion-angle MAE n/a")
    print(f"internode-spacing MAE {np.mean(inode_mae):.2f} cm" if inode_mae else "internode MAE n/a")


if __name__ == "__main__":
    main()
