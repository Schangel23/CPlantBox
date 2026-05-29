"""Does multi-view ACCUMULATION beat the occlusion wall? (GT-validated upper bound)

The real-data multi-date probe (probe_multidate_coreg.py) showed registration is
feasible (ICP 2.46cm on a 2-day pair) and accumulation adds 37% novel points and
+4 leaves -- but with no GT we can't tell genuine occlusion recovery from
registration ghosting. This isolates the ACCUMULATION benefit under PERFECT
registration: the same synthetic plant is occluded from K independent viewpoints
(no growth, no registration error) and the views are unioned, then scored vs GT.

If recall jumps with K, multi-date co-registration is worth the registration
effort (the real probe shows registration works). If it does not, accumulation is
not the lever and the wall is intrinsic to the segmenter.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/eval_multiview_accum.py --n 12 --views 1 3 5
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


def accumulate_views(comp, clab, rng, k):
    """Union of k independent occluded views of the same complete cloud, then
    voxel-dedup. Labels carried through. Perfect registration (same plant)."""
    allp, alll = [], []
    for _ in range(k):
        p, l = J.occlude_labelled(comp.copy(), clab.copy(), rng)
        allp.append(p); alll.append(l)
    P = np.vstack(allp); L = np.concatenate(alll)
    # voxel-dedup (scale-invariant like the judge), keep first label per voxel
    vox = float(np.clip(0.012 * np.ptp(P[:, 2]), 0.1, 0.6))
    mn = P.min(0); ijk = np.floor((P - mn) / vox).astype(np.int64)
    _, u = np.unique(ijk, axis=0, return_index=True)
    return P[u], L[u]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--views", type=int, nargs="+", default=[1, 3, 5])
    a = ap.parse_args()

    res = {k: {"recall": [], "ng": [], "iou": [], "cerr": []} for k in a.views}
    for i in range(a.n):
        rng = np.random.default_rng(i)
        day = int(rng.integers(40, 93))
        plant = grow_plant(J.DEFAULT_XML, simulation_time=day, seed=i)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, clab = J.sample_labelled(mesh, 16384, rng)
        for k in a.views:
            vrng = np.random.default_rng(1000 + i)        # same view-stream per plant across k? no: k views
            pts, gt = accumulate_views(comp, clab, vrng, k)
            organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400)
            s = J.score(gt, J.predicted_labels(organs, pts))
            if s is None:
                continue
            res[k]["recall"].append(s["iou_ge50"]); res[k]["ng"].append(s["n_gt"])
            res[k]["iou"].append(s["mean_iou"]); res[k]["cerr"].append(s["count_err"])
        print(f"[{i+1}/{a.n}] d{day:3d} done")

    print(f"\n=== MULTI-VIEW ACCUMULATION (perfect registration, N={a.n}) ===")
    print(f"{'views':>6} {'recall@.5':>12} {'meanIoU':>9} {'count-err':>10}")
    for k in a.views:
        ng = sum(res[k]["ng"]); rc = sum(res[k]["recall"])
        iou = np.mean(res[k]["iou"]) if res[k]["iou"] else 0
        ce = np.mean(res[k]["cerr"]) if res[k]["cerr"] else 0
        print(f"{k:>6} {rc:>4}/{ng:<4} ({100*rc/max(ng,1):>3.0f}%) {iou:>9.3f} {ce:>+10.2f}")


if __name__ == "__main__":
    main()
