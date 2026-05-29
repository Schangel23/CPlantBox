"""Why are PARTIAL leaves (best IoU in [lo,hi)) stuck under .5?

IoU = inter / (inter + gt_only + pred_only). A leaf can sit at .3-.5 because:
  COVERAGE loss  - gt_only large: GT-leaf points the prediction MISSED (they
                   leaked to pseudostem / other leaves / unassigned). Fix =
                   capture more (theft/bleed reduction).
  OVER-CLAIM     - pred_only large: the predicted instance includes points that
                   are NOT this GT leaf (false positives from neighbours /
                   GT-pseudostem). Fix = trim the prediction.
For each matched (gt,pred) PARTIAL pair we report the mean gt_only and pred_only
fractions (of the union), and for the missed GT points their destination.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/diag_partial.py --cache dart/coupling/output/synth_cache
"""
import argparse, glob, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import mongraphseg_graph as mg

MIN_PTS = 50


def pred_full(organs, pts):
    pred = np.zeros(len(pts), dtype=np.int64)
    tree = cKDTree(pts)
    if "pseudostem" in organs and len(organs["pseudostem"]):
        pred[tree.query(organs["pseudostem"])[1]] = -1
    lid = 0
    for k in sorted(organs):
        if k.startswith("leaf") and len(organs[k]):
            lid += 1; pred[tree.query(organs[k])[1]] = lid
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="dart/coupling/output/synth_cache")
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.50)
    ap.add_argument("--assign", default="segment")
    ap.add_argument("--distal_seed_start", type=float, default=0.5)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.cache, "cloud_*.npz")))
    rows = []          # (gt_only_frac, pred_only_frac, inter_frac, miss_ps, miss_other, miss_un)
    for f in files:
        d = np.load(f); pts, gt = d["pts"], d["gt"]
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign=a.assign,
                                             pseudostem_basal_only=True, split_merged=True,
                                             distal_seed_start=a.distal_seed_start,
                                             geodesic_k=15)
        pred = pred_full(organs, pts)
        pr_ids = [x for x in np.unique(pred) if x > 0]
        for g in [x for x in np.unique(gt) if x != 0]:
            gm = gt == g; ng = int(gm.sum())
            if ng < MIN_PTS:
                continue
            best_iou, best_p = 0.0, 0
            for p in pr_ids:
                pm = pred == p; inter = int((gm & pm).sum())
                if inter == 0: continue
                iou = inter / int((gm | pm).sum())
                if iou > best_iou: best_iou, best_p = iou, p
            if not (a.lo <= best_iou < a.hi) or best_p == 0:
                continue
            pm = pred == best_p
            inter = int((gm & pm).sum())
            union = int((gm | pm).sum())
            gt_only = int((gm & ~pm).sum())
            pred_only = int((pm & ~gm).sum())
            # destination of MISSED gt points
            miss = gm & ~pm
            nmiss = max(int(miss.sum()), 1)
            miss_ps = float((pred[miss] == -1).sum()) / nmiss
            miss_other = float((pred[miss] > 0).sum()) / nmiss
            miss_un = float((pred[miss] == 0).sum()) / nmiss
            rows.append((gt_only/union, pred_only/union, inter/union,
                         miss_ps, miss_other, miss_un))
    R = np.array(rows)
    print(f"\n=== PARTIAL leaves (IoU in [{a.lo},{a.hi}), {len(R)} leaves) ===")
    print(f"  mean IoU (inter/union)     {R[:,2].mean():.3f}")
    print(f"  gt_only  (missed coverage) {R[:,0].mean()*100:4.0f}% of union")
    print(f"  pred_only(over-claim/FP)   {R[:,1].mean()*100:4.0f}% of union")
    print(f"\n  of the MISSED gt points: -> pseudostem {R[:,3].mean()*100:3.0f}%  "
          f"-> other leaf {R[:,4].mean()*100:3.0f}%  -> unassigned {R[:,5].mean()*100:3.0f}%")
    # how many would cross .5 if we eliminated only over-claim? only coverage?
    # IoU_no_fp = inter/(inter+gt_only) ; IoU_no_miss = inter/(inter+pred_only)
    inter = R[:,2]; gto = R[:,0]; pro = R[:,1]
    iou_no_fp = inter/(inter+gto)
    iou_no_miss = inter/(inter+pro)
    print(f"\n  if over-claim removed: {(iou_no_fp>=0.5).mean()*100:3.0f}% cross .5")
    print(f"  if coverage perfected: {(iou_no_miss>=0.5).mean()*100:3.0f}% cross .5")


if __name__ == "__main__":
    main()
