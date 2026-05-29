"""Is 100% recall@.5 even reachable on this synthetic data, or is there an
occlusion ceiling? For every GT leaf we bucket by surviving point count and
report: detected-at-all (best IoU>0), recovered (best IoU>=.5). If the missed
leaves are mostly tiny (few points), that's an information ceiling no segmenter
fix can cross. If missed leaves have plenty of points, they were merged/stolen
-> fixable.

Runs against the cached clouds (fast).
    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/diag_detectability.py --cache dart/coupling/output/synth_cache --basal_only
"""
import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg

BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="dart/coupling/output/synth_cache")
    ap.add_argument("--basal_only", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.cache, "cloud_*.npz")))

    # rows: (n_pts, best_iou)
    rows = []
    for f in files:
        d = np.load(f)
        pts, gt = d["pts"], d["gt"]
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment",
                                              pseudostem_basal_only=a.basal_only)
        pred = J.predicted_labels(organs, pts)
        gt_ids = [x for x in np.unique(gt) if x != 0]
        pr_ids = [x for x in np.unique(pred) if x != 0]
        for g in gt_ids:
            gm = gt == g
            ng = int(gm.sum())
            best = 0.0
            for p in pr_ids:
                pm = pred == p
                inter = int((gm & pm).sum())
                if inter == 0:
                    continue
                best = max(best, inter / int((gm | pm).sum()))
            rows.append((ng, best))
    R = np.array(rows, float)
    print(f"\n=== detectability vs surviving point count "
          f"(basal_only={a.basal_only}, {len(R)} GT leaves) ===")
    print(" pts bucket | n_leaves | detected(IoU>0) | recovered(IoU>=.5) | mean best-IoU")
    for lo, hi in BUCKETS:
        m = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if not m.any():
            continue
        sub = R[m]
        det = (sub[:, 1] > 0).mean() * 100
        rec = (sub[:, 1] >= 0.5).mean() * 100
        lbl = f"{lo}-{hi if hi < 10**8 else '+'}"
        print(f"  {lbl:>9} | {int(m.sum()):8d} | {det:6.0f}%         | "
              f"{rec:6.0f}%            | {sub[:,1].mean():.3f}")
    print(f"\n  overall recall@.5 = {(R[:,1]>=0.5).mean()*100:.0f}%  "
          f"| ceiling if all leaves with >=50 pts recovered = "
          f"{((R[:,0]>=50)).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
