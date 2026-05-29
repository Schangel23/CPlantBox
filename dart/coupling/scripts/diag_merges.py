"""Under-count autopsy: where do the MISSED-but-recoverable GT leaves go?

count-err is -1.71 (segmenter misses ~1.7 leaves/plant). For each GT leaf with
>=MIN_PTS surviving points (i.e. recoverable, not an occlusion casualty) that is
NOT recovered (best IoU < .5), classify its dominant failure:

  MERGED   - its points sit mostly in a predicted leaf that ALSO owns >=MIN_PTS
             of another GT leaf (two GT blades -> one predicted instance)
  STOLEN   - its points went mostly to the pseudostem (label 0)
  PRUNED   - its points are mostly unassigned (no instance / branch removed)
  PARTIAL  - has a best match but IoU just under .5 (truncated, not missing)

Also reports, per predicted leaf, how many GT leaves it spans (merge factor).

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/diag_merges.py --cache dart/coupling/output/synth_cache --basal_only
"""
import argparse
import glob
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg

MIN_PTS = 50   # "recoverable" floor from the detectability probe


def pred_full(organs, pts):
    """0=unassigned, -1=pseudostem, 1..L=leaves."""
    pred = np.zeros(len(pts), dtype=np.int64)
    tree = cKDTree(pts)
    if "pseudostem" in organs and len(organs["pseudostem"]):
        pred[tree.query(organs["pseudostem"])[1]] = -1
    lid = 0
    for k in sorted(organs):
        if k.startswith("leaf") and len(organs[k]):
            lid += 1
            pred[tree.query(organs[k])[1]] = lid
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="dart/coupling/output/synth_cache")
    ap.add_argument("--basal_only", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.cache, "cloud_*.npz")))

    cls = {"MERGED": 0, "STOLEN": 0, "PRUNED": 0, "PARTIAL": 0, "RECOVERED": 0}
    n_recoverable = 0
    merge_spans = []   # per predicted leaf: # of GT leaves it dominantly owns
    for f in files:
        d = np.load(f)
        pts, gt = d["pts"], d["gt"]
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment",
                                              pseudostem_basal_only=a.basal_only)
        pred = pred_full(organs, pts)
        gt_ids = [x for x in np.unique(gt) if x != 0]
        pr_ids = [x for x in np.unique(pred) if x > 0]

        # which GT leaf does each predicted leaf most own?
        owns = {p: [] for p in pr_ids}
        # precompute, per GT leaf, its best pred + iou + dominant destination
        for g in gt_ids:
            gm = gt == g
            ng = int(gm.sum())
            if ng < MIN_PTS:
                continue
            n_recoverable += 1
            best_iou, best_p = 0.0, 0
            for p in pr_ids:
                pm = pred == p
                inter = int((gm & pm).sum())
                if inter == 0:
                    continue
                iou = inter / int((gm | pm).sum())
                if iou > best_iou:
                    best_iou, best_p = iou, p
            if best_iou >= 0.5:
                cls["RECOVERED"] += 1
                if best_p:
                    owns[best_p].append(g)
                continue
            # failure: dominant destination of this GT leaf's points
            to_ps = float((pred[gm] == -1).mean())
            to_un = float((pred[gm] == 0).mean())
            to_leaf = float((pred[gm] > 0).mean())
            if best_iou > 0.30:
                cls["PARTIAL"] += 1
            elif to_leaf >= max(to_ps, to_un):
                cls["MERGED"] += 1
                if best_p:
                    owns[best_p].append(g)
            elif to_ps >= to_un:
                cls["STOLEN"] += 1
            else:
                cls["PRUNED"] += 1
        for p, gl in owns.items():
            if gl:
                merge_spans.append(len(gl))

    print(f"\n=== under-count autopsy (basal_only={a.basal_only}, "
          f"{n_recoverable} recoverable GT leaves >= {MIN_PTS} pts) ===")
    for k in ["RECOVERED", "PARTIAL", "MERGED", "STOLEN", "PRUNED"]:
        print(f"  {k:10s} {cls[k]:4d}  ({cls[k]/max(n_recoverable,1)*100:4.0f}%)")
    ms = np.array(merge_spans) if merge_spans else np.array([0])
    multi = int((ms >= 2).sum())
    print(f"\n  predicted leaves spanning >=2 GT leaves (merges): {multi} "
          f"/ {len(ms)} pred-leaves  (max span {int(ms.max())})")


if __name__ == "__main__":
    main()
