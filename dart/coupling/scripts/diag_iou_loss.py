"""Where is the IoU lost? Per GT leaf, categorise the failure.

Right leaf count but recall@.5 stuck at 39% => predicted leaves are spatially
PARTIAL. This decomposes, for each GT leaf, where its points actually go:
  * frac to its best-matching predicted leaf (the recovered blade)
  * frac to the PSEUDOSTEM instance (label 0)        -> pseudostem theft
  * frac to OTHER predicted leaves                    -> leaf<->leaf bleed
  * frac unassigned
and reports the best IoU. Tells us whether to fix the pseudostem split
(theft), the insertion point (truncation), or leaf separation (bleed).

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/diag_iou_loss.py --n 6
"""
import argparse
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


def pred_full(organs, pts):
    """Per-point predicted label: 0 = pseudostem/non-leaf, 1..L = leaf id."""
    pred = np.zeros(len(pts), dtype=np.int64)
    tree = cKDTree(pts)
    # pseudostem first (so leaves can overwrite shared nearest assignments)
    if "pseudostem" in organs and len(organs["pseudostem"]):
        _, idx = tree.query(organs["pseudostem"]); pred[idx] = -1
    lid = 0
    for k in sorted(organs):
        if not k.startswith("leaf") or len(organs[k]) == 0:
            continue
        lid += 1
        _, idx = tree.query(organs[k]); pred[idx] = lid
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--assign", default="segment")
    ap.add_argument("--basal_only", action="store_true")
    a = ap.parse_args()
    agg = []
    for i in range(a.n):
        rng = np.random.default_rng(i)
        day = int(rng.integers(40, 93))
        plant = grow_plant(J.DEFAULT_XML, simulation_time=day, seed=i)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, clab = J.sample_labelled(mesh, 16384, rng)
        pts, gt = J.occlude_labelled(comp, clab, rng)
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign=a.assign,
                                              pseudostem_basal_only=a.basal_only)
        pred = pred_full(organs, pts)
        for g in [x for x in np.unique(gt) if x != 0]:
            gm = gt == g
            ng = int(gm.sum())
            if ng < 10:
                continue
            # best predicted leaf for this GT leaf
            best_iou, best_p, best_frac = 0.0, 0, 0.0
            for p in [x for x in np.unique(pred[gm]) if x > 0]:
                inter = int((gm & (pred == p)).sum())
                union = int((gm | (pred == p)).sum())
                iou = inter / union
                if iou > best_iou:
                    best_iou, best_p, best_frac = iou, p, inter / ng
            to_ps = float((pred[gm] == -1).mean())
            to_other = float(((pred[gm] > 0) & (pred[gm] != best_p)).mean())
            to_unassigned = float((pred[gm] == 0).mean())
            agg.append((best_iou, best_frac, to_ps, to_other, to_unassigned))
    A = np.array(agg)
    print(f"\n=== IoU-loss decomposition (assign={a.assign}, {len(A)} GT leaves) ===")
    print(f"best IoU                mean {A[:,0].mean():.3f}   <.5: {(A[:,0]<0.5).mean()*100:.0f}% of leaves")
    print(f"GT-leaf pts -> best pred-leaf   {A[:,1].mean()*100:.0f}%")
    print(f"GT-leaf pts -> PSEUDOSTEM       {A[:,2].mean()*100:.0f}%   (theft)")
    print(f"GT-leaf pts -> OTHER pred-leaf  {A[:,3].mean()*100:.0f}%   (leaf-leaf bleed)")
    print(f"GT-leaf pts -> unassigned       {A[:,4].mean()*100:.0f}%")
    # among the <.5 leaves, what's the dominant loss channel?
    lo = A[A[:,0] < 0.5]
    if len(lo):
        print(f"\n  among IoU<.5 leaves ({len(lo)}): to-best {lo[:,1].mean()*100:.0f}%  "
              f"to-pseudostem {lo[:,2].mean()*100:.0f}%  to-other {lo[:,3].mean()*100:.0f}%  "
              f"unassigned {lo[:,4].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
