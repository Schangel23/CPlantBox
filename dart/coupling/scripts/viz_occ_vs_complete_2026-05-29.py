"""Occluded vs fully-intact, same plants, winning stack. For 3 paired clouds
(same seed/day) show: occ GT | occ PRED | complete GT | complete PRED, with
per-plant recall@.5. Isolates how much occlusion vs the model limits recall.
"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import mongraphseg_graph as mg

OCC = "dart/coupling/output/synth_cache"
CMP = "dart/coupling/output/synth_cache_complete"
SEEDS = [6, 0, 17]   # whorl, mature, median


def pred_labels(organs, pts):
    pred = np.zeros(len(pts), dtype=np.int64); tree = cKDTree(pts); lid = 0
    for k in sorted(organs):
        if k.startswith("leaf") and len(organs[k]):
            lid += 1; pred[tree.query(organs[k])[1]] = lid
    return pred


def recall_at(gt, pred, thr=0.5):
    g = [x for x in np.unique(gt) if x]; p = [x for x in np.unique(pred) if x]
    if not g: return 0.0
    iou = np.zeros((len(g), len(p)))
    for i, gg in enumerate(g):
        gm = gt == gg
        for j, pp in enumerate(p):
            inter = int((gm & (pred == pp)).sum())
            if inter: iou[i, j] = inter / int((gm | (pred == pp)).sum())
    if not p: return 0.0
    r, c = linear_sum_assignment(-iou)
    return int((iou[r, c] >= thr).sum()) / len(g)


def seg(pts):
    o = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="geodesic",
                                    pseudostem_basal_only=True, split_merged=True)
    return pred_labels(o, pts)


def draw(ax, pts, lab, title):
    cmap = plt.get_cmap("tab20")
    ax.scatter(pts[lab <= 0, 0], pts[lab <= 0, 2], s=2, c="0.85")
    for l in [x for x in np.unique(lab) if x > 0]:
        m = lab == l; ax.scatter(pts[m, 0], pts[m, 2], s=2, color=cmap((int(l)-1) % 20))
    ax.set_title(title, fontsize=9); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def load(cache, seed):
    d = np.load(os.path.join(cache, f"cloud_{seed:03d}.npz"))
    return d["pts"], d["gt"]


def main():
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row, seed in enumerate(SEEDS):
        po, go = load(OCC, seed); pc, gc = load(CMP, seed)
        predo = seg(po); predc = seg(pc)
        ro, rc = recall_at(go, predo), recall_at(gc, predc)
        ng_o = len([x for x in np.unique(go) if x]); ng_c = len([x for x in np.unique(gc) if x])
        draw(axes[row, 0], po, go, f"OCCLUDED GT  #{seed}  {ng_o} leaves ({len(po)} pts)")
        draw(axes[row, 1], po, predo, f"occluded PRED  recall@.5={ro*100:.0f}%")
        draw(axes[row, 2], pc, gc, f"COMPLETE GT  #{seed}  {ng_c} leaves ({len(pc)} pts)")
        draw(axes[row, 3], pc, predc, f"complete PRED  recall@.5={rc*100:.0f}%")
    fig.suptitle("Occluded vs fully-intact synthetic clouds — same plants, winning stack "
                 "(basal+split+geodesic)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "dart/coupling/output/mongraphseg_occ_vs_complete_2026-05-29.png"
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
