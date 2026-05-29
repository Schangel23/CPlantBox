"""Reassessment visual: winning-stack prediction vs GT across a difficulty
spread (best / median / worst by per-plant recall@.5). Each plant = GT | pred
pair, coloured by leaf instance. Title carries n_gt, n_pred, recall@.5.

Winning stack: pseudostem_basal_only + split_merged + assign=geodesic (full-path).

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/viz_state_2026-05-29.py
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

CACHE = "dart/coupling/output/synth_cache"


def pred_labels(organs, pts):
    pred = np.zeros(len(pts), dtype=np.int64)
    tree = cKDTree(pts); lid = 0
    for k in sorted(organs):
        if k.startswith("leaf") and len(organs[k]):
            lid += 1; pred[tree.query(organs[k])[1]] = lid
    return pred


def recall_at(gt, pred, thr=0.5):
    g = [x for x in np.unique(gt) if x]; p = [x for x in np.unique(pred) if x]
    if not g: return 0.0, 0, 0
    iou = np.zeros((len(g), len(p)))
    for i, gg in enumerate(g):
        gm = gt == gg
        for j, pp in enumerate(p):
            pm = pred == pp; inter = int((gm & pm).sum())
            if inter: iou[i, j] = inter / int((gm | pm).sum())
    if p:
        r, c = linear_sum_assignment(-iou)
        rec = int((iou[r, c] >= thr).sum())
    else:
        rec = 0
    return rec / len(g), len(g), len(p)


def draw(ax, pts, lab, title, show_nonleaf=True):
    cmap = plt.get_cmap("tab20")
    if show_nonleaf:
        ax.scatter(pts[lab <= 0, 0], pts[lab <= 0, 2], s=2, c="0.8")
    for l in [x for x in np.unique(lab) if x > 0]:
        m = lab == l
        ax.scatter(pts[m, 0], pts[m, 2], s=2, color=cmap((int(l) - 1) % 20))
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    files = sorted(glob.glob(os.path.join(CACHE, "cloud_*.npz")))
    recs = []
    for f in files:
        d = np.load(f); pts, gt = d["pts"], d["gt"]
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="geodesic",
                                             pseudostem_basal_only=True, split_components=True)
        pred = pred_labels(organs, pts)
        rec, ng, npd = recall_at(gt, pred)
        recs.append((rec, ng, npd, f, pts, gt, pred))
    recs.sort(key=lambda x: -x[0])
    overall = np.mean([r[0] for r in recs])
    # pick best 2, middle 2, worst 2 -> 3 rows x (2 plants); each plant = GT|pred
    n = len(recs)
    pick = [recs[0], recs[1], recs[n//2 - 1], recs[n//2], recs[-2], recs[-1]]
    tags = ["best", "best", "median", "median", "worst", "worst"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for i, (sel, tag) in enumerate(zip(pick, tags)):
        rec, ng, npd, f, pts, gt, pred = sel
        name = os.path.basename(f).replace("cloud_", "").replace(".npz", "")
        r, c = i // 2, 2 * (i % 2)
        draw(axes[r, c], pts, gt, f"GT [{tag} #{name}]  {ng} true leaves")
        draw(axes[r, c + 1], pts, pred,
             f"PRED  {npd} leaves  recall@.5 = {rec*100:.0f}%")
    fig.suptitle(f"MonGraphSeg winning stack (basal + geodesic + component-split)   |   "
                 f"mean recall@.5 = {overall*100:.0f}%   (n={n}, GT|PRED pairs)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "dart/coupling/output/mongraphseg_state_compsplit_2026-05-29.png"
    fig.savefig(out, dpi=120); print("wrote", out)

    # per-plant recall distribution (the headline reassessment signal)
    pr = np.array([r[0] for r in recs]) * 100
    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pr, bins=np.arange(0, 101, 10), color="steelblue", edgecolor="k")
    ax.axvline(overall*100, color="red", ls="--", label=f"mean {overall*100:.0f}%")
    ax.axvline(50, color="green", ls=":", label="recall@.5 threshold band")
    ax.set_xlabel("per-plant recall@.5 (%)"); ax.set_ylabel("# plants")
    ax.set_title(f"Per-plant recall spread (n={n}): best {pr.max():.0f}%, "
                 f"worst {pr.min():.0f}%, max {pr.max():.0f}%"); ax.legend()
    fig2.tight_layout()
    out2 = "dart/coupling/output/mongraphseg_recall_hist_compsplit_2026-05-29.png"
    fig2.savefig(out2, dpi=120); print("wrote", out2)
    print(f"per-plant recall: {sorted([round(r[0]*100) for r in recs], reverse=True)}")


if __name__ == "__main__":
    main()
