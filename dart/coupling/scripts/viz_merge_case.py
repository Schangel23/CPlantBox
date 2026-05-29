"""Render the worst merge: the predicted leaf that spans the most GT leaves.
GT (coloured by true id) vs basal-only prediction, with the merged predicted
instance highlighted. Tells us the geometry of a merge (azimuthal? stacked?
tangled?) so we can choose a split mechanism.
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg


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
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.cache, "cloud_*.npz")))
    # find global worst merge
    worst = None  # (span, file, pred_id, gt_list)
    cache = {}
    for f in files:
        d = np.load(f); pts, gt = d["pts"], d["gt"]
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment",
                                              pseudostem_basal_only=True)
        pred = pred_full(organs, pts); cache[f] = (pts, gt, pred)
        for p in [x for x in np.unique(pred) if x > 0]:
            pm = pred == p
            owned = []
            for g in [x for x in np.unique(gt) if x != 0]:
                gm = gt == g
                if int((gm & pm).sum()) >= 0.4 * int(gm.sum()) and int(gm.sum()) >= 50:
                    owned.append(g)
            if len(owned) >= 2 and (worst is None or len(owned) > worst[0]):
                worst = (len(owned), f, p, owned)
    span, f, pid, owned = worst
    pts, gt, pred = cache[f]
    print(f"worst merge: {os.path.basename(f)} pred leaf {pid} spans {span} GT leaves {owned}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 7))
    cmap = plt.get_cmap("tab20")
    # GT
    for g in [x for x in np.unique(gt) if x != 0]:
        m = gt == g; ax[0].scatter(pts[m,0], pts[m,2], s=3, color=cmap((g-1)%20))
    ax[0].scatter(pts[gt==0,0], pts[gt==0,2], s=3, c="0.8")
    ax[0].set_title("GT (by true leaf id)")
    # pred
    for p in [x for x in np.unique(pred) if x > 0]:
        m = pred == p; ax[1].scatter(pts[m,0], pts[m,2], s=3, color=cmap((p-1)%20))
    ax[1].scatter(pts[pred<=0,0], pts[pred<=0,2], s=3, c="0.8")
    ax[1].set_title("prediction (basal-only)")
    # merged instance, coloured by its TRUE gt id, in azimuth view
    pm = pred == pid
    cx, cy = pts[gt==0,0].mean() if (gt==0).any() else pts[:,0].mean(), \
             pts[gt==0,1].mean() if (gt==0).any() else pts[:,1].mean()
    az = np.degrees(np.arctan2(pts[pm,1]-cy, pts[pm,0]-cx))
    for g in owned:
        sel = pm & (gt==g)
        ax[2].scatter(az[(gt[pm]==g)], pts[pm,2][gt[pm]==g], s=4, color=cmap((g-1)%20), label=f"gt{g}")
    ax[2].set_title(f"merged pred {pid}: azimuth vs z (colour=true id)")
    ax[2].set_xlabel("azimuth (deg)"); ax[2].set_ylabel("z (cm)"); ax[2].legend(fontsize=7)
    for k in (0,1): ax[k].set_aspect("equal"); ax[k].set_xlabel("x"); ax[k].set_ylabel("z")
    fig.suptitle(f"worst merge ({os.path.basename(f)}, span {span})")
    fig.tight_layout()
    out = "dart/coupling/output/merge_case.png"; fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
