"""Visualize WHY intact-synthetic leaf segmentation caps ~59%: adjacent maize
blades touch below the point-sampling density, so the surface is one manifold
that no local cue can split.

4 panels on one example plant:
  (a) GT leaves coloured (X-Z) -- the dense intact canopy.
  (b) zoom on the closest-touching leaf pair -- the two surfaces interleave.
  (c) per-leaf intra-spacing vs min inter-leaf gap -- gaps fall BELOW spacing.
  (d) geodesic-from-collar field with the persistence tips found (2) vs the
      true GT tips (one per leaf): only leaves with a real gap are recoverable.

Run from CPlantBox root:
  PYTHONPATH=. cpbenv/bin/python dart/coupling/scripts/viz_touching_floor_2026-05-30.py
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
import mongraphseg_graph as mg
from geodesic_assignment import build_knn_graph

CACHE = "dart/coupling/output/synth_cache_complete/cloud_000.npz"
OUT = "dart/coupling/output/mongraphseg_touching_floor_2026-05-30.png"


def main():
    d = np.load(CACHE)
    pts, gt = d["pts"], d["gt"]
    leaves = sorted(set(gt.tolist()) - {0})
    cmap = plt.get_cmap("tab20")

    # ---- per-leaf intra spacing & min inter-leaf gap -------------------------
    trees = {l: cKDTree(pts[gt == l]) for l in leaves}
    intra, inter, closest = {}, {}, {}
    for l in leaves:
        P = pts[gt == l]
        if len(P) < 3:
            continue
        intra[l] = float(np.median(trees[l].query(P, k=2)[0][:, 1]))
        best, bl = 1e9, -1
        for m in leaves:
            if m == l or len(pts[gt == m]) < 3:
                continue
            dm = trees[m].query(P)[0].min()
            if dm < best:
                best, bl = dm, m
        inter[l], closest[l] = best, bl
    real = [l for l in leaves if l in intra]
    # closest-touching pair among SUBSTANTIAL blades (so the zoom shows two real
    # laminae interleaving, not a tiny fragment)
    big = [l for l in real if (gt == l).sum() >= 150
           and (gt == closest[l]).sum() >= 150]
    a_leaf = min(big or real, key=lambda l: inter[l])
    b_leaf = closest[a_leaf]

    # ---- geodesic field + persistence tips on the non-stem surface ----------
    _, dbg = mg.segment_plant_pseudostem(
        pts, n_skel_nodes=400, pseudostem_basal_only=False, return_debug=True)
    ps_mask = dbg["leaf_id"] == 0
    idx = np.where(~ps_mask)[0]
    Pn = pts[idx]
    dd = KDTree(Pn).query(Pn, k=4)[0]
    edge = float(np.clip(3 * np.median(dd[:, 1:]), 1.5, 6))
    G = build_knn_graph(Pn, k=12, max_edge_cm=edge)
    src = np.where(KDTree(pts[ps_mask]).query(Pn)[0] <= 3.0)[0]
    geo = dijkstra(G, directed=False, indices=src)
    geo = geo.min(axis=0)
    geo_plot = np.where(np.isfinite(geo), geo, np.nan)
    tips = mg._persistent_tips(G.tocsr(), geo, 6.0, 6.0)

    # true GT tips = farthest-from-base point of each leaf
    base_xyz = pts[ps_mask].mean(0)
    gt_tips = []
    for l in real:
        P = pts[gt == l]
        gt_tips.append(P[np.argmax(np.linalg.norm(P - base_xyz, axis=1))])
    gt_tips = np.array(gt_tips)

    # ============================ FIGURE =====================================
    fig, ax = plt.subplots(1, 4, figsize=(22, 7.5))

    # (a) GT leaves
    for l in leaves:
        m = gt == l
        ax[0].scatter(pts[m, 0], pts[m, 2], s=3, color=cmap((l - 1) % 20))
    ax[0].scatter(pts[gt == 0, 0], pts[gt == 0, 2], s=3, c="0.82", label="stem")
    ax[0].set_title(f"(a) GT: {len(real)} leaves, intact\n(each colour = one true leaf)")
    ax[0].set_xlabel("x (cm)"); ax[0].set_ylabel("z (cm)"); ax[0].set_aspect("equal")
    # mark the zoom box around the closest pair
    Pa, Pb = pts[gt == a_leaf], pts[gt == b_leaf]
    cz = np.vstack([Pa, Pb])
    cx0, cx1 = cz[:, 0].min(), cz[:, 0].max()
    cz0, cz1 = cz[:, 2].min(), cz[:, 2].max()
    ax[0].add_patch(plt.Rectangle((cx0, cz0), cx1 - cx0, cz1 - cz0,
                    fill=False, ec="red", lw=1.5))

    # (b) zoom on the closest-touching pair, contrasting colours, tight crop
    #     around the contact (the point on a_leaf nearest leaf b)
    dab = trees[b_leaf].query(Pa)[0]
    cpt = Pa[np.argmin(dab)]
    R = 6.0
    ax[1].scatter(Pa[:, 0], Pa[:, 2], s=16, color="#1f77b4", label=f"leaf {a_leaf}")
    ax[1].scatter(Pb[:, 0], Pb[:, 2], s=16, color="#ff7f0e", label=f"leaf {b_leaf}")
    ax[1].set_xlim(cpt[0] - R, cpt[0] + R); ax[1].set_ylim(cpt[2] - R, cpt[2] + R)
    ax[1].set_title(f"(b) closest blades {a_leaf} & {b_leaf} (zoom at contact)\n"
                    f"gap {inter[a_leaf]:.2f} cm  <  on-leaf spacing {intra[a_leaf]:.2f} cm")
    ax[1].set_xlabel("x (cm)"); ax[1].set_ylabel("z (cm)")
    ax[1].set_aspect("equal"); ax[1].legend(markerscale=1.6, loc="upper right")

    # (c) intra spacing vs inter-leaf gap
    xs = np.arange(len(real))
    ax[2].bar(xs - 0.2, [intra[l] for l in real], width=0.4, color="0.5",
              label="intra-leaf spacing")
    ax[2].bar(xs + 0.2, [inter[l] for l in real], width=0.4, color="crimson",
              label="min inter-leaf gap")
    ax[2].axhline(np.median([intra[l] for l in real]), ls="--", c="k", lw=1)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(real)
    ax[2].set_xlabel("leaf id"); ax[2].set_ylabel("distance (cm)")
    n_below = sum(inter[l] < intra[l] for l in real)
    ax[2].set_title(f"(c) {n_below}/{len(real)} leaves: gap < own spacing\n"
                    "= touching below sampling density")
    ax[2].legend()

    # (d) geodesic field + tips
    sc = ax[3].scatter(Pn[:, 0], Pn[:, 2], s=3, c=geo_plot, cmap="viridis")
    ax[3].scatter(gt_tips[:, 0], gt_tips[:, 2], marker="o", s=90,
                  facecolors="none", edgecolors="k", lw=1.5,
                  label=f"true tips ({len(gt_tips)})")
    Tt = Pn[tips]
    ax[3].scatter(Tt[:, 0], Tt[:, 2], marker="*", s=320, c="red",
                  edgecolors="k", label=f"persistence tips ({len(tips)})")
    ax[3].set_title("(d) geodesic-from-collar field\nonly leaves with a real "
                    "gap form a persistent peak")
    ax[3].set_xlabel("x (cm)"); ax[3].set_ylabel("z (cm)")
    ax[3].set_aspect("equal"); ax[3].legend(loc="upper right")
    fig.colorbar(sc, ax=ax[3], shrink=0.6, label="geodesic dist (cm)")

    fig.suptitle("Why intact-synthetic leaf segmentation caps ~59%: adjacent "
                 "blades touch -> one manifold, no local cue separates them",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=120)
    print("wrote", OUT)
    print(f"closest pair leaf {a_leaf}<->{b_leaf}: gap {inter[a_leaf]:.2f} cm, "
          f"spacing {intra[a_leaf]:.2f} cm")
    print(f"persistence tips found: {len(tips)} vs {len(real)} true leaves")


if __name__ == "__main__":
    main()
