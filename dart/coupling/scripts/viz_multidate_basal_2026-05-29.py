"""Run MonGraphSeg on the REAL multi-date accumulated FP4D clouds,
baseline (full pseudostem) vs basal-only, side by side.

These clouds are the 2-date lower-stem-anchored ICP accumulations (the
"2-timestep filling reconstruction"): denser input that improved coverage but
left segmentation recall capped. Here we check whether the basal-pseudostem
anti-theft fix changes the leaf decomposition on that denser real input.
No GT labels exist for real scans -> we report leaf count + a visual only.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/viz_multidate_basal_2026-05-29.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
import mongraphseg_graph as mg

CLOUDS = [
    ("Plot03", "/home/lukas/pointr/organs/multidate_Plot03_m15p0_m62p4_accum.npy"),
    ("Plot04", "/home/lukas/pointr/organs/multidate_Plot04_5p3_103p4_accum.npy"),
]


def labels_from_organs(organs, pts):
    lab = np.zeros(len(pts), dtype=int)
    tree = cKDTree(pts)
    if "pseudostem" in organs and len(organs["pseudostem"]):
        lab[tree.query(organs["pseudostem"])[1]] = 0
    lid = 0
    for k in sorted(organs):
        if k.startswith("leaf") and len(organs[k]):
            lid += 1
            lab[tree.query(organs[k])[1]] = lid
    return lab, lid


def panel(ax, pts, lab, title):
    cmap = plt.get_cmap("tab20")
    ps = lab == 0
    ax.scatter(pts[ps, 0], pts[ps, 2], s=2, c="0.7")
    for l in [x for x in np.unique(lab) if x > 0]:
        m = lab == l
        ax.scatter(pts[m, 0], pts[m, 2], s=2, color=cmap((l - 1) % 20))
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)")


def main():
    fig, axes = plt.subplots(len(CLOUDS), 2, figsize=(11, 5 * len(CLOUDS)))
    if len(CLOUDS) == 1:
        axes = axes[None, :]
    for r, (name, path) in enumerate(CLOUDS):
        pts = np.load(path)[:, :3].astype(float)
        org_base = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment")
        org_basal = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment",
                                                pseudostem_basal_only=True)
        lab_b, nb = labels_from_organs(org_base, pts)
        lab_a, na = labels_from_organs(org_basal, pts)
        ps_b = int((lab_b == 0).sum()); ps_a = int((lab_a == 0).sum())
        print(f"{name}: {len(pts)} pts | baseline {nb} leaves "
              f"(pseudostem {ps_b} pts) | basal-only {na} leaves (pseudostem {ps_a} pts)")
        panel(axes[r, 0], pts, lab_b, f"{name} baseline: {nb} leaves, ps={ps_b}")
        panel(axes[r, 1], pts, lab_a, f"{name} basal-only: {na} leaves, ps={ps_a}")
    fig.suptitle("MonGraphSeg on REAL multi-date accumulated FP4D clouds")
    fig.tight_layout()
    out = "dart/coupling/output/multidate_basal_pseudostem.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
