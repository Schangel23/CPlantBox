"""Before/after visual for the basal-pseudostem anti-theft fix.

Renders, side by side on the same synthetic occluded maize cloud:
  (left)  assign=segment, pseudostem = full bundle column  (baseline, theft)
  (right) assign=segment, pseudostem = basal-only           (anti-theft)
Points coloured by predicted instance (pseudostem grey, leaves by tab20).
Also runs the segmenter on the real FP4D cloud and reports leaf count.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/viz_basal_pseudostem_2026-05-29.py --seed 3
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
import mongraphseg_graph as mg
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


def labels_from_organs(organs, pts):
    """0 = pseudostem, 1..L = leaves (nearest-organ-point vote)."""
    from scipy.spatial import cKDTree
    lab = np.zeros(len(pts), dtype=int)
    tree = cKDTree(pts)
    if "pseudostem" in organs and len(organs["pseudostem"]):
        lab[tree.query(organs["pseudostem"])[1]] = 0
    lid = 0
    for k in sorted(organs):
        if k.startswith("leaf") and len(organs[k]):
            lid += 1
            lab[tree.query(organs[k])[1]] = lid
    return lab


def panel(ax, pts, lab, title):
    cmap = plt.get_cmap("tab20")
    ps = lab == 0
    ax.scatter(pts[ps, 0], pts[ps, 2], s=2, c="0.7", label="pseudostem")
    for l in [x for x in np.unique(lab) if x > 0]:
        m = lab == l
        ax.scatter(pts[m, 0], pts[m, 2], s=2, color=cmap((l - 1) % 20))
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="dart/coupling/output/basal_pseudostem_beforeafter.png")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    day = int(rng.integers(60, 90))
    plant = grow_plant(J.DEFAULT_XML, simulation_time=day, seed=a.seed)
    mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
    comp, clab = J.sample_labelled(mesh, 16384, rng)
    pts, gt = J.occlude_labelled(comp, clab, rng)

    org_base = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment")
    org_basal = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign="segment",
                                            pseudostem_basal_only=True)
    lab_base = labels_from_organs(org_base, pts)
    lab_basal = labels_from_organs(org_basal, pts)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    # GT reference
    cmap = plt.get_cmap("tab20")
    axes[0].scatter(pts[gt == 0, 0], pts[gt == 0, 2], s=2, c="0.7")
    for l in [x for x in np.unique(gt) if x > 0]:
        m = gt == l
        axes[0].scatter(pts[m, 0], pts[m, 2], s=2, color=cmap((l - 1) % 20))
    axes[0].set_title(f"GT (day {day}, {int((gt>0).sum())} leaf pts)", fontsize=10)
    axes[0].set_aspect("equal"); axes[0].set_xlabel("x (cm)"); axes[0].set_ylabel("z (cm)")
    panel(axes[1], pts, lab_base, "baseline: full pseudostem (theft)")
    panel(axes[2], pts, lab_basal, "fix: basal-only pseudostem")
    fig.suptitle(f"MonGraphSeg pseudostem-theft fix  (seed {a.seed})")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")

    # ── real FP4D cloud ──
    real_path = "/home/lukas/pointr/organs/whole_plant.npy"
    try:
        rpts = np.load(real_path)
        if rpts.ndim == 2 and rpts.shape[1] >= 3:
            rpts = rpts[:, :3].astype(float)
            ro_base = mg.segment_plant_pseudostem(rpts, n_skel_nodes=400, assign="segment")
            ro_basal = mg.segment_plant_pseudostem(rpts, n_skel_nodes=400, assign="segment",
                                                   pseudostem_basal_only=True)
            nb = sum(1 for k in ro_base if k.startswith("leaf"))
            na = sum(1 for k in ro_basal if k.startswith("leaf"))
            print(f"real cloud {real_path}: {len(rpts)} pts | "
                  f"leaves baseline={nb} basal-only={na}")
        else:
            print(f"real cloud has unexpected shape {rpts.shape}; skipped")
    except Exception as e:
        print(f"real cloud run failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
