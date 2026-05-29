"""Feasibility probe: multi-date per-plant co-registration for occlusion infill.

FP4D is a time series (Plot04: 5 dates, Plot03: 9 dates; consecutive gaps 2-7 d).
A single scan self-occludes ~2-3 basal leaves. Idea: register near-in-time scans
of the SAME plant (rooted -> base-anchored) and ACCUMULATE -> a denser, more
complete cloud with real points, recovering occluded structure. Over a 2-day gap
the plant is near-stable, so rigid registration is valid (no growth blur).

This probe takes Plot04 230619->230621 (2-day gap), crops the SAME plant from
both, ICP-refines (both are base-centred by crop_plant_window so they start
coarsely aligned), accumulates, and tests:
  * how much the plant grew in 2 days (height / extent / point-count delta)
  * registration RMSE
  * does the union FILL occlusion (point gain, hole reduction)?
  * does segmentation recover MORE leaves on the accumulated cloud?

    cd /home/lukas/PHD/CPlantBox
    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/probe_multidate_coreg.py
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
from loader import load_las, separate_plants_along_row, crop_plant_window
import mongraphseg_graph as mg

FP4D = "/home/lukas/PHD/Resources/PHENOROAM DATA ASSIMILATION May 2026/doi-10.60507-fk2-hyi2ds"
PLOT = "Plot04"
DATE_A, DATE_B = "230619", "230621"   # 2-day gap; B is the whole_plant.npy date


def load_row(date):
    return load_las(f"{FP4D}/{PLOT}/{date}.las", height_lo_m=0.15, voxel_m=0.005)


def icp(src, dst, iters=40, reject_pct=80):
    """Point-to-point ICP (Kabsch), trimmed. Returns (aligned_src, rmse)."""
    s = src.copy()
    tree = cKDTree(dst)
    for _ in range(iters):
        d, idx = tree.query(s)
        thr = np.percentile(d, reject_pct)
        m = d <= thr
        A, B = s[m], dst[idx[m]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = cb - R @ ca
        s = (R @ s.T).T + t
    d, _ = cKDTree(dst).query(s)
    return s, float(np.sqrt((d ** 2).mean()))


def voxel(p, v=0.3):
    mn = p.min(0); ijk = np.floor((p - mn) / v).astype(np.int64)
    _, u = np.unique(ijk, axis=0, return_index=True)
    return p[u]


def desc(p, tag):
    h = np.ptp(p[:, 2])
    print(f"  {tag:18s} n={len(p):6d} h={h:5.1f}cm "
          f"xext={np.ptp(p[:,0]):5.1f} yext={np.ptp(p[:,1]):5.1f}")
    return h


def n_leaves(p):
    org = mg.segment_plant_pseudostem(p, n_skel_nodes=400)
    return sum(1 for k in org if k.startswith("leaf") and len(org[k]) > 0)


def main():
    print(f"=== load {PLOT} {DATE_A} + {DATE_B} (2-day gap) ===")
    row_b = load_row(DATE_B)
    ax_b, cen_b = separate_plants_along_row(row_b)
    row_a = load_row(DATE_A)
    ax_a, cen_a = separate_plants_along_row(row_a)
    print(f"  row_axis a/b = {ax_a}/{ax_b}; plants a/b = {len(cen_a)}/{len(cen_b)}")

    # choose a well-captured plant in B: tallest crop among a few mid-row candidates
    best = None
    for k in range(len(cen_b)):
        cb = crop_plant_window(row_b, ax_b, cen_b[k], window_cm=20.0,
                               cross_row_window_cm=25.0)
        if len(cb) < 1500:
            continue
        h = np.ptp(cb[:, 2]) if len(cb) else 0
        if best is None or h > best[2]:
            best = (k, cen_b[k], h, cb)
    if best is None:
        print("  no good plant found"); return
    k, centre, hb, crop_b = best
    print(f"  chosen plant k={k} centre={centre:.1f} on row_axis {ax_b}")

    # same plant in A = nearest centre to `centre` along the row axis
    j = int(np.argmin(np.abs(np.asarray(cen_a) - centre)))
    crop_a = crop_plant_window(row_a, ax_a, cen_a[j], window_cm=20.0,
                               cross_row_window_cm=25.0)
    print(f"  matched A plant j={j} centre={cen_a[j]:.1f} (Δcentre {abs(cen_a[j]-centre):.1f}cm)")

    print("\n=== crops (both base-centred) ===")
    ha = desc(crop_a, f"{DATE_A} (earlier)")
    hb = desc(crop_b, f"{DATE_B} (later)")
    print(f"  2-day growth: Δheight {hb-ha:+.1f}cm, Δpts {len(crop_b)-len(crop_a):+d}")

    print("\n=== ICP register A -> B ===")
    reg_a, rmse = icp(crop_a, crop_b)
    print(f"  ICP RMSE {rmse:.2f}cm")

    # accumulate and measure occlusion infill
    acc = voxel(np.vstack([crop_b, reg_a]), v=0.3)
    # novel points from A not present in B (filled holes)
    d, _ = cKDTree(crop_b).query(reg_a)
    novel = int((d > 1.0).sum())
    print("\n=== accumulation ===")
    desc(crop_b, "B alone")
    desc(acc, "B + reg(A)")
    print(f"  points from A novel vs B (>1cm) = {novel} "
          f"({100*novel/max(len(reg_a),1):.0f}% of A) -> occlusion infill")

    print("\n=== segmentation: single vs accumulated ===")
    nb = n_leaves(crop_b)
    nacc = n_leaves(acc)
    print(f"  leaves: B-alone {nb}  |  B+reg(A) {nacc}  (Δ {nacc-nb:+d})")

    # save + render
    np.save("/home/lukas/pointr/organs/coreg_B.npy", crop_b.astype(np.float32))
    np.save("/home/lukas/pointr/organs/coreg_accum.npy", acc.astype(np.float32))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    ax[0].scatter(crop_b[:, 1], crop_b[:, 2], s=2, c="#1f77b4")
    ax[0].set_title(f"B {DATE_B} alone\n{len(crop_b)}pts, {nb} leaves")
    ax[1].scatter(reg_a[:, 1], reg_a[:, 2], s=2, c="#d62728")
    ax[1].set_title(f"A {DATE_A} registered\nRMSE {rmse:.2f}cm")
    ax[2].scatter(acc[:, 1], acc[:, 2], s=2, c="#2ca02c")
    ax[2].set_title(f"B + reg(A) accumulated\n{len(acc)}pts, {nacc} leaves")
    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("y (cm)")
    ax[0].set_ylabel("z (cm)")
    fig.suptitle(f"Multi-date co-registration {PLOT} {DATE_A}->{DATE_B} (2-day, occlusion infill)")
    fig.tight_layout()
    out = "/home/lukas/pointr/organs/coreg_probe.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  render -> {out}")


if __name__ == "__main__":
    main()
