"""GATE A: does the monocot phyllotaxy + ribbon prior actually separate
touching maize blades?  No ML, no segmenter -- pure geometry on the LABELLED
cache, so we can compute an ORACLE CEILING before writing any production code.

Pipeline under test (intrinsic, axis-relative):
  axis (c, a) = PCA of the non-leaf core (bottom 60% by height, tassel-trimmed)
  per point:  h = (p-c).a   r = |w|   phi = atan2(w.v_hat, w.u_hat)   w = p-c-h*a
  per GT leaf i: collar (h_i, phi_i) = median of its lowest-r 15% points.

Two assignment ceilings (oracle collars from GT, so this bounds what the
unsupervised --attach path can reach):

  T1 raw   : argmin_i  alpha*angdist(phi_p, phi_i) + beta*|h_p - h_i|
             -- expected to FAIL on drooping distal points (an upper leaf's
                tip arcs back DOWN into a lower same-azimuth leaf's h-band).
  T2 geo   : azimuth-GATED geodesic from each collar seed (cut kNN edges whose
             endpoints straddle Delta-phi > tau).  This is the proposed method's
             ceiling -- flow cannot leak across a touching contact into a blade
             of different azimuth.

Reported per cache: recall@.5 = sum(IoU>=.5)/sum(n_gt)  (same metric as the
59% baseline) for T1 and T2, plus collar count-err.  A 4-panel (h,phi)/(phi,r)
plot per plant shows whether GT leaves are disjoint strips.

SUCCESS GATE: T2 recall@.5 >= 85% intact.  If not, the prior is insufficient
as stated -- fix the prior (e.g. add ribbon-monotonicity tiebreak) before
asking Codex to build the unsupervised collar detector.

    PYTHONPATH=. cpbenv/bin/python dart/coupling/scripts/viz_cylindrical_unwrap.py \
        --cache dart/coupling/output/synth_cache_complete
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_axis(pts, nonleaf):
    """PCA axis of the non-leaf core, tassel-trimmed (bottom 60% by z)."""
    nl = pts[nonleaf]
    if len(nl) < 10:
        nl = pts
    zhi = np.quantile(nl[:, 2], 0.6)
    core = nl[nl[:, 2] <= zhi]
    if len(core) < 10:
        core = nl
    c = core.mean(0)
    _, _, vt = np.linalg.svd(core - c, full_matrices=False)
    a = vt[0]
    if a[2] < 0:
        a = -a
    tilt = np.degrees(np.arccos(np.clip(abs(a[2]), 0, 1)))
    return c, a, tilt


def cylindrical(pts, c, a):
    d = pts - c
    h = d @ a
    w = d - np.outer(h, a)
    r = np.linalg.norm(w, axis=1)
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = tmp - (tmp @ a) * a
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    phi = np.arctan2(w @ v, w @ u)
    return h, r, phi, u, v


def angdist(p, q):
    """circular distance in radians."""
    return np.abs(np.angle(np.exp(1j * (p - q))))


def circmean(phi):
    return float(np.angle(np.mean(np.exp(1j * phi))))


def collars(h, r, phi, gt, frac=0.15):
    """oracle (h_i, phi_i) per GT leaf from its lowest-r `frac` points."""
    out = {}
    for g in np.unique(gt):
        if g == 0:
            continue
        m = gt == g
        rg = r[m]
        thr = np.quantile(rg, frac)
        sel = m.copy()
        sel[m] = rg <= thr
        out[int(g)] = (float(np.median(h[sel])), circmean(phi[sel]))
    return out


def recall_at_50(gt, pred):
    """sum(IoU>=.5)/n_gt over Hungarian-matched leaf instances."""
    gids = [i for i in np.unique(gt) if i != 0]
    pids = [i for i in np.unique(pred) if i != 0]
    if not gids:
        return 0, 0
    if not pids:
        return 0, len(gids)
    iou = np.zeros((len(gids), len(pids)))
    gm = {g: gt == g for g in gids}
    pm = {p: pred == p for p in pids}
    for a_, g in enumerate(gids):
        for b_, p in enumerate(pids):
            inter = np.count_nonzero(gm[g] & pm[p])
            if inter:
                iou[a_, b_] = inter / np.count_nonzero(gm[g] | pm[p])
    r, c = linear_sum_assignment(-iou)
    return int(np.count_nonzero(iou[r, c] >= 0.5)), len(gids)


def assign_t1(h, phi, leaf_mask, col, alpha, beta):
    ids = sorted(col)
    H = np.array([col[i][0] for i in ids])
    P = np.array([col[i][1] for i in ids])
    pred = np.zeros(len(h), np.int64)
    idx = np.where(leaf_mask)[0]
    cost = (alpha * angdist(phi[idx][:, None], P[None, :])
            + beta * np.abs(h[idx][:, None] - H[None, :]))
    pred[idx] = np.array(ids)[np.argmin(cost, axis=1)]
    return pred


def assign_t2(pts, h, phi, r, leaf_mask, col, k, tau_deg, max_edge):
    """azimuth-gated geodesic from oracle collar seeds."""
    idx = np.where(leaf_mask)[0]
    P = pts[idx]
    n = len(P)
    tree = cKDTree(P)
    dist, nbr = tree.query(P, k=min(k + 1, n))
    tau = np.radians(tau_deg)
    phL = phi[idx]
    rows, cols, w = [], [], []
    for i in range(n):
        for jj in range(1, nbr.shape[1]):
            j = nbr[i, jj]
            d = dist[i, jj]
            if d > max_edge:
                continue
            if angdist(phL[i], phL[j]) > tau:      # gate: don't cross azimuth
                continue
            rows.append(i); cols.append(j); w.append(d)
    G = csr_matrix((w, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    # seed = the lowest-r point of each leaf (its collar), mapped into idx-space
    ids = sorted(col)
    seeds, seed_leaf = [], []
    # collar seed = nearest leaf point to (h_i, phi_i) in (h,phi) space
    Hh, Pp = h[idx], phi[idx]
    for g in ids:
        hi, pi = col[g]
        dd = (Hh - hi) ** 2 + angdist(Pp, pi) ** 2
        seeds.append(int(np.argmin(dd)))
        seed_leaf.append(g)
    D = dijkstra(G, directed=False, indices=seeds)   # (nseeds, n)
    nearest = np.argmin(D, axis=0)
    reach = np.isfinite(D[nearest, np.arange(n)])
    pred = np.zeros(len(h), np.int64)
    lab = np.array(seed_leaf)[nearest]
    lab[~reach] = 0
    pred[idx] = lab
    return pred


def leaf_azimuths(phi, r, gt, rmin_frac=0.5):
    """per-leaf azimuth from its HIGH-r points (stable phi), oracle from GT."""
    out = {}
    for g in np.unique(gt):
        if g == 0:
            continue
        m = gt == g
        rg = r[m]
        thr = np.quantile(rg, rmin_frac)        # distal half: phi is stable
        sel = m.copy(); sel[m] = rg >= thr
        out[int(g)] = circmean(phi[sel])
    return out


def assign_t3(phi, leaf_mask, az):
    """azimuth-ONLY: nearest leaf full-blade azimuth (no height term)."""
    ids = sorted(az); P = np.array([az[i] for i in ids])
    pred = np.zeros(len(phi), np.int64)
    idx = np.where(leaf_mask)[0]
    pred[idx] = np.array(ids)[np.argmin(angdist(phi[idx][:, None], P[None, :]), axis=1)]
    return pred


def assign_t4(h, phi, r, leaf_mask, az, col, col_deg=25.0):
    """azimuth nearest; same-azimuth columns (Dphi<col_deg) split by collar h.
    A point joins the column-leaf with the largest collar h_i that is still
    <= the point's own height projected up the spine (droop-aware: use r-lifted
    height proxy h + 0 -- keep simple: nearest collar h)."""
    ids = sorted(az)
    P = np.array([az[i] for i in ids])
    idx = np.where(leaf_mask)[0]
    col_to = np.argmin(angdist(phi[idx][:, None], P[None, :]), axis=1)
    # build azimuth columns
    order = np.argsort(P)
    colid = np.zeros(len(ids), int); cc = 0
    for k in range(1, len(order)):
        if angdist(P[order[k]], P[order[k - 1]]) > np.radians(col_deg):
            cc += 1
        colid[order[k]] = cc
    colid[order[0]] = 0
    Hc = np.array([col[i][0] for i in ids])
    pred = np.zeros(len(h), np.int64)
    hL = h[idx]
    for pi in range(len(idx)):
        ci = colid[col_to[pi]]
        members = [j for j in range(len(ids)) if colid[j] == ci]
        if len(members) == 1:
            pred[idx[pi]] = ids[members[0]]
        else:
            # nearest collar height among same-azimuth leaves
            mh = np.array([Hc[j] for j in members])
            pred[idx[pi]] = ids[members[int(np.argmin(np.abs(mh - hL[pi])))]]
    return pred


def fourpanel(h, r, phi, gt, path, title):
    leaf = gt != 0
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    sc = ax[0].scatter(np.degrees(phi[leaf]), h[leaf], c=gt[leaf], s=2,
                       cmap="tab20")
    ax[0].set_xlabel("phi (deg)"); ax[0].set_ylabel("h (cm)")
    ax[0].set_title("(h, phi) -- strips should be disjoint")
    ax[1].scatter(np.degrees(phi[leaf]), r[leaf], c=gt[leaf], s=2, cmap="tab20")
    ax[1].set_xlabel("phi (deg)"); ax[1].set_ylabel("r (cm)")
    ax[1].set_title("(phi, r)")
    ax[2].scatter(np.degrees(phi[~leaf]), h[~leaf], c="0.6", s=2)
    ax[2].set_xlabel("phi (deg)"); ax[2].set_ylabel("h (cm)")
    ax[2].set_title("non-leaf (axis core)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="dart/coupling/output/synth_cache_complete")
    ap.add_argument("--r0", type=float, default=2.0, help="collar band (unused in oracle, kept for parity)")
    ap.add_argument("--tau", type=float, default=50.0, help="azimuth gate deg")
    ap.add_argument("--alpha", type=float, default=20.0, help="T1 azimuth weight")
    ap.add_argument("--beta", type=float, default=1.0, help="T1 height weight")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max_edge", type=float, default=3.0)
    ap.add_argument("--collar_frac", type=float, default=0.15)
    ap.add_argument("--nplots", type=int, default=6)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.cache, "cloud_*.npz")))
    outdir = os.path.join(a.cache, "_unwrap_viz")
    os.makedirs(outdir, exist_ok=True)
    g1t = g1g = g2t = g2g = g3g = g4g = 0
    cerr = []
    print(f"=== GATE A unwrap | cache {a.cache} | {len(files)} plants "
          f"| tau={a.tau} alpha={a.alpha} beta={a.beta} ===")
    for n, f in enumerate(files):
        d = np.load(f)
        pts, gt = d["pts"], d["gt"]
        c, ax_, tilt = fit_axis(pts, gt == 0)
        h, r, phi, _, _ = cylindrical(pts, c, ax_)
        leaf_mask = gt != 0
        col = collars(h, r, phi, gt, a.collar_frac)
        cerr.append(len(col) - len([i for i in np.unique(gt) if i != 0]))
        az = leaf_azimuths(phi, r, gt)
        p1 = assign_t1(h, phi, leaf_mask, col, a.alpha, a.beta)
        p2 = assign_t2(pts, h, phi, r, leaf_mask, col, a.k, a.tau, a.max_edge)
        p3 = assign_t3(phi, leaf_mask, az)
        p4 = assign_t4(h, phi, r, leaf_mask, az, col)
        a1, gtot = recall_at_50(gt, p1)
        a2, _ = recall_at_50(gt, p2)
        a3, _ = recall_at_50(gt, p3)
        a4, _ = recall_at_50(gt, p4)
        g1g += a1; g2g += a2; g3g += a3; g4g += a4; g1t += gtot; g2t += gtot
        print(f"[{n+1:2d}/{len(files)}] seed{int(d['seed']):3d} d{int(d['day']):3d} "
              f"tilt{tilt:4.1f} L={gtot:2d} | T1 {a1:2d} T2 {a2:2d} "
              f"T3az {a3:2d} T4az+h {a4:2d} /{gtot:2d}")
        if n < a.nplots:
            fourpanel(h, r, phi, gt, os.path.join(outdir, f"unwrap_{n:02d}.png"),
                      f"{os.path.basename(f)} tilt={tilt:.1f}deg L={gtot}")
    print(f"\n--- ORACLE CEILINGS (intact) ---")
    print(f"T1 raw (h,phi)        recall@.5 {g1g}/{g1t} ({100*g1g/max(g1t,1):.0f}%)")
    print(f"T2 gated-geodesic     recall@.5 {g2g}/{g2t} ({100*g2g/max(g2t,1):.0f}%)")
    print(f"T3 azimuth-only       recall@.5 {g3g}/{g2t} ({100*g3g/max(g2t,1):.0f}%)")
    print(f"T4 azimuth+h-split    recall@.5 {g4g}/{g2t} ({100*g4g/max(g2t,1):.0f}%)  <-- prior ceiling")
    print(f"collar count-err      mean {np.mean(cerr):+.2f} abs {np.mean(np.abs(cerr)):.2f}")
    print(f"plots -> {outdir}/unwrap_*.png")


if __name__ == "__main__":
    main()
