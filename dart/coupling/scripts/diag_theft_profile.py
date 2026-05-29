"""Where along each GT blade are the stolen points?

37% of GT-blade points get assigned to the pseudostem. Is that theft
concentrated in the PROXIMAL blade (insertion truncation -> fixable by lowering
the insertion) or spread along the whole blade (assignment failure)?

For every GT leaf we project its points onto the leaf's own principal axis
(proximal=0 .. distal=1) and report, in 5 bins along that axis, the fraction of
points stolen by the pseudostem. A monotone front-loaded profile (high theft at
bin 0, ~0 at bin 4) means pure insertion truncation.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/diag_theft_profile.py --n 6
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

NBIN = 5


def pred_full(organs, pts):
    pred = np.zeros(len(pts), dtype=np.int64)
    tree = cKDTree(pts)
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
    a = ap.parse_args()

    bins_theft = np.zeros(NBIN)   # stolen-by-pseudostem count per bin
    bins_tot = np.zeros(NBIN)     # total GT points per bin
    # also: proximal-blade height. measure stolen pts' z vs leaf z-range
    n_leaves = 0
    for i in range(a.n):
        rng = np.random.default_rng(i)
        day = int(rng.integers(40, 93))
        plant = grow_plant(J.DEFAULT_XML, simulation_time=day, seed=i)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, clab = J.sample_labelled(mesh, 16384, rng)
        pts, gt = J.occlude_labelled(comp, clab, rng)
        organs = mg.segment_plant_pseudostem(pts, n_skel_nodes=400, assign=a.assign)
        pred = pred_full(organs, pts)
        for g in [x for x in np.unique(gt) if x != 0]:
            gm = gt == g
            if int(gm.sum()) < 20:
                continue
            n_leaves += 1
            P = pts[gm]
            # principal axis of the blade: proximal = end nearest the plant axis (x,y)~0?
            # robust: PCA, orient so axis points away from the cloud centroid-to-base.
            c = P.mean(0)
            U = P - c
            _, _, vt = np.linalg.svd(U, full_matrices=False)
            axis = vt[0]
            t = U @ axis
            # orient proximal->distal: proximal end is closer to plant vertical axis
            # (min horizontal radius). pick the end with smaller mean radius.
            r = np.linalg.norm(P[:, :2], axis=1)
            lo_end = t < np.median(t)
            if r[lo_end].mean() > r[~lo_end].mean():
                t = -t  # flip so small t = proximal (near stem)
            tn = (t - t.min()) / (np.ptp(t) + 1e-9)
            stolen = pred[gm] == -1
            idx = np.clip((tn * NBIN).astype(int), 0, NBIN - 1)
            for b in range(NBIN):
                m = idx == b
                bins_tot[b] += int(m.sum())
                bins_theft[b] += int((m & stolen).sum())

    frac = bins_theft / np.maximum(bins_tot, 1)
    print(f"\n=== theft profile along blade (assign={a.assign}, {n_leaves} leaves) ===")
    print("bin (proximal->distal) | frac stolen by pseudostem | n_pts")
    for b in range(NBIN):
        bar = "#" * int(frac[b] * 40)
        print(f"  {b}  [{b/NBIN:.1f}-{(b+1)/NBIN:.1f}]   {frac[b]*100:5.1f}%  {bar:<40} ({int(bins_tot[b])})")
    print(f"\noverall theft = {bins_theft.sum()/max(bins_tot.sum(),1)*100:.1f}%")


if __name__ == "__main__":
    main()
