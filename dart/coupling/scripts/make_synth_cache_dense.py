"""Dense, low-jitter complete cache to test whether the touching-manifold floor
is a SAMPLING artifact (coarse voxel + 1mm jitter) vs physical leaf contact.
Matches eval seeds/days so it is comparable to synth_cache_complete.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/make_synth_cache_dense.py --n 8 \
        --n_sample 80000 --vox 0.05 --jitter 0.02 --out dart/coupling/output/synth_cache_dense
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


def dense_labelled(pts, lab, rng, vox, jitter):
    mn = pts.min(0); ijk = np.floor((pts - mn) / vox).astype(np.int64)
    _, u = np.unique(ijk, axis=0, return_index=True)
    pts, lab = pts[u], lab[u]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_seeds", type=int, default=None,
                    help="alias for --n, kept for spacing-sweep scripts")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--day_lo", type=int, default=40)
    ap.add_argument("--day_hi", type=int, default=92)
    ap.add_argument("--n_sample", type=int, default=80000)
    ap.add_argument("--vox", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--xml", default=J.DEFAULT_XML)
    ap.add_argument("--out", default="dart/coupling/output/synth_cache_dense")
    ap.add_argument("--output", default=None,
                    help="alias for --out")
    a = ap.parse_args()
    if a.n_seeds is not None:
        a.n = a.n_seeds
    if a.output is not None:
        a.out = a.output
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.n):
        seed = a.seed0 + i
        rng = np.random.default_rng(seed)
        day = int(rng.integers(a.day_lo, a.day_hi + 1))
        plant = grow_plant(a.xml, simulation_time=day, seed=seed)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, comp_lab = J.sample_labelled(mesh, a.n_sample, rng)
        pts, gt = dense_labelled(comp, comp_lab, rng, a.vox, a.jitter)
        np.savez(os.path.join(a.out, f"cloud_{i:03d}.npz"), pts=pts, gt=gt, seed=seed, day=day)
        n_gt = len([x for x in np.unique(gt) if x != 0])
        print(f"[{i+1}/{a.n}] seed {seed:3d} d{day:3d} pts {len(pts):6d} GT {n_gt:2d}")
    print(f"wrote {a.n} dense clouds to {a.out}  (vox={a.vox} jitter={a.jitter})")


if __name__ == "__main__":
    main()
