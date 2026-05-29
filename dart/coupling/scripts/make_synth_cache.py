"""Pre-generate labelled occluded synthetic maize clouds for fast segmenter
iteration. grow_plant + loft + sample + occlude is the expensive part (~20 s/
plant); the segmenter is milliseconds. Cache once, then iterate
segment-and-score against the cache in seconds.

Mirrors eval_segmenter_synthetic.py's fill=none path EXACTLY (same per-plant
RNG: seed = seed0 + i, same day draw) so a cached run == a live --fill none run.

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/make_synth_cache.py --n 24 --out dart/coupling/output/synth_cache
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
sys.path.insert(0, "dart/coupling/scripts")
import eval_segmenter_synthetic as J
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--day_lo", type=int, default=40)
    ap.add_argument("--day_hi", type=int, default=92)
    ap.add_argument("--n_sample", type=int, default=16384)
    ap.add_argument("--xml", default=J.DEFAULT_XML)
    ap.add_argument("--out", default="dart/coupling/output/synth_cache")
    ap.add_argument("--complete", action="store_true",
                    help="fully-intact clouds (no occlusion masks) to probe the "
                         "occlusion-vs-model bottleneck; same seeds/days as occluded")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.n):
        seed = a.seed0 + i
        rng = np.random.default_rng(seed)
        day = int(rng.integers(a.day_lo, a.day_hi + 1))
        plant = grow_plant(a.xml, simulation_time=day, seed=seed)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, comp_lab = J.sample_labelled(mesh, a.n_sample, rng)
        if a.complete:
            pts, gt = J.complete_labelled(comp, comp_lab, rng)
        else:
            pts, gt = J.occlude_labelled(comp, comp_lab, rng)
        np.savez(os.path.join(a.out, f"cloud_{i:03d}.npz"),
                 pts=pts, gt=gt, seed=seed, day=day)
        n_gt = int(len([x for x in np.unique(gt) if x != 0]))
        print(f"[{i+1}/{a.n}] seed {seed:3d} d{day:3d} pts {len(pts):5d} GT {n_gt:2d}")
    print(f"wrote {a.n} clouds to {a.out}")


if __name__ == "__main__":
    main()
