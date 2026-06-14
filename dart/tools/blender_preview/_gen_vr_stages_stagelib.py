"""Like _gen_vr_stages.py, but with plant_01 STAGE-CORRELATED donor swap.

Supersedes _gen_vr_stages_staged.py (whole-plant alpha blend toward ONE young
donor). Here every leaf wears plant_01's *real* blade shape at its own
developmental fraction, interpolated between plant_01's labelled dates, via
``stage_donor_swap.apply_stage_donors`` + the per-rank ``plant01_stage_library``.
Lower mature leaves get broad mature blades, upper emerging leaves get the
youngest measured shapes, the whole row transitions smoothly with stage.

Run:
    cd /home/lukas/PHD/CPlantBox
    COUPLING_VR_OUTDIR=dart/coupling/output/vr_stages_stagelib \
      PYTHONPATH=.:/home/lukas/pointr cpbenv/bin/python \
      dart/tools/blender_preview/_gen_vr_stages_stagelib.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# grow_plant resolves shape_distribution_path relative to CWD.
os.chdir("/home/lukas/PHD/CPlantBox")
sys.path.insert(0, "/home/lukas/pointr")

from dart.coupling.config import DATA_DIR
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.geometry.cplantbox_adapter import (
    get_plantsim_feature_kwargs_from_env,
)
from dart.coupling.growth.grow import grow_plant
from dart.coupling.growth.phenology import count_visible_leaves, detect_v_stage

from dart.tools.blender_preview._gen_vr_stages import DAYS, SEED
from stage_donor_swap import apply_stage_donors

# plant_01 (Mirza) cultivar: the *production trajectory* itself carries the FP4D
# internode kinetics (per-rank t_col blend, S3), and apply_stage_donors swaps in
# the FP4D blade shape per rank/stage (S1/S2 coverage-weighted ribbon blend).
# We deliberately do NOT apply the pose/collar overrides (apply_plant01_cultivar):
# they clump leaves and mangle the architecture. Keep the real growth, modify
# only the blade shape (+ kinetics baked in the XML).
CULTIVAR_XML = DATA_DIR / "maize_mirza_plant01.xml"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(os.environ.get(
    "COUPLING_VR_OUTDIR",
    PROJECT_ROOT / "dart" / "coupling" / "output" / "vr_stages_stagelib"))


def grow_one(day: int) -> dict | None:
    t0 = time.time()
    print(f"\n=== Day {day:3d} (seed {SEED}) STAGE-LIB ===")
    plant = grow_plant(str(CULTIVAR_XML), simulation_time=day, seed=SEED,
                       enable_photosynthesis=False)
    label = detect_v_stage(plant)
    counts = count_visible_leaves(plant)
    out_path = OUT_DIR / f"maize_day{day:03d}_{label}.obj"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    organs = extract_organs_for_lofter(
        plant, species="maize", skip_roots=True,
        **get_plantsim_feature_kwargs_from_env(),
    )
    # Blade-shape blend ONLY (coverage-weighted FP4D<->MF3D ribbon blend). The
    # stem/collars come from the real grown plant (FP4D internode kinetics baked
    # into the XML); no pose/collar override.
    rep = apply_stage_donors(organs, vstage=int(counts["collared"]))
    nsw = len(rep)
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
    mesh.to_obj(str(out_path), group_by_organ=True, write_materials=True)
    dt = time.time() - t0
    print(f"  {label:16s} leaves={counts['total']:>2}  swapped={nsw}/{len(rep)}  "
          f"tris={mesh.n_triangles}  {dt:.1f}s  {out_path.name}")
    return {"day": day, "label": label, "swapped": nsw}


def main() -> int:
    print(f"OUT_DIR = {OUT_DIR}")
    for day in DAYS:
        try:
            grow_one(day)
        except Exception as exc:  # noqa: BLE001
            print(f"  day {day} FAILED: {exc!r}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
