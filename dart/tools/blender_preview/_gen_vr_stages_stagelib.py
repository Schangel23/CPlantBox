"""Generate VR stage OBJs from the plant_01<->MF3D FAITHFUL reconstruction.

Supersedes the grow+donor-swap path (which produced 2x-too-big / uniform /
gap-stem plants). Each stage is built directly from plant_01's MEASURED geometry
within coverage (== the NURBS point cloud) and grows continuously into the MF3D
mature-stature anchor beyond it -- one continuously-developing plant, keyed on
developmental clock (leaf count). See pointr/reconstruct_blended.py.

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

os.chdir("/home/lukas/PHD/CPlantBox")
sys.path.insert(0, "/home/lukas/pointr")

from reconstruct_detailed import write_obj_date  # RAW per-date cloud NURBS (fit_date) -> production lofter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(os.environ.get(
    "COUPLING_VR_OUTDIR",
    PROJECT_ROOT / "dart" / "coupling" / "output" / "vr_stages_stagelib"))

# VR stages = the actual plant_01 scan dates. Each lofts the RAW assimilated
# cloud NURBS (fit_date) so height + nadir + azimuth match the scan exactly
# (vs the old clock-interpolated reconstruct_blended that compressed height).
DATES = ["230606", "230613", "230619", "230621",
         "230625", "230629", "230705", "230711"]


def gen_one(date: str) -> dict | None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # name maize_clk<date>.obj so _build_vr_blend's maize_clk* glob picks them up
    # and chronological date strings sort into developmental order.
    out_path = OUT_DIR / f"maize_clk{date}.obj"
    info = write_obj_date(date, str(out_path))
    dt = time.time() - t0
    print(f"  {date}  leaves={info['n']:>2}  top={info['top']}cm  "
          f"{dt:.1f}s  {out_path.name}")
    return info


def main() -> int:
    print(f"OUT_DIR = {OUT_DIR}")
    # clear stale clock-named OBJs so the row is only the date stages
    for old in OUT_DIR.glob("maize_clk*p*.obj"):
        old.unlink()
        old.with_suffix(".mtl").unlink(missing_ok=True)
    for date in DATES:
        try:
            gen_one(date)
        except Exception as exc:  # noqa: BLE001
            print(f"  {date} FAILED: {exc!r}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
