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

from reconstruct_detailed import write_obj   # coherent clock reconstruction, height-corrected to scans

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(os.environ.get(
    "COUPLING_VR_OUTDIR",
    PROJECT_ROOT / "dart" / "coupling" / "output" / "vr_stages_stagelib"))

# developmental clocks: integer leaf-count stages within plant_01 coverage
# (clk<=12) + the continuous morph into MF3D maturity (12->15).
CLOCKS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12.75, 13.5, 14.25, 15]


def gen_one(clock: float) -> dict | None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{clock:05.2f}".replace(".", "p")
    out_path = OUT_DIR / f"maize_clk{tag}.obj"
    info = write_obj(clock, str(out_path))
    dt = time.time() - t0
    print(f"  clk {clock:5.2f}  leaves={info['n']:>2}  m={info['m']}  "
          f"top={info['top']}cm  {dt:.1f}s  {out_path.name}")
    return info


def main() -> int:
    print(f"OUT_DIR = {OUT_DIR}")
    for old in list(OUT_DIR.glob("maize_clk23*.obj")):   # drop stale date-named OBJs
        old.unlink()
        old.with_suffix(".mtl").unlink(missing_ok=True)
    for clock in CLOCKS:
        try:
            gen_one(clock)
        except Exception as exc:  # noqa: BLE001
            print(f"  clk {clock} FAILED: {exc!r}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
