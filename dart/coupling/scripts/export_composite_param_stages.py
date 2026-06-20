"""Export FP4D-intercept vs current-parametric CP previews for maize stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import plantbox as pb

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dart.coupling.growth.grow import grow_plant


DATA = Path("dart/coupling/data")
DEFAULT_XML = DATA / "maize_calibrated.xml"
DEFAULT_DIST = DATA / "plant01_leaf_shape_dist.json"
DEFAULT_OUT = Path("dart/coupling/output/parametric_stage_validation")
STAGES = {"V1": 12.0, "V3": 26.0, "V7": 58.0, "V9": 78.0, "V13": 112.0}


def _points(cps) -> list[list[float]]:
    return [[float(p.x), float(p.y), float(p.z)] for p in cps]


def _reference_rank(dist, rank: int, curvature_scale: float) -> list[list[float]]:
    shape = dist.makeShape(rank, 0.0, 0, curvature_scale)
    return _points(shape.sampleCanonicalGrid(dist.numCpsU(), dist.numCpsV(), 1.0, 1.0))


def export_stages(xml: Path, dist_json: Path, out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dist = pb.LeafShapeDistribution.load(str(dist_json))
    manifest = {
        "xml": str(xml),
        "distribution": str(dist_json),
        "seed": seed,
        "stages": [],
    }

    for stage, day in STAGES.items():
        plant = grow_plant(str(xml), simulation_time=day, seed=seed)
        leaves = [o for o in plant.getOrgans() if o.organType() == 4 and o.getNumberOfNodes() >= 3]
        leaves.sort(key=lambda o: o.getId())
        exported = []
        for rank, leaf in enumerate(leaves[: dist.numRanks()]):
            cps = leaf.getEffectiveSurfaceCPs()
            if not cps:
                continue
            try:
                lrp = plant.getOrganRandomParameter(4, rank + 2)
                curvature_scale = float(lrp.midrib_curvature_scale)
            except Exception:
                curvature_scale = 1.0
            exported.append(
                {
                    "rank": rank,
                    "leaf_id": int(leaf.getId()),
                    "current": _points(cps),
                    "reference": _reference_rank(dist, rank, curvature_scale),
                }
            )
        payload = {"stage": stage, "day": day, "leaves": exported}
        path = out_dir / f"{stage}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        manifest["stages"].append({"stage": stage, "day": day, "path": str(path), "leaves": len(exported)})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = export_stages(args.xml, args.distribution, args.output, args.seed)
    print(manifest)


if __name__ == "__main__":
    main()
