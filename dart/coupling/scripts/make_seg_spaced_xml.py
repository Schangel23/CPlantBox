"""Generate maize XML variants with larger inter-leaf spacing for segmentation.

The calibrated source XML is copied, then selected parameters are multiplied.
The source file is never modified.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"
OUT_DIR = REPO_ROOT / "dart/coupling/data"


def _params(organ):
    return {p.get("name"): p for p in organ.findall("parameter")}


def _mul_param(params, name, mult):
    p = params.get(name)
    if p is None or p.get("value") is None:
        return None
    old = float(p.get("value"))
    new = old * mult
    p.set("value", repr(new))
    return old, new


def _leaf_tri_area_by_organ(xml_path, seed=0, day=70):
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from dart.coupling.growth.grow import grow_plant
    from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
    from dart.coupling.geometry.g1_to_g3 import loft_organs

    plant = grow_plant(str(xml_path), simulation_time=day, seed=seed)
    mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
    tri = mesh.vertices[mesh.indices]
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ) * 0.5
    type_by_id = {int(m["organ_id"]): str(m.get("type", "")) for m in mesh.organ_meta}
    out = {}
    for oid in np.unique(mesh.organ_ids):
        oid = int(oid)
        if not type_by_id.get(oid, "").startswith("leaf"):
            continue
        out[oid] = float(area[mesh.organ_ids == oid].sum())
    return out


def _width_live(tree, width_mult):
    if width_mult == 1.0:
        return True, "width_mult=1.0"
    with tempfile.TemporaryDirectory(prefix="seg_width_live_") as td:
        base = Path(td) / "base.xml"
        test = Path(td) / "test.xml"
        tree.write(base, encoding="utf-8", xml_declaration=False)
        test_tree = ET.parse(base)
        for leaf in test_tree.getroot().findall("leaf"):
            params = _params(leaf)
            for name in ("Width_blade", "lmax", "areaMax"):
                _mul_param(params, name, width_mult)
        test_tree.write(test, encoding="utf-8", xml_declaration=False)
        a0 = _leaf_tri_area_by_organ(base)
        a1 = _leaf_tri_area_by_organ(test)
    common = sorted(set(a0) & set(a1))
    if not common:
        return False, "no common rendered leaf organs found"
    ratios = np.array([a1[i] / a0[i] for i in common if a0[i] > 0], dtype=float)
    ratio = float(np.median(ratios))
    live = abs(ratio - width_mult * width_mult) < 0.15 or abs(ratio - width_mult) < 0.15
    return live, (
        f"seed=0 day=70 median rendered leaf tri-area ratio {ratio:.3f} "
        f"(before median {np.median([a0[i] for i in common]):.3f} cm^2, "
        f"after median {np.median([a1[i] for i in common]):.3f} cm^2)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln_mult", type=float, default=1.0)
    ap.add_argument("--theta_mult", type=float, default=1.0)
    ap.add_argument("--width_mult", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    src = DEFAULT_XML
    out = OUT_DIR / f"maize_seg_spaced_{a.tag}.xml"
    tree = ET.parse(src)
    root = tree.getroot()

    width_live = True
    width_msg = "width_mult=1.0"
    if a.width_mult != 1.0:
        width_live, width_msg = _width_live(tree, a.width_mult)
        if not width_live:
            print(f"WARNING: width_mult appears inert; skipping width edits ({width_msg})")
        else:
            print(f"width_mult verified live: {width_msg}")

    ln_changes = []
    theta_changes = []
    width_changes = []

    for stem in root.findall("stem"):
        params = _params(stem)
        lmax = params.get("lmax")
        if stem.get("name") == "mainstem" and lmax is not None and abs(float(lmax.get("value")) - 210.0) < 1e-6:
            changed = _mul_param(params, "ln", a.ln_mult)
            if changed is not None:
                ln_changes.append((stem.get("name"),) + changed)

    for leaf in root.findall("leaf"):
        params = _params(leaf)
        changed = _mul_param(params, "theta", a.theta_mult)
        if changed is not None:
            theta_changes.append((leaf.get("name"),) + changed)
        if width_live and a.width_mult != 1.0:
            for name in ("Width_blade", "lmax", "areaMax"):
                changed = _mul_param(params, name, a.width_mult)
                if changed is not None:
                    width_changes.append((leaf.get("name"), name) + changed)

    tree.write(out, encoding="utf-8", xml_declaration=False)
    print(f"source: {src}")
    print(f"wrote:  {out}")
    print(f"mainstem ln changes: {len(ln_changes)}")
    for name, old, new in ln_changes:
        print(f"  {name}: ln {old:g} -> {new:g}")
    print(f"leaf theta changes: {len(theta_changes)} leaves, mult={a.theta_mult:g}")
    if theta_changes:
        vals = np.array([x[2] for x in theta_changes], dtype=float)
        print(f"  theta range after: {vals.min():.6g}..{vals.max():.6g}")
    print(f"leaf width-field changes: {len(width_changes)} params, mult={a.width_mult:g}")
    if a.width_mult != 1.0:
        print(f"width liveness evidence: {width_msg}")


if __name__ == "__main__":
    main()
