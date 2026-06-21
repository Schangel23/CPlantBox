"""Fit per-rank kappa(s) from FP4D RECON and write it into the maize leaf RPs.

For each maize leaf subType, take its shape_rank_index -> RECON rank, pull that
rank's last-reliable-stage midrib (cps_{r}[stage][:, 3, :], the v=mid column),
compute kappa(s) (curvature_profile.kappa_profile), and write a
<parameter name="leafCurvature" phi="..." kappa="..."/> element plus flip
tropismT 8 -> 9 (tt_curvature_profile). Dry-run by default; pass --write to edit.

  cpbenv/bin/python dart/coupling/scripts/fit_curvature_profile_to_recon.py [--write]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curvature_profile import kappa_profile  # noqa: E402

LIB = Path("/home/lukas/pointr/fp4d_plants_basecarry/Plot03/plant_01/plant01_stage_library.npz")
XML = Path("dart/coupling/data/maize_mirza_plant01.xml")
V_MID = 3          # midrib column of the 7-wide (0..6) RECON CP grid
N_KNOTS = 12


def recon_kappa_by_rank():
    """rank -> (phi, kappa, arc_len_cm) from the last reliable stage."""
    d = np.load(LIB, allow_pickle=True)
    ranks = sorted(int(k.split("_")[1]) for k in d.files if re.fullmatch(r"cps_\d+", k))
    out = {}
    for r in ranks:
        cps = d[f"cps_{r}"]                      # (n_stage, 18, 7, 3)
        rel = np.asarray(d[f"reliable_{r}"]).ravel().astype(bool)
        stages = np.where(rel)[0]
        if len(stages) == 0:
            continue
        st = int(stages[-1])                     # last reliable stage
        mid = cps[st][:, V_MID, :]               # (18, 3) midrib polyline
        arc = float(np.linalg.norm(np.diff(mid, axis=0), axis=1).sum())
        phi, kap = kappa_profile(mid, n_knots=N_KNOTS)
        out[r] = (phi, kap, arc)
    return out


def fmt(vals):
    return " ".join(f"{v:.6g}" for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rk = recon_kappa_by_rank()
    max_rank = max(rk)
    print(f"RECON ranks with kappa: {sorted(rk)}  (max={max_rank})")

    xml = XML.read_text()
    blocks = list(re.finditer(r'(<leaf name="([^"]+)" subType="(\d+)">)(.*?)(</leaf>)', xml, re.S))
    print(f"leaf subTypes in XML: {len(blocks)}\n")

    edits = []
    for m in blocks:
        body = m.group(4)
        st = m.group(3)
        sri_m = re.search(r'shape_rank_index" value="(-?\d+)"', body)
        sri = int(sri_m.group(1)) if sri_m else -1
        rank = min(sri, max_rank) if sri >= 0 else max_rank   # clamp ranks beyond RECON
        phi, kap_recon, arc = rk[rank]
        lmax_m = re.search(r'lmax" value="([0-9.]+)"', body)
        lmax = float(lmax_m.group(1)) if lmax_m else float("nan")
        # Reproduce RECON's leaf SHAPE at the leaf's own calibrated length lmax:
        # scaling a curve by lmax/L_recon scales its curvature by L_recon/lmax.
        # This preserves the turning DISTRIBUTION and TOTAL turning (the kappa(s)
        # profile vs normalized arc), decoupled from the size knob (lmax via FA).
        kap = kap_recon * (arc / max(lmax, 1e-9))
        tt = re.search(r'tropismT" value="(\d+)"', body)
        print(f"subType={st:>2} sri={sri:>2} -> rank={rank:>2}  "
              f"arc_recon={arc:6.2f}cm  lmax={lmax:6.2f}cm  tt={tt.group(1) if tt else '?'}  "
              f"turn_recon={float((kap_recon.mean()*arc)*57.3):6.1f}deg  "
              f"kappa_xml[1/cm] min={kap.min():.4f} max={kap.max():.4f} mean={kap.mean():.4f}")

        # build the new leaf block body
        new_body = re.sub(r'(<parameter name="tropismT" value=")\d+(" ?/?>)',
                          r'\g<1>9\g<2>', body)
        # remove any stale leafCurvature, then insert fresh just before close
        new_body = re.sub(r'\s*<parameter name="leafCurvature"[^>]*/>', '', new_body)
        ins = (f'        <parameter name="leafCurvature" '
               f'phi="{fmt(phi)}" kappa="{fmt(kap)}" />\n    ')
        new_body = new_body.rstrip() + "\n" + ins
        edits.append((m.group(0), m.group(1) + new_body + m.group(5)))

    if not args.write:
        print("\n[dry-run] pass --write to edit the XML")
        return

    for old, new in edits:
        xml = xml.replace(old, new, 1)
    XML.write_text(xml)
    print(f"\n[written] {XML}  ({len(edits)} leaf subTypes flipped to tt=9 + leafCurvature)")


if __name__ == "__main__":
    main()
