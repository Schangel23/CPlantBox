"""Validate CurvatureProfileTropism: native grown getNodes kappa(s) vs RECON.

Success metric (per the plan): the native grown leaf-skeleton curvature profile
must OVERLAY the RECON profile per rank, not merely match total turning. Compared
in length-invariant turning-rate units dtheta/du = kappa * L (rad per normalized
arc), so the lmax/L_recon size difference does not confound the shape comparison.

  cpbenv/bin/python dart/coupling/scripts/validate_curvature_profile.py
Writes /home/lukas/pointr/kappa_profiles.png and prints per-rank L1 error.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # CPlantBox root for `dart`
from curvature_profile import kappa_profile          # noqa: E402
from fit_curvature_profile_to_recon import recon_kappa_by_rank  # noqa: E402

from dart.coupling.growth.grow import grow_plant      # noqa: E402

XML = "dart/coupling/data/maize_mirza_plant01.xml"
N_KNOTS = 12
OUT = "/home/lukas/pointr/kappa_profiles.png"


def main():
    rk = recon_kappa_by_rank()
    max_rank = max(rk)
    p = grow_plant(XML, simulation_time=90, seed=42)
    leaves = [o for o in p.getOrgans() if o.organType() == 4]

    # one curve per RECON rank: pick the grown leaf whose shape_rank_index == rank
    by_rank = {}
    for o in leaves:
        rp = o.getOrganRandomParameter()
        sri = int(getattr(rp, "shape_rank_index", -1))
        rank = min(sri, max_rank) if sri >= 0 else max_rank
        nodes = np.array([[n.x, n.y, n.z] for n in o.getNodes()])
        if len(nodes) < 6:
            continue
        by_rank.setdefault(rank, nodes)   # first (lowest subType) wins per rank

    ranks = sorted(by_rank)
    ncol = 4
    nrow = int(np.ceil(len(ranks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    errs = []
    for ax, rank in zip(axes.ravel(), ranks):
        phi_r, kap_r, arc_r = rk[rank]
        dtheta_recon = kap_r * arc_r                       # rad per normalized arc

        nodes = by_rank[rank]
        L_nat = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
        phi_n, kap_n = kappa_profile(nodes, n_knots=N_KNOTS)
        dtheta_nat = kap_n * L_nat

        l1 = float(np.mean(np.abs(dtheta_nat - dtheta_recon)))
        errs.append((rank, l1, float(dtheta_recon.sum() * (phi_r[1] - phi_r[0])),
                     float(dtheta_nat.sum() * (phi_n[1] - phi_n[0]))))
        ax.plot(phi_r, np.degrees(dtheta_recon), "o-", label="RECON", color="tab:blue")
        ax.plot(phi_n, np.degrees(dtheta_nat), "s--", label="native", color="tab:red")
        ax.set_title(f"rank {rank}  L1={np.degrees(l1):.1f}deg/u")
        ax.set_xlabel("arc fraction")
        ax.set_ylabel("dtheta/du [deg]")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(ranks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"\n{'rank':>4} {'L1[deg/u]':>10} {'turn_recon':>11} {'turn_native':>12}")
    for rank, l1, tr, tn in errs:
        print(f"{rank:>4} {np.degrees(l1):>10.2f} {np.degrees(tr):>10.1f}deg {np.degrees(tn):>11.1f}deg")
    print(f"\nmean L1 = {np.degrees(np.mean([e[1] for e in errs])):.2f} deg/u")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
