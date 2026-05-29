from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src/visualisation/pheno4d_to_g1")

from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.growth.grow import grow_plant
from dart.coupling.scripts.eval_segmenter_synthetic import (
    DEFAULT_XML,
    occlude_labelled,
    sample_labelled,
)
import mongraphseg_graph as mg


def n_terminals(graph):
    return sum(1 for n in graph.nodes() if graph.degree(n) == 1)


def main():
    totals = {
        "gt": 0,
        "initial": 0,
        "collapse": 0,
        "prune": 0,
        "ground": 0,
    }
    print(
        "plant_id GT_leaves terminals_after_build_initial_graph "
        "terminals_after_collapse_skeleton_tree terminals_after_prune "
        "terminals_after_remove_ground_nodes"
    )
    for plant_id in range(8):
        seed = plant_id
        rng = np.random.default_rng(seed)
        day = int(rng.integers(40, 93))
        plant = grow_plant(DEFAULT_XML, simulation_time=day, seed=seed)
        mesh = loft_organs(extract_organs_for_lofter(plant), use_nurbs_backend=True)
        comp, comp_lab = sample_labelled(mesh, 16384, rng)
        pts, gt = occlude_labelled(comp, comp_lab, rng)
        gt_leaves = len([i for i in np.unique(gt) if i != 0])

        contracted = mg.contract_point_cloud(pts)[0]
        sel = mg.farthest_point_resample(contracted, 400)
        nodes = contracted[sel]

        graph = mg.build_initial_graph(nodes, k=3)
        tree = mg.collapse_skeleton_tree(graph)
        pruned = mg.prune_short_branches(tree.copy(), min_branch_len_cm=3.0)
        grounded, _ = mg.remove_ground_nodes(pruned, angle_threshold_deg=50)

        row = {
            "gt": gt_leaves,
            "initial": n_terminals(graph),
            "collapse": n_terminals(tree),
            "prune": n_terminals(pruned),
            "ground": n_terminals(grounded),
        }
        for key, value in row.items():
            totals[key] += value
        print(
            f"{plant_id} {row['gt']} {row['initial']} {row['collapse']} "
            f"{row['prune']} {row['ground']}"
        )

    print(
        f"SUM {totals['gt']} {totals['initial']} {totals['collapse']} "
        f"{totals['prune']} {totals['ground']}"
    )


if __name__ == "__main__":
    main()
