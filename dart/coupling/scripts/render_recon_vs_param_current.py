"""Render FP4D-intercept vs current-parametric stage previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("dart/coupling/output/parametric_stage_validation")


def _norm_xz(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[:, 0]
    z = points[:, 2]
    x_scale = max(float(np.max(np.abs(x))), 1e-9)
    z_min = float(np.min(z))
    z_scale = max(float(np.max(z) - z_min), 1e-9)
    return x / x_scale, (z - z_min) / z_scale


def _plot_leaf(ax, leaf: dict, title: str) -> None:
    ref = np.asarray(leaf["reference"], dtype=float)
    cur = np.asarray(leaf["current"], dtype=float)
    rx, rz = _norm_xz(ref)
    cx, cz = _norm_xz(cur)
    ax.scatter(rx, rz, s=6, c="#222222", alpha=0.45, label="FP4D intercept")
    ax.scatter(cx, cz, s=5, c="#0072B2", alpha=0.50, label="current")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])


def render(input_dir: Path, output: Path) -> Path:
    manifest = json.loads((input_dir / "manifest.json").read_text())
    stages = manifest["stages"]
    fig, axes = plt.subplots(len(stages), 3, figsize=(8.5, 2.7 * len(stages)), squeeze=False)
    for row, stage_meta in enumerate(stages):
        payload = json.loads(Path(stage_meta["path"]).read_text())
        leaves = payload["leaves"]
        picks = [0, max(0, len(leaves) // 2), max(0, len(leaves) - 1)]
        for col, idx in enumerate(picks):
            ax = axes[row][col]
            if not leaves:
                ax.set_axis_off()
                continue
            leaf = leaves[min(idx, len(leaves) - 1)]
            _plot_leaf(ax, leaf, f"{payload['stage']} rank {leaf['rank']}")
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT / "recon_vs_param_current.png")
    args = parser.parse_args()
    print(render(args.input, args.output))


if __name__ == "__main__":
    main()
