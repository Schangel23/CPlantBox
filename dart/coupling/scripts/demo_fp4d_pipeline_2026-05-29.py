#!/usr/bin/env python3
"""Real-cloud FP4D maize demo: fill -> MonGraphSeg -> NURBS."""

import os
import sys
import ctypes
import importlib.util
import subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cplantbox")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, "/home/lukas/pointr")

from combined_fill import combined_fill


ROOT = Path("/home/lukas/PHD/CPlantBox")
OUT_DIR = Path("/home/lukas/pointr/organs")
FALLBACK_OUT_DIR = ROOT / "dart" / "coupling" / "output" / "fp4d_demo_2026-05-29"
PHENO4D_DIR = ROOT / "src" / "visualisation" / "pheno4d_to_g1"
INPUTS = [
    ("whole_plant", OUT_DIR / "whole_plant.npy", "Plot04/230621"),
    ("whole_plant_hard", OUT_DIR / "whole_plant_hard.npy", "Plot03/230705"),
]
MAX_HARD_POINTS = 8000
MIN_NURBS_POINTS = 120


def ensure_suitesparse_symbols():
    """Expose SuiteSparse symbols needed by this local CPlantBox build."""
    shim = Path("/tmp/cplantbox_suitesparse_shim.so")
    libs = [
        ROOT / "src/external/suitsparse/lib/libamd.a",
        ROOT / "src/external/suitsparse/lib/libbtf.a",
        ROOT / "src/external/suitsparse/lib/libcolamd.a",
        ROOT / "src/external/suitsparse/lib/libsuitesparseconfig.a",
    ]
    if not all(p.exists() for p in libs):
        return
    if not shim.exists() or any(p.stat().st_mtime > shim.stat().st_mtime for p in libs):
        cmd = [
            "gcc",
            "-shared",
            "-o",
            str(shim),
            "-Wl,--whole-archive",
            *[str(p) for p in libs],
            "-Wl,--no-whole-archive",
            "-lm",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)


def load_module(module_name, path):
    """Load an in-tree module without importing src/__init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ensure_suitesparse_symbols()
mongraphseg_graph = load_module("mongraphseg_graph", PHENO4D_DIR / "mongraphseg_graph.py")
nurbs_leaf_fit = load_module("nurbs_leaf_fit", PHENO4D_DIR / "nurbs_leaf_fit.py")
segment_plant_pseudostem = mongraphseg_graph.segment_plant_pseudostem
fit_leaf_nurbs_surface = nurbs_leaf_fit.fit_leaf_nurbs_surface
nrbeval_grid = nurbs_leaf_fit.nrbeval_grid
to_nurbs_patch = nurbs_leaf_fit.to_nurbs_patch


def recenter_z(pts):
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts.copy()
    pts[:, 2] -= float(pts[:, 2].min())
    return pts


def voxel_downsample(points, max_points=MAX_HARD_POINTS):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points, None

    def sample_at(voxel_size):
        keys = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(keep)]

    span = np.ptp(points, axis=0)
    diag = float(np.linalg.norm(span))
    lo, hi = 1e-6, max(diag / 5.0, 1.0)
    best = None
    best_voxel = hi
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        cand = sample_at(mid)
        if len(cand) > max_points:
            lo = mid
        else:
            best = cand
            best_voxel = mid
            hi = mid

    if best is None:
        best = sample_at(hi)
        best_voxel = hi

    if len(best) > max_points:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(best), size=max_points, replace=False))
        best = best[keep]
    return best, best_voxel


def organ_keys(seg):
    return sorted(
        [k for k in seg if k.startswith("leaf_")],
        key=lambda x: int(x.split("_", 1)[1]),
    )


def leaf_count(seg):
    return len(organ_keys(seg))


def segment_cloud(pts):
    return segment_plant_pseudostem(
        pts, n_skel_nodes=250, min_leaf_len_cm=3.0, prune="length"
    )


def principal_axis(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if axis[2] < 0:
        axis = -axis
    return axis


def skeleton_metrics(seg):
    rows = []
    for key in organ_keys(seg):
        pts = np.asarray(seg[key], dtype=np.float64)
        if len(pts) == 0:
            continue
        h = float(pts[:, 2].min())
        cutoff = np.quantile(pts[:, 2], 0.35) if len(pts) >= 10 else pts[:, 2].max()
        base_pts = pts[pts[:, 2] <= cutoff]
        if len(base_pts) < 3:
            base_pts = pts[np.argsort(pts[:, 2])[: min(len(pts), 20)]]
        axis = principal_axis(base_pts)
        angle = float(np.degrees(np.arccos(np.clip(abs(axis[2]), -1.0, 1.0))))
        rows.append({"leaf": key, "n_points": len(pts), "height": h, "angle": angle})

    rows.sort(key=lambda r: r["height"])
    heights = np.array([r["height"] for r in rows], dtype=float)
    gaps = np.diff(heights) if len(heights) > 1 else np.array([], dtype=float)
    for i, row in enumerate(rows):
        row["next_gap"] = float(gaps[i]) if i < len(gaps) else np.nan
    return rows


def print_metrics_table(rows):
    print("  leaf      points  insertion_z_cm  base_axis_angle_deg  next_internode_cm")
    print("  --------  ------  --------------  -------------------  ----------------")
    if not rows:
        print("  (no segmented leaves)")
        return
    for r in rows:
        gap = "" if np.isnan(r["next_gap"]) else f"{r['next_gap']:.2f}"
        print(
            f"  {r['leaf']:<8}  {r['n_points']:>6d}  "
            f"{r['height']:>14.2f}  {r['angle']:>19.1f}  {gap:>16}"
        )


def color_cycle(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(max(n, 1))]


def set_axes_equal(ax, pts):
    pts = np.asarray(pts)
    center = pts.mean(axis=0)
    radius = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), float(np.ptp(pts[:, 2]))) / 2
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_xlabel("x cm")
    ax.set_ylabel("y cm")
    ax.set_zlabel("z cm")
    ax.view_init(elev=18, azim=-62)


def plot_segmentation(ax, seg, title, added=None):
    keys = ["pseudostem"] + organ_keys(seg)
    colors = color_cycle(len(keys))
    for i, key in enumerate(keys):
        pts = seg.get(key)
        if pts is None or len(pts) == 0:
            continue
        size = 4 if key.startswith("leaf_") else 3
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, color=colors[i], alpha=0.82, label=key)
    if added is not None and len(added):
        ax.scatter(
            added[:, 0],
            added[:, 1],
            added[:, 2],
            s=7,
            color="magenta",
            alpha=0.95,
            label="fill_added",
            depthshade=False,
        )
    all_pts = np.vstack([v for v in seg.values() if len(v)])
    if added is not None and len(added):
        all_pts = np.vstack([all_pts, added])
    set_axes_equal(ax, all_pts)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, markerscale=2)


def render_before_after(name, raw_seg, filled_seg, added):
    path = OUT_DIR / f"demo_{name}_beforeafter.png"
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_segmentation(ax1, raw_seg, "raw segmentation")
    plot_segmentation(ax2, filled_seg, "filled segmentation", added=added)
    fig.suptitle(f"{name}: FP4D raw vs combined_fill")
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def fit_nurbs_for_leaves(seg):
    fits = {}
    reports = []
    for key in organ_keys(seg):
        pts = np.asarray(seg[key], dtype=np.float64)
        if len(pts) < MIN_NURBS_POINTS:
            msg = f"{key}: SKIP sparse ({len(pts)} pts < {MIN_NURBS_POINTS})"
            print(f"  {msg}")
            reports.append(msg)
            continue
        try:
            fit = fit_leaf_nurbs_surface(pts)
            patch = to_nurbs_patch(fit)
            fit["patch"] = patch
            fits[key] = fit
            rms_cm = float(fit["fit_rms"])
            msg = f"{key}: RMS {rms_cm:.3f} cm ({10.0 * rms_cm:.1f} mm), n_used={fit['n_used']}"
            print(f"  {msg}")
            reports.append(msg)
        except Exception as exc:
            msg = f"{key}: FAIL {type(exc).__name__}: {exc}"
            print(f"  {msg}")
            reports.append(msg)
    return fits, reports


def render_nurbs(name, seg, fits):
    path = OUT_DIR / f"demo_{name}_nurbs.png"
    leaves = organ_keys(seg)
    n = max(len(leaves), 1)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5.2 * cols, 4.8 * rows), constrained_layout=True)
    if not leaves:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.text2D(0.2, 0.5, "No segmented leaves", transform=ax.transAxes)
    for i, key in enumerate(leaves, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        pts = np.asarray(seg[key], dtype=np.float64)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, color="#2f6f9f", alpha=0.55)
        if key in fits:
            grid = nrbeval_grid(fits[key], nu=50, nv=18)
            ax.plot_surface(
                grid[:, :, 0],
                grid[:, :, 1],
                grid[:, :, 2],
                color="#f0b429",
                alpha=0.48,
                linewidth=0,
                antialiased=True,
            )
            rms = float(fits[key]["fit_rms"])
            ax.set_title(f"{key} RMS {rms:.2f} cm")
        else:
            ax.set_title(f"{key} no NURBS")
        set_axes_equal(ax, pts)
    fig.suptitle(f"{name}: segmented leaves with fitted NURBS surfaces")
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def save_figure(fig, requested_path):
    requested_path = Path(requested_path)
    try:
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(requested_path, dpi=180)
        return requested_path
    except OSError as exc:
        if exc.errno != 30:
            raise
        FALLBACK_OUT_DIR.mkdir(parents=True, exist_ok=True)
        fallback = FALLBACK_OUT_DIR / requested_path.name
        fig.savefig(fallback, dpi=180)
        print(f"Requested PNG path is read-only: {requested_path}")
        print(f"Saved fallback PNG instead: {fallback}")
        return fallback


def run_one(name, npy_path, label, rng):
    print(f"\n=== {name} ({label}) ===")
    raw = recenter_z(np.load(npy_path))
    print(f"Loaded {npy_path}: {len(raw)} points, z=[{raw[:,2].min():.2f}, {raw[:,2].max():.2f}] cm")

    if name == "whole_plant_hard" and len(raw) > MAX_HARD_POINTS:
        before = len(raw)
        raw, voxel = voxel_downsample(raw, MAX_HARD_POINTS)
        print(
            f"Voxel-downsampled {name}: {before} -> {len(raw)} points "
            f"(voxel ~= {voxel:.3f} cm) because the cloud is dense."
        )

    print("Segmenting raw cloud...")
    raw_seg = segment_cloud(raw)
    raw_count = leaf_count(raw_seg)
    print(f"Raw leaf count: {raw_count}")

    print("Running combined_fill...")
    fill_failed = False
    try:
        added = combined_fill(raw, rng, return_added_only=True)
        filled = np.vstack([raw.astype(np.float32), added.astype(np.float32)])
        filled = recenter_z(filled)
        if len(added):
            added = recenter_z(added)
        print(f"combined_fill added {len(added)} points; filled cloud has {len(filled)} points.")
    except Exception as exc:
        fill_failed = True
        added = np.zeros((0, 3), dtype=np.float64)
        filled = raw
        print(f"combined_fill FAILED ({type(exc).__name__}: {exc}); continuing with raw cloud only.")

    print("Segmenting filled cloud...")
    filled_seg = segment_cloud(filled)
    filled_count = leaf_count(filled_seg)
    print(f"Filled leaf count: {filled_count}")
    print(f"Leaf count summary: raw={raw_count}, filled={filled_count}")
    if fill_failed:
        print("Fill step failed, so filled segmentation is the raw-only fallback.")
    elif filled_count > raw_count:
        recovered = [f"leaf_{i}" for i in range(raw_count + 1, filled_count + 1)]
        print(f"Recovered by fill: {', '.join(recovered)} present in filled but absent/merged in raw.")
    elif filled_count == raw_count:
        print("Fill did not change the segmentation leaf count on this real cloud.")
    else:
        print("Fill reduced the segmentation leaf count on this real cloud, likely by merging/smoothing fragments.")

    print("\nSkeleton-graph metrics on filled segmentation:")
    metrics = skeleton_metrics(filled_seg)
    print_metrics_table(metrics)

    print("\nPer-leaf NURBS fits on filled segmentation:")
    fits, fit_reports = fit_nurbs_for_leaves(filled_seg)

    before_after_png = render_before_after(name, raw_seg, filled_seg, added)
    nurbs_png = render_nurbs(name, filled_seg, fits)
    print(f"Before/after PNG: {before_after_png}")
    print(f"NURBS PNG:        {nurbs_png}")

    return {
        "name": name,
        "raw_count": raw_count,
        "filled_count": filled_count,
        "metrics": metrics,
        "fit_reports": fit_reports,
        "pngs": [before_after_png, nurbs_png],
    }


def main():
    os.chdir(ROOT)
    rng = np.random.default_rng(42)
    results = [run_one(name, path, label, rng) for name, path, label in INPUTS]

    print("\n=== Verification ===")
    missing = []
    for result in results:
        for png in result["pngs"]:
            ok = png.exists() and png.stat().st_size > 0
            print(f"{png}: {'OK' if ok else 'MISSING'}")
            if not ok:
                missing.append(str(png))
    if missing:
        raise RuntimeError("Missing expected PNG outputs: " + ", ".join(missing))

    print("\n=== Final summary ===")
    for result in results:
        print(
            f"{result['name']}: raw leaves={result['raw_count']}, "
            f"filled leaves={result['filled_count']}, PNGs="
            f"{', '.join(str(p) for p in result['pngs'])}"
        )


if __name__ == "__main__":
    main()
