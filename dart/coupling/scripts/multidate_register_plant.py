"""Register one FieldPheno4D plant across all available dates.

Produces a dense accumulated (N, 3) cm float32 cloud for one plant, using a
lower-stem anchor zone for rigid ICP. This intentionally stops before organ
segmentation, filling, or NURBS fitting.

Run from the CPlantBox root:

    PYTHONPATH=. OMP_NUM_THREADS=1 cpbenv/bin/python \
        dart/coupling/scripts/multidate_register_plant.py --plot Plot04
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


sys.path.insert(0, "src/visualisation/pheno4d_to_g1")
from loader import load_las, separate_plants_along_row, crop_plant_window


FP4D = Path(
    "/home/lukas/PHD/Resources/PHENOROAM DATA ASSIMILATION May 2026/"
    "doi-10.60507-fk2-hyi2ds"
)
DEFAULT_OUT = Path("/home/lukas/pointr/organs")


@dataclass
class DateResult:
    date: str
    centre: float
    delta_centre: float
    n_pts: int
    height: float
    delta_height: float
    anchor_rmse: float
    whole_anchor_rmse: float
    novel_pct: float
    included: bool
    reason: str
    cloud_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", default="Plot04")
    p.add_argument("--plant-centre", default="auto",
                   help="row-axis centre in cm on the reference date, or 'auto'")
    p.add_argument("--ref-date", default=None,
                   help="reference date stem, e.g. 230621; default: densest date")
    p.add_argument("--dates", nargs="*", default=None,
                   help="date stems to use; default: all .las files in plot dir")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--anchor-height-cm", type=float, default=18.0,
                   help="use points with 0 <= z <= this height for ICP")
    p.add_argument("--anchor-rmse-max-cm", type=float, default=1.5)
    p.add_argument("--max-height-delta-cm", type=float, default=25.0)
    p.add_argument("--min-auto-pts", type=int, default=1500)
    p.add_argument("--window-cm", type=float, default=20.0)
    p.add_argument("--cross-row-window-cm", type=float, default=25.0)
    p.add_argument("--accum-voxel-cm", type=float, default=0.3)
    p.add_argument("--data-root", default=str(FP4D))
    return p.parse_args()


def las_dates(data_root: Path, plot: str, dates: list[str] | None) -> dict[str, Path]:
    plot_dir = data_root / plot
    if not plot_dir.exists():
        raise FileNotFoundError(f"missing plot directory: {plot_dir}")
    found = {p.stem: p for p in sorted(plot_dir.glob("*.las"))}
    if not found:
        raise FileNotFoundError(f"no .las files in {plot_dir}")
    if dates is None:
        return found
    missing = [d for d in dates if d not in found]
    if missing:
        raise FileNotFoundError(f"requested dates missing for {plot}: {missing}")
    return {d: found[d] for d in dates}


def load_row(path: Path) -> np.ndarray:
    return load_las(str(path), height_lo_m=0.15, voxel_m=0.005,
                    ground_method="height")


def voxel(p: np.ndarray, v: float = 0.3) -> np.ndarray:
    mn = p.min(0)
    ijk = np.floor((p - mn) / v).astype(np.int64)
    _, u = np.unique(ijk, axis=0, return_index=True)
    return p[u]


def kabsch_icp_transform(src: np.ndarray, dst: np.ndarray, iters: int = 40,
                         reject_pct: float = 80.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP (Kabsch), trimmed. Returns (R, t, rmse)."""
    if len(src) < 3 or len(dst) < 3:
        return np.eye(3), np.zeros(3), float("inf")
    s = src.copy()
    r_tot = np.eye(3)
    t_tot = np.zeros(3)
    tree = cKDTree(dst)
    for _ in range(iters):
        d, idx = tree.query(s)
        thr = np.percentile(d, reject_pct)
        m = d <= thr
        if m.sum() < 3:
            break
        A, B = s[m], dst[idx[m]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca
        s = (R @ s.T).T + t
        r_tot = R @ r_tot
        t_tot = R @ t_tot + t
    d, _ = cKDTree(dst).query((r_tot @ src.T).T + t_tot)
    return r_tot, t_tot, float(np.sqrt((d ** 2).mean()))


def apply_transform(p: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ p.T).T + t


def bounded_cloud(p: np.ndarray, max_pts: int = 25000) -> np.ndarray:
    if len(p) <= max_pts:
        return p
    idx = np.linspace(0, len(p) - 1, max_pts).astype(np.int64)
    return p[idx]


def rmse_to_dst(src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0 or len(dst) == 0:
        return float("inf")
    d, _ = cKDTree(dst).query(src)
    return float(np.sqrt((d ** 2).mean()))


def anchor(p: np.ndarray, height_cm: float) -> np.ndarray:
    return p[(p[:, 2] >= 0.0) & (p[:, 2] <= height_cm)]


def crop_for_centre(row: np.ndarray, row_axis: int, centre: float,
                    args: argparse.Namespace) -> np.ndarray:
    return crop_plant_window(row, row_axis, centre, window_cm=args.window_cm,
                             cross_row_window_cm=args.cross_row_window_cm)


def estimate_base_xy(row: np.ndarray, row_axis: int, centre: float,
                     args: argparse.Namespace) -> np.ndarray:
    coord = row[:, row_axis]
    sel = (coord >= centre - args.window_cm / 2) & (coord <= centre + args.window_cm / 2)
    P = row[sel].copy()
    if len(P) == 0:
        return np.array([centre, 0.0])
    cross = 1 - row_axis
    z_floor = float(P[:, 2].min())
    base_mask = P[:, 2] <= z_floor + 5.0
    if base_mask.sum() < 10:
        base_mask = P[:, 2] <= np.percentile(P[:, 2], 5)
    base_cross = float(np.median(P[base_mask, cross]))
    if args.cross_row_window_cm and args.cross_row_window_cm > 0:
        P = P[np.abs(P[:, cross] - base_cross) <= args.cross_row_window_cm / 2]
    if len(P) == 0:
        xy = np.zeros(2)
        xy[row_axis] = centre
        xy[cross] = base_cross
        return xy
    return P[int(np.argmin(P[:, 2])), :2].copy()


def choose_reference(paths: dict[str, Path], requested: str | None) -> str:
    if requested is not None:
        if requested not in paths:
            raise FileNotFoundError(f"reference date {requested} not in inventory")
        return requested
    print("\n=== reference-date scan: densest loaded date ===", flush=True)
    counts = {}
    for date, path in paths.items():
        print(f"[ref-scan] loading {date}: {path}", flush=True)
        counts[date] = len(load_row(path))
        print(f"[ref-scan] {date}: {counts[date]:,} row pts", flush=True)
    ref = max(counts, key=counts.get)
    print(f"[ref-scan] selected ref date {ref} ({counts[ref]:,} pts)", flush=True)
    return ref


def choose_ref_plant(ref_row: np.ndarray, ref_axis: int, ref_centres: np.ndarray,
                     args: argparse.Namespace) -> tuple[float, np.ndarray]:
    if args.plant_centre != "auto":
        centre = float(args.plant_centre)
        crop = crop_for_centre(ref_row, ref_axis, centre, args)
        return centre, crop

    best = None
    for k, centre in enumerate(ref_centres):
        crop = crop_for_centre(ref_row, ref_axis, float(centre), args)
        if len(crop) < args.min_auto_pts:
            continue
        height = float(np.ptp(crop[:, 2]))
        if best is None or height > best[2]:
            best = (k, float(centre), height, crop)
    if best is None:
        raise RuntimeError(
            f"auto-pick failed: no plant with >= {args.min_auto_pts} pts on ref date"
        )
    k, centre, height, crop = best
    print(f"[ref] auto-picked plant k={k} centre={centre:.1f} cm "
          f"height={height:.1f} cm n={len(crop):,}", flush=True)
    return centre, crop


def register_date(date: str, path: Path, ref_centre: float, ref_cloud: np.ndarray,
                  ref_anchor: np.ndarray, ref_height: float,
                  args: argparse.Namespace, out_dir: Path,
                  filename_tag: str) -> tuple[DateResult, np.ndarray]:
    print(f"\n=== date {date}: load / split / crop / register ===", flush=True)
    row = load_row(path)
    row_axis, centres = separate_plants_along_row(row)
    j = int(np.argmin(np.abs(np.asarray(centres) - ref_centre)))
    centre = float(centres[j])
    crop = crop_for_centre(row, row_axis, centre, args)
    delta_centre = abs(centre - ref_centre)
    height = float(np.ptp(crop[:, 2])) if len(crop) else 0.0
    delta_height = height - ref_height
    src_anchor = anchor(crop, args.anchor_height_cm)

    if delta_centre > args.window_cm:
        reason = "centre-miss"
        R, t, anchor_rmse = np.eye(3), np.zeros(3), float("inf")
        whole_anchor_rmse = float("inf")
    elif len(src_anchor) < 20 or len(ref_anchor) < 20:
        reason = "too-few-anchor-points"
        R, t, anchor_rmse = np.eye(3), np.zeros(3), float("inf")
        whole_anchor_rmse = float("inf")
    else:
        R, t, anchor_rmse = kabsch_icp_transform(src_anchor, ref_anchor)
        Rw, tw, _ = kabsch_icp_transform(bounded_cloud(crop), bounded_cloud(ref_cloud))
        whole_anchor_rmse = rmse_to_dst(apply_transform(src_anchor, Rw, tw), ref_anchor)
        reason = ""

    reg = apply_transform(crop, R, t)
    d, _ = cKDTree(ref_cloud).query(reg) if len(reg) else (np.array([]), None)
    novel_pct = float(100.0 * (d > 1.0).sum() / max(len(reg), 1))
    included = (
        len(reg) > 0
        and np.isfinite(anchor_rmse)
        and anchor_rmse <= args.anchor_rmse_max_cm
        and abs(delta_height) <= args.max_height_delta_cm
        and reason == ""
    )
    if not included and reason == "":
        reasons = []
        if anchor_rmse > args.anchor_rmse_max_cm:
            reasons.append("high-anchor-rmse")
        if abs(delta_height) > args.max_height_delta_cm:
            reasons.append("height-delta")
        reason = "+".join(reasons)

    cloud_path = out_dir / f"multidate_{filename_tag}_{date}_registered.npy"
    np.save(cloud_path, reg.astype(np.float32))
    print(f"[date {date}] saved registered cloud -> {cloud_path}", flush=True)

    result = DateResult(
        date=date, centre=centre, delta_centre=delta_centre, n_pts=len(reg),
        height=height, delta_height=delta_height, anchor_rmse=anchor_rmse,
        whole_anchor_rmse=whole_anchor_rmse, novel_pct=novel_pct,
        included=included, reason=reason or "ok", cloud_path=cloud_path)
    return result, reg


def format_table(results: list[DateResult]) -> str:
    lines = [
        "date   | Δcentre | n_pts  | Δheight_vs_ref | anchor-RMSE | novel% | flag",
        "-------+---------+--------+----------------+-------------+--------+----------------",
    ]
    for r in results:
        flag = "INCLUDED" if r.included else f"EXCLUDED:{r.reason}"
        lines.append(
            f"{r.date} | {r.delta_centre:7.1f} | {r.n_pts:6d} |"
            f" {r.delta_height:14.1f} | {r.anchor_rmse:11.3f} |"
            f" {r.novel_pct:6.1f} | {flag}"
        )
    return "\n".join(lines)


def render_qc(plot: str, ref_date: str, ref_cloud: np.ndarray,
              results: list[DateResult], clouds: dict[str, np.ndarray],
              accum: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results) + 1
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows),
                             squeeze=False)
    axes_flat = axes.ravel()

    ref_sample = bounded_cloud(ref_cloud, max_pts=8000)
    for ax, r in zip(axes_flat, results):
        reg = bounded_cloud(clouds[r.date], max_pts=8000)
        ax.scatter(ref_sample[:, 1], ref_sample[:, 2], s=1, c="#b8b8b8",
                   alpha=0.35, linewidths=0)
        ax.scatter(reg[:, 1], reg[:, 2], s=1, c="#1677b8",
                   alpha=0.8, linewidths=0)
        flag = "IN" if r.included else "OUT"
        ax.set_title(
            f"{r.date} {flag}\n"
            f"n={r.n_pts:,} rmse={r.anchor_rmse:.2f}cm "
            f"dh={r.delta_height:+.1f}cm novel={r.novel_pct:.0f}%"
        )
        ax.set_xlabel("y (cm)")
        ax.set_ylabel("z (cm)")
        ax.set_aspect("equal", adjustable="box")

    ax = axes_flat[len(results)]
    acc_sample = bounded_cloud(accum, max_pts=25000)
    ax.scatter(acc_sample[:, 1], acc_sample[:, 2], s=1, c="#26823d",
               alpha=0.8, linewidths=0)
    ax.set_title(f"accumulated\nn={len(accum):,}")
    ax.set_xlabel("y (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_aspect("equal", adjustable="box")

    for ax in axes_flat[len(results) + 1:]:
        ax.axis("off")
    fig.suptitle(f"{plot} multi-date plant registration, ref {ref_date}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = las_dates(data_root, args.plot, args.dates)
    print(f"Inventory {args.plot}: " +
          ", ".join(f"{d}={p.name}" for d, p in paths.items()), flush=True)

    ref_date = choose_reference(paths, args.ref_date)
    print(f"\n=== reference {ref_date}: load / split / choose plant ===", flush=True)
    ref_row = load_row(paths[ref_date])
    ref_axis, ref_centres = separate_plants_along_row(ref_row)
    ref_centre, ref_cloud = choose_ref_plant(ref_row, ref_axis, ref_centres, args)
    ref_base_xy = estimate_base_xy(ref_row, ref_axis, ref_centre, args)
    ref_height = float(np.ptp(ref_cloud[:, 2]))
    ref_anchor = anchor(ref_cloud, args.anchor_height_cm)
    if len(ref_anchor) < 20:
        raise RuntimeError(f"reference anchor has too few points: {len(ref_anchor)}")

    filename_tag = (
        f"{args.plot}_{ref_base_xy[0]:.1f}_{ref_base_xy[1]:.1f}"
        .replace("-", "m").replace(".", "p")
    )
    print(f"[ref] row_axis={ref_axis} centre={ref_centre:.1f} cm "
          f"base_xy=({ref_base_xy[0]:.1f},{ref_base_xy[1]:.1f}) cm "
          f"height={ref_height:.1f} cm anchor_n={len(ref_anchor):,}", flush=True)

    results = []
    clouds = {}
    for date, path in paths.items():
        result, reg = register_date(date, path, ref_centre, ref_cloud, ref_anchor,
                                    ref_height, args, out_dir, filename_tag)
        results.append(result)
        clouds[date] = reg

    included_clouds = [clouds[r.date] for r in results if r.included]
    if not included_clouds:
        raise RuntimeError("no dates passed inclusion thresholds")
    accum = voxel(np.vstack(included_clouds), v=args.accum_voxel_cm).astype(np.float32)
    accum_path = out_dir / f"multidate_{filename_tag}_accum.npy"
    np.save(accum_path, accum)

    qc_path = out_dir / f"multidate_{filename_tag}_qc.png"
    render_qc(args.plot, ref_date, ref_cloud, results, clouds, accum, qc_path)

    table = format_table(results)
    lower_not_better = [
        r.date for r in results
        if r.date != ref_date and r.anchor_rmse >= r.whole_anchor_rmse
    ]
    finding = ""
    if len(lower_not_better) >= 2:
        finding = (
            "\nFinding: lower-stem anchoring did not beat whole-cloud ICP on "
            f"{len(lower_not_better)} dates ({', '.join(lower_not_better)}), "
            "using anchor-region RMSE after each estimated transform."
        )

    report_path = out_dir / f"multidate_{filename_tag}_report.txt"
    report = (
        f"plot={args.plot}\nref_date={ref_date}\nref_centre_cm={ref_centre:.3f}\n"
        f"ref_base_xy_cm={ref_base_xy[0]:.3f},{ref_base_xy[1]:.3f}\n"
        f"anchor_height_cm={args.anchor_height_cm:.1f}\n"
        f"anchor_rmse_max_cm={args.anchor_rmse_max_cm:.2f}\n"
        f"max_height_delta_cm={args.max_height_delta_cm:.1f}\n\n"
        f"{table}\n\n"
        "whole-cloud comparison: lower-stem anchor RMSE vs whole-cloud transform "
        "evaluated on the same anchor region\n"
    )
    for r in results:
        report += (
            f"{r.date}: lower={r.anchor_rmse:.3f} cm, "
            f"whole={r.whole_anchor_rmse:.3f} cm\n"
        )
    report += finding + "\n"
    report += f"\naccum_npy={accum_path}\nqc_png={qc_path}\n"
    report_path.write_text(report)

    print("\n" + table, flush=True)
    if finding:
        print(finding, flush=True)
    rmse_vals = [r.anchor_rmse for r in results if np.isfinite(r.anchor_rmse)]
    included = [r.date for r in results if r.included]
    print(f"\nanchor-RMSE range: {min(rmse_vals):.3f}..{max(rmse_vals):.3f} cm",
          flush=True)
    print(f"included dates: {', '.join(included)}", flush=True)
    print(f"accumulated cloud -> {accum_path} shape={accum.shape} dtype={accum.dtype}",
          flush=True)
    print(f"QC render -> {qc_path}", flush=True)
    print(f"report -> {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
