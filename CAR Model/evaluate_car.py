#!/usr/bin/env python3
"""Full-dataset evaluation of the CAR resist model on LithoBench.

Runs the car_resist_v2.py pipeline on every sample in the dataset and
produces publication-quality metrics and figures.

Usage:
    python scripts/lithobench/evaluate_car.py \
        --dataset_name LithoBench-StdMetal \
        --dataset_path /path/to/LithoBenchData \
        --output_dir   /path/to/eval_output \
        --n_vis 5 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, binary_erosion, disk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import computation functions from car_resist_v2.py (same directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from car_resist_v2 import (
    build_config,
    calibrate_tdev,
    compute_iou,
    load_dataset,
    run_tile,
)
from resist_model.development import DevelopmentParameters


# ---------------------------------------------------------------------------
# New metrics
# ---------------------------------------------------------------------------

def compute_mse_edge_band(
    pred: torch.Tensor,
    gt_cont: torch.Tensor,
    gt_bin: torch.Tensor,
    radius: int = 10,
) -> float:
    """MSE between pred and gt_cont restricted to the ±radius-px band around
    the GT binary boundary.

    Parameters
    ----------
    pred    : resist_continuous field, shape (H, W)
    gt_cont : LB resist (continuous), shape (H, W)
    gt_bin  : LB printed (binary), shape (H, W) — values 0 or 1
    radius  : dilation radius in pixels (default 10)
    """
    gt_bool = gt_bin.cpu().numpy().astype(bool)
    d = disk(radius)
    band = binary_dilation(gt_bool, d) & ~binary_erosion(gt_bool, d)

    pred_np = pred.cpu().numpy()
    gt_np = gt_cont.cpu().numpy()

    if band.sum() == 0:
        return float("nan")

    diff = pred_np[band] - gt_np[band]
    return float((diff ** 2).mean())


def compute_epe(
    car_bin: torch.Tensor,
    lb_bin: torch.Tensor,
) -> tuple[float, float, float]:
    """Edge Placement Error between the CAR and LB GT binary contours.

    For each point on the LB GT contour, find the nearest point on the CAR
    contour.  Returns (mean_epe, max_epe, std_epe) in pixels (= nm, dx=1nm).
    """
    car_np = car_bin.cpu().numpy()
    lb_np = lb_bin.cpu().numpy()

    car_contours = find_contours(car_np, level=0.5)
    lb_contours = find_contours(lb_np, level=0.5)

    if not car_contours or not lb_contours:
        return float("nan"), float("nan"), float("nan")

    car_pts = np.concatenate(car_contours, axis=0)   # (N, 2)
    lb_pts = np.concatenate(lb_contours, axis=0)     # (M, 2)

    tree = cKDTree(car_pts)
    dists, _ = tree.query(lb_pts, k=1)

    return float(dists.mean()), float(dists.max()), float(dists.std())


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_continuous_comparison(
    resist_continuous: torch.Tensor,
    lb_resist: torch.Tensor,
    out_path: Path,
    sample_id: int,
):
    """3-panel continuous resist comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = resist_continuous.cpu().numpy()
    lb = lb_resist.cpu().numpy()
    diff = rc - lb

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    # Panels 1 & 2: shared viridis colorbar [0, 1]
    im0 = axes[0].imshow(lb, cmap="viridis", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("LB Ground Truth", fontsize=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(rc, cmap="viridis", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("CAR Model", fontsize=10)
    axes[1].axis("off")

    # Shared colorbar for panels 1 & 2
    fig.colorbar(im1, ax=[axes[0], axes[1]], shrink=0.8, label="Resist fraction")

    # Panel 3: signed difference, RdBu_r centred at 0
    im2 = axes[2].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (CAR \u2212 LB)", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="\u0394 resist")

    fig.suptitle(f"Sample {sample_id} \u2014 Continuous Resist Comparison", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_contour_overlay(
    car_bin: torch.Tensor,
    lb_bin: torch.Tensor,
    out_path: Path,
    sample_id: int,
):
    """CAR binary as background with LB GT contour overlaid in cyan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    car_np = car_bin.cpu().numpy()
    lb_np = lb_bin.cpu().numpy()
    lb_contours = find_contours(lb_np, level=0.5)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(car_np, cmap="gray", origin="lower", vmin=0, vmax=1)
    for contour in lb_contours:
        ax.plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=1.5)
    ax.set_title(f"Sample {sample_id} \u2014 CAR Printed vs LB Contour", fontsize=10)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full-dataset CAR model evaluation with extended metrics and figures")

    # Dataset
    p.add_argument("--dataset_name", type=str, default="LithoBench-StdMetal")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--split",        type=str, default="test")
    p.add_argument("--max_samples",  type=int, default=None)

    # Grid
    p.add_argument("--tile_size", type=int,   default=2048)
    p.add_argument("--pixel_nm",  type=float, default=1.0)

    # Exposure
    p.add_argument("--e_peak",     type=float, default=90.0)
    p.add_argument("--exposure_c", type=float, default=0.05)
    p.add_argument("--acid_yield", type=float, default=0.85)

    # PEB chemistry
    p.add_argument("--acid_diff",     type=float, default=16.1)
    p.add_argument("--quencher_diff", type=float, default=4.6)
    p.add_argument("--k_quench",      type=float, default=1000.0)
    p.add_argument("--k_loss",        type=float, default=0.001)
    p.add_argument("--k_deprot",      type=float, default=0.01)
    p.add_argument("--q0",            type=float, default=0.25)
    p.add_argument("--peb_time",      type=float, default=60.0)
    p.add_argument("--max_step",      type=float, default=5.0)

    # Development
    p.add_argument("--r_min",          type=float, default=1.55)
    p.add_argument("--r_max",          type=float, default=150.0)
    p.add_argument("--mack_a",         type=float, default=0.003)
    p.add_argument("--mack_n",         type=float, default=5.6)
    p.add_argument("--t_dev",          type=float, default=30.0)
    p.add_argument("--film_thickness", type=float, default=100.0)

    # Calibration
    p.add_argument("--calibrate_tdev", action="store_true", default=False)
    p.add_argument("--i_th",           type=float, default=0.225)

    # Output
    p.add_argument("--n_vis",  type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype",  type=str, default="float32",
                   choices=["float32", "float64"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    # Build simulation config and development parameters
    sim_config = build_config(args)
    dev_params = DevelopmentParameters(
        rate_min_nm_s=args.r_min,
        rate_max_nm_s=args.r_max,
        mack_a=args.mack_a,
        selectivity_n=args.mack_n,
        development_time_s=args.t_dev,
        film_thickness_nm=args.film_thickness,
    )

    if args.calibrate_tdev:
        t_cal = calibrate_tdev(
            sim_config, dev_params,
            args.e_peak, args.q0, args.acid_yield, args.exposure_c, args.i_th,
            device=device,
        )
        dev_params = DevelopmentParameters(
            rate_min_nm_s=args.r_min,
            rate_max_nm_s=args.r_max,
            mack_a=args.mack_a,
            selectivity_n=args.mack_n,
            development_time_s=t_cal,
            film_thickness_nm=args.film_thickness,
        )
        print(f"Calibrated t_dev = {t_cal:.4f} s")
    else:
        print(f"Fixed t_dev = {args.t_dev:.4f} s")

    # Load dataset
    ds, idx_start, idx_end = load_dataset(args.dataset_path, args.dataset_name, args.split)
    n_tiles = idx_end - idx_start
    if args.max_samples is not None:
        n_tiles = min(n_tiles, args.max_samples)
    print(f"Dataset: {args.dataset_name}  samples [{idx_start}, {idx_start + n_tiles})")

    # Per-sample results
    rows: list[dict] = []

    pbar = tqdm(range(n_tiles), desc="Evaluating", unit="tile", dynamic_ncols=True)
    for i in pbar:
        idx = idx_start + i

        aerial_np    = ds.variables["litho"][idx].astype(np.float64)
        gt_printed_np = ds.variables["printed"][idx].astype(np.float64)
        gt_resist_np  = ds.variables["resist"][idx].astype(np.float64)

        aerial     = torch.tensor(aerial_np,     dtype=dtype, device=device)
        gt_printed = torch.tensor(gt_printed_np, dtype=dtype, device=device)
        gt_resist  = torch.tensor(gt_resist_np,  dtype=dtype, device=device)

        results = run_tile(
            aerial, sim_config, dev_params,
            args.e_peak, args.q0, args.acid_yield, args.exposure_c,
        )

        iou          = compute_iou(results["printed"], gt_printed)
        mse_global   = float(((results["resist_continuous"] - gt_resist) ** 2).mean().item())
        mse_edge     = compute_mse_edge_band(results["resist_continuous"], gt_resist, gt_printed)
        epe_mean, epe_max, epe_std = compute_epe(results["printed"], gt_printed)

        rows.append({
            "sample_id":    idx,
            "IOU":          iou,
            "MSE_global":   mse_global,
            "MSE_edge_band": mse_edge,
            "EPE_mean":     epe_mean,
            "EPE_max":      epe_max,
            "EPE_std":      epe_std,
        })

        pbar.set_postfix(
            IOU=f"{iou:.4f}",
            MSE=f"{mse_global:.5f}",
            EPE=f"{epe_mean:.2f}" if not np.isnan(epe_mean) else "nan",
        )

        if i < args.n_vis:
            save_continuous_comparison(
                results["resist_continuous"], gt_resist,
                fig_dir / f"continuous_comparison_{idx:04d}.png", idx,
            )
            save_contour_overlay(
                results["printed"], gt_printed,
                fig_dir / f"contour_overlay_{idx:04d}.png", idx,
            )

    ds.close()

    # ---------------------------------------------------------------------------
    # Write metrics CSV
    # ---------------------------------------------------------------------------
    fieldnames = ["sample_id", "IOU", "MSE_global", "MSE_edge_band",
                  "EPE_mean", "EPE_max", "EPE_std"]

    # Compute summary stats (ignoring NaNs)
    def _mean(key): return float(np.nanmean([r[key] for r in rows]))
    def _std(key):  return float(np.nanstd( [r[key] for r in rows]))

    mean_row = {k: (_mean(k) if k != "sample_id" else "mean") for k in fieldnames}
    std_row  = {k: (_std(k)  if k != "sample_id" else "std")  for k in fieldnames}

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)
    print(f"\nMetrics saved to {csv_path}")

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  {'':>10s} | {'IOU':>8s} | {'MSE_global':>10s} | "
          f"{'MSE_edge':>10s} | {'EPE_mean':>8s} | {'EPE_max':>8s} | {'EPE_std':>8s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")

    def _fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "  nan  "

    for label, row in [("mean", mean_row), ("std", std_row)]:
        print(f"  {label:>10s} | "
              f"{_fmt(row['IOU']):>8s} | "
              f"{_fmt(row['MSE_global']):>10s} | "
              f"{_fmt(row['MSE_edge_band']):>10s} | "
              f"{_fmt(row['EPE_mean']):>8s} | "
              f"{_fmt(row['EPE_max']):>8s} | "
              f"{_fmt(row['EPE_std']):>8s}")

    print(f"{'='*72}")
    print(f"\nFigures saved to {fig_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
