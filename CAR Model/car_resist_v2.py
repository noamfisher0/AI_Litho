#!/usr/bin/env python3
"""CAR resist evaluation using the DCT-based resist_model solver.

Wraps the professor's resist_model package (DCT Neumann BCs, palindromic
reaction splitting, explicit protection field) for evaluation on LithoBench
aerial images.  Follows the same CLI / sweep / output conventions as the
existing car_resist_evaluation.py.

Usage (from the cameleon repo root):
    python scripts/lithobench/car_resist_v2.py \
        --dataset_name LithoBench-StdMetal \
        --dataset_path /cluster/work/math/camlab-data/tmp_share/LithoBenchData \
        --output_dir   /cluster/scratch/nfisher/results/direction3_v2/default \
        --n_vis 8 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import the professor's resist_model package.
# It lives at  scripts/lithobench/resist_model/  relative to the repo.
# This script lives at  scripts/lithobench/car_resist_v2.py.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from resist_model.config import Grid, PEBParameters, SimulationConfig
from resist_model.exposure import acid_from_dose
from resist_model.solver import simulate
from resist_model.development import (
    DevelopmentParameters,
    mack_rate,
    vertical_development,
)

# ---------------------------------------------------------------------------
# Dataset loading  (NetCDF, same pattern as the existing evaluation scripts)
# ---------------------------------------------------------------------------
import netCDF4 as nc


def load_dataset(path: str, name: str, split: str = "test"):
    """Return a netCDF4 Dataset handle and the index range for *split*.

    StdMetal / StdContact are pure test sets (no train split).
    MetalSet / ViaSet use a 90/10 train/test split.
    """
    ds = nc.Dataset(Path(path) / f"{name}.nc", "r")
    n_total = ds.dimensions["member"].size

    # StdMetal (271) and StdContact (165) are evaluation-only
    is_std = "Std" in name
    if is_std or split == "all":
        return ds, 0, n_total

    n_train = int(0.9 * n_total)
    if split == "test":
        return ds, n_train, n_total
    else:
        return ds, 0, n_train


# ---------------------------------------------------------------------------
# Core pipeline  (single tile, no batch dimension)
# ---------------------------------------------------------------------------

def build_config(args) -> SimulationConfig:
    """Construct a SimulationConfig from CLI args."""
    grid = Grid(
        nx=args.tile_size,
        ny=args.tile_size,
        lx_nm=float(args.tile_size) * args.pixel_nm,
        ly_nm=float(args.tile_size) * args.pixel_nm,
    )
    chemistry = PEBParameters(
        acid_diffusivity_nm2_s=args.acid_diff,
        quencher_diffusivity_nm2_s=args.quencher_diff,
        k_quench_s=args.k_quench,
        k_loss_s=args.k_loss,
        k_deprotection_s=args.k_deprot,
    )
    return SimulationConfig(
        grid=grid,
        chemistry=chemistry,
        peb_time_s=args.peb_time,
        max_step_s=args.max_step,
    )


def run_tile(
    aerial: torch.Tensor,          # (H, W), float64, I ∈ [0, 1]
    config: SimulationConfig,
    dev_params: DevelopmentParameters,
    e_peak: float,
    q0: float,
    acid_yield: float,
    exposure_c: float,
) -> dict[str, torch.Tensor]:
    """Run the full pipeline on a single tile.  Returns dict of 2-D fields."""

    # 1. Exposure: I → dose → acid
    dose = e_peak * aerial
    exposure = acid_from_dose(
        dose,
        exposure_constant_cm2_mj=exposure_c,
        acid_yield=acid_yield,
    )
    a0 = exposure.acid
    q0_field = torch.full_like(a0, q0)
    p0 = torch.ones_like(a0)

    # 2. PEB solve  (returns final acid, quencher, protection)
    t0 = time.perf_counter()
    a_final, q_final, p_final = simulate((a0, q0_field, p0), config)
    peb_ms = (time.perf_counter() - t0) * 1e3

    # 3. Development
    rate = mack_rate(p_final, dev_params)
    rate_norm = (rate - dev_params.rate_min_nm_s) / (dev_params.rate_max_nm_s - dev_params.rate_min_nm_s)
    thickness_nm, remaining_frac = vertical_development(p_final, dev_params)
    # LithoBench convention: printed=1 where resist is removed (features).
    # Positive-tone: exposed → dissolved → cleared → feature present.
    printed = (remaining_frac <= 0.0).to(aerial.dtype)

    # Background-corrected continuous output.
    # Even fully protected resist (p=1) dissolves at r_min, giving a nonzero floor
    # in (1 - remaining_frac). Subtract that analytically known constant and rescale
    # to [0, 1] so the background matches the LB continuous resist (which is zero).
    floor = (dev_params.rate_min_nm_s * dev_params.development_time_s
             / dev_params.film_thickness_nm)
    if floor >= 1.0:
        print(f"WARNING: floor={floor:.3f} >= 1.0 — fully protected resist erodes "
              f"completely. Parameters are unphysical.")
    raw = 1.0 - remaining_frac
    resist_continuous = torch.clamp((raw - floor) / (1.0 - floor), 0.0, 1.0)

    return {
        "aerial": aerial,
        "acid_initial": a0,
        "acid_final": a_final,
        "quencher_final": q_final,
        "protection": p_final,
        "deprotection": 1.0 - p_final,        # (1-p): same polarity as LB resist
        "rate": rate,
        "rate_norm": rate_norm,                # (r-r_min)/(r_max-r_min) ∈ [0,1]
        "remaining_frac": remaining_frac,
        "car_continuous": 1.0 - remaining_frac,
        "resist_continuous": resist_continuous, # background-corrected, matches LB polarity
        "printed": printed,
        "peb_ms": peb_ms,
    }


# ---------------------------------------------------------------------------
# Calibration: find t_dev such that I = I_th → print boundary
# ---------------------------------------------------------------------------

def calibrate_tdev(
    config: SimulationConfig,
    dev_params: DevelopmentParameters,
    e_peak: float,
    q0: float,
    acid_yield: float,
    exposure_c: float,
    i_th: float = 0.225,
    device: torch.device | str | None = None,
) -> float:
    """Find the development time where a pixel at I = I_th is just cleared.

    Returns t_dev_calibrated (seconds).
    """
    # Run a tiny uniform 4×4 field at I = I_th through the full PEB.
    cal_grid = Grid(nx=4, ny=4, lx_nm=4.0 * config.grid.dx_nm,
                    ly_nm=4.0 * config.grid.dy_nm)
    cal_config = SimulationConfig(
        grid=cal_grid,
        chemistry=config.chemistry,
        peb_time_s=config.peb_time_s,
        max_step_s=config.max_step_s,
    )
    aerial = torch.full((4, 4), i_th, dtype=torch.float64, device=device)
    dose = e_peak * aerial
    exposure = acid_from_dose(dose, exposure_constant_cm2_mj=exposure_c,
                              acid_yield=acid_yield)
    a0 = exposure.acid
    q0_f = torch.full_like(a0, q0)
    p0 = torch.ones_like(a0)
    _, _, p_cal = simulate((a0, q0_f, p0), cal_config)

    # The rate at this protection level
    rate_at_th = mack_rate(p_cal, dev_params)
    r_th = float(rate_at_th[0, 0].item())       # uniform field → all pixels equal

    if r_th <= 0:
        print(f"  WARNING: rate at I_th={i_th} is {r_th:.4e} — threshold pixel "
              f"does not develop.  Returning default t_dev.")
        return dev_params.development_time_s

    # t_dev such that remaining = h0 - r * t = 0  →  t = h0 / r
    t_cal = dev_params.film_thickness_nm / r_th
    return t_cal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.5):
    p = (pred > thresh).float()
    g = (gt > thresh).float()
    inter = (p * g).sum()
    union = ((p + g) > 0).float().sum()
    return (inter / union).item() if union > 0 else 1.0


def compute_mse(pred: torch.Tensor, gt: torch.Tensor):
    return ((pred - gt) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_vis(
    results: dict[str, torch.Tensor],
    gt_printed: torch.Tensor,
    gt_resist: torch.Tensor,
    out_path: Path,
    idx: int,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row 1: physics pipeline internals
    # Row 2: continuous latent images + binary printed
    # Row 3: LB references + differences
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)

    row1 = [
        ("aerial",         "Aerial image $I$",       "viridis", None),
        ("acid_initial",   "Initial acid $a_0$",      "viridis", None),
        ("acid_final",     "Acid after PEB",          "viridis", None),
        ("quencher_final", "Quencher after PEB",      "viridis", None),
    ]
    row2 = [
        ("protection",      "Protection $p$",              "viridis", (0, 1)),
        ("rate_norm",       "Rate $(r-r_{min})/\\Delta r$", "viridis", (0, 1)),
        ("resist_continuous", "Resist continuous (corr.)", "viridis", (0, 1)),
        ("printed",         "CAR printed (bin.)",          "gray",    (0, 1)),
    ]

    for ax, (key, title, cmap, vlim) in zip(axes[0], row1):
        f = results[key].detach().cpu().numpy()
        kw = {"cmap": cmap, "origin": "lower"}
        if vlim is not None:
            kw["vmin"], kw["vmax"] = vlim
        im = ax.imshow(f, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)

    for ax, (key, title, cmap, vlim) in zip(axes[1], row2):
        f = results[key].detach().cpu().numpy()
        kw = {"cmap": cmap, "origin": "lower"}
        if vlim is not None:
            kw["vmin"], kw["vmax"] = vlim
        im = ax.imshow(f, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)

    # Row 3: LithoBench references and differences
    # LB resist (continuous)
    ax = axes[2, 0]
    im = ax.imshow(gt_resist.cpu().numpy(), cmap="viridis", origin="lower",
                   vmin=0, vmax=1)
    ax.set_title("LB resist (continuous)", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # LB printed (binary)
    ax = axes[2, 1]
    im = ax.imshow(gt_printed.cpu().numpy(), cmap="gray", origin="lower",
                   vmin=0, vmax=1)
    ax.set_title("LB printed (binary)", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # |CAR printed − LB printed|
    ax = axes[2, 2]
    diff_bin = (results["printed"] - gt_printed).abs().cpu().numpy()
    im = ax.imshow(diff_bin, cmap="Reds", origin="lower", vmin=0, vmax=1)
    ax.set_title("|CAR − LB printed|", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # |(1 − remaining_frac) − LB resist|  (continuous domain gap, matches MSE metric)
    ax = axes[2, 3]
    diff_cont = (results["resist_continuous"] - gt_resist).abs().cpu().numpy()
    im = ax.imshow(diff_cont, cmap="Reds", origin="lower", vmin=0, vmax=0.5)
    ax.set_title("|resist\\_continuous − LB resist|", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Sample {idx}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

SWEEP_MAP = {
    "e_peak":       "e_peak",
    "acid_diff":    "acid_diff",
    "quencher_diff":"quencher_diff",
    "k_quench":     "k_quench",
    "k_loss":       "k_loss",
    "k_deprot":     "k_deprot",
    "q0":           "q0",
    "peb_time":     "peb_time",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="CAR resist evaluation on LithoBench (DCT-based resist_model solver)")

    # Dataset
    p.add_argument("--dataset_name", type=str, default="LithoBench-StdMetal")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--split",        type=str, default="test")
    p.add_argument("--max_samples",  type=int, default=None,
                   help="Cap on number of tiles to process (None = all)")

    # Grid
    p.add_argument("--tile_size",  type=int,   default=2048)
    p.add_argument("--pixel_nm",   type=float, default=1.0)

    # Exposure
    p.add_argument("--e_peak",     type=float, default=90.0,
                   help="Peak dose (mJ/cm²)")
    p.add_argument("--exposure_c", type=float, default=0.05,
                   help="Dill C exposure constant (cm²/mJ)")
    p.add_argument("--acid_yield", type=float, default=0.85,
                   help="Acid generation yield η_A ∈ [0,1]")

    # PEB chemistry
    p.add_argument("--acid_diff",     type=float, default=16.1,
                   help="Acid diffusivity D_A (nm²/s)")
    p.add_argument("--quencher_diff", type=float, default=4.6,
                   help="Quencher diffusivity D_Q (nm²/s)")
    p.add_argument("--k_quench",      type=float, default=1000.0,
                   help="Neutralization rate κ_Q (1/s)")
    p.add_argument("--k_loss",        type=float, default=0.001,
                   help="Volumetric acid loss k_L (1/s)")
    p.add_argument("--k_deprot",      type=float, default=0.01,
                   help="Deprotection rate κ_dep (1/s)")
    p.add_argument("--q0",            type=float, default=0.25,
                   help="Initial quencher loading (normalized)")
    p.add_argument("--peb_time",      type=float, default=60.0,
                   help="PEB duration (s)")
    p.add_argument("--max_step",      type=float, default=5.0,
                   help="Maximum PEB time step (s)")

    # Development
    p.add_argument("--r_min",     type=float, default=1.55,  help="nm/s")
    p.add_argument("--r_max",     type=float, default=150.0, help="nm/s")
    p.add_argument("--mack_a",    type=float, default=0.003)
    p.add_argument("--mack_n",    type=float, default=5.6)
    p.add_argument("--t_dev",     type=float, default=30.0,
                   help="Development time (s).  Fixed by default.")
    p.add_argument("--film_thickness", type=float, default=100.0, help="nm")

    # Calibration
    p.add_argument("--calibrate_tdev", action="store_true", default=False,
                   help="Auto-calibrate t_dev so I=I_th is at print edge")
    p.add_argument("--i_th",          type=float, default=0.225)

    # Sweep
    p.add_argument("--sweep_param",  type=str, default=None,
                   choices=list(SWEEP_MAP.keys()),
                   help="Parameter to sweep")
    p.add_argument("--sweep_values", type=str, default=None,
                   help="Comma-separated sweep values")
    p.add_argument("--sweep_vis",    type=str, default=None,
                   help="Comma-separated subset of sweep values for vis")

    # Output
    p.add_argument("--n_vis",   type=int, default=8)
    p.add_argument("--device",  type=str, default="cpu",
                   help="'cpu' or 'cuda'")
    p.add_argument("--dtype",   type=str, default="float32",
                   choices=["float32", "float64"])

    return p.parse_args()


def run_sweep_point(
    args,
    ds,
    idx_start: int,
    idx_end: int,
    sweep_val: float | None,
    sweep_label: str,
    out_dir: Path,
    vis_this_point: bool,
):
    """Evaluate one operating point.  Returns a dict of aggregate metrics."""

    # Apply sweep override
    if sweep_val is not None:
        setattr(args, SWEEP_MAP[args.sweep_param], sweep_val)

    # Build configs
    sim_config = build_config(args)
    dev_params = DevelopmentParameters(
        rate_min_nm_s=args.r_min,
        rate_max_nm_s=args.r_max,
        mack_a=args.mack_a,
        selectivity_n=args.mack_n,
        development_time_s=args.t_dev,
        film_thickness_nm=args.film_thickness,
    )
    t_cal = args.t_dev

    # Optionally calibrate t_dev to anchor print edge at I = I_th
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
        print(f"  Calibrated t_dev = {t_cal:.4f} s  "
              f"(rate at I_th={args.i_th}: pixel just clears)")
    else:
        print(f"  Fixed t_dev = {t_cal:.4f} s (no calibration)")

    # Process tiles
    n_tiles = idx_end - idx_start
    if args.max_samples is not None:
        n_tiles = min(n_tiles, args.max_samples)

    ious, mses_resist, peb_times = [], [], []
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    vis_dir = out_dir / f"vis_{sweep_label}"

    pbar = tqdm(range(n_tiles), desc="    Tiles", unit="tile",
                dynamic_ncols=True)
    for i in pbar:
        idx = idx_start + i

        # Load aerial (litho), printed GT, and resist (continuous) from NetCDF
        aerial_np = ds.variables["litho"][idx].astype(np.float64)
        gt_printed_np = ds.variables["printed"][idx].astype(np.float64)
        gt_resist_np = ds.variables["resist"][idx].astype(np.float64)

        aerial = torch.tensor(aerial_np, dtype=dtype, device=device)
        gt_printed = torch.tensor(gt_printed_np, dtype=dtype, device=device)
        gt_resist = torch.tensor(gt_resist_np, dtype=dtype, device=device)

        # Run pipeline
        results = run_tile(
            aerial, sim_config, dev_params,
            args.e_peak, args.q0, args.acid_yield, args.exposure_c,
        )

        # Metrics
        iou = compute_iou(results["printed"], gt_printed)
        mse_resist = compute_mse(results["resist_continuous"], gt_resist)

        ious.append(iou)
        mses_resist.append(mse_resist)
        peb_times.append(results["peb_ms"])

        pbar.set_postfix(IOU=f"{np.mean(ious):.4f}",
                         MSE=f"{np.mean(mses_resist):.5f}",
                         ms=f"{results['peb_ms']:.0f}")

        # Visualization
        if vis_this_point and i < args.n_vis:
            save_vis(results, gt_printed, gt_resist,
                     vis_dir / f"sample_{idx:04d}.png", idx)

    return {
        "sweep_value": sweep_val if sweep_val is not None else "default",
        "t_dev": t_cal,
        "iou_mean": np.mean(ious),
        "iou_std": np.std(ious),
        "mse_resist_mean": np.mean(mses_resist),
        "peb_ms_mean": np.mean(peb_times),
        "n_tiles": n_tiles,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all parameter values for reproducibility
    params_path = out_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Params:   {params_path}")

    print(f"Dataset:  {args.dataset_name}")
    print(f"Output:   {out_dir}")
    print(f"E_peak:   {args.e_peak} mJ/cm²")
    print(f"PEB:      D_A={args.acid_diff}, D_Q={args.quencher_diff}, "
          f"κ_Q={args.k_quench}, k_L={args.k_loss}, κ_dep={args.k_deprot}")
    print(f"Quencher: q0={args.q0}")
    print(f"Solver:   t_PEB={args.peb_time}s, Δt_max={args.max_step}s "
          f"→ {math.ceil(args.peb_time / args.max_step)} steps")
    print()

    # Load dataset
    ds, idx_start, idx_end = load_dataset(
        args.dataset_path, args.dataset_name, args.split)
    print(f"Loaded {args.dataset_name}: samples [{idx_start}, {idx_end}) "
          f"({idx_end - idx_start} tiles)")

    # Determine sweep
    if args.sweep_param and args.sweep_values:
        sweep_vals = [float(v) for v in args.sweep_values.split(",")]
        vis_vals = set()
        if args.sweep_vis:
            vis_vals = {float(v) for v in args.sweep_vis.split(",")}
        else:
            vis_vals = set(sweep_vals)
    else:
        sweep_vals = [None]
        vis_vals = {None}

    # Run sweep
    all_results = []
    for sv in sweep_vals:
        label = f"{sv}" if sv is not None else "default"
        print(f"\n{'='*60}")
        if sv is not None:
            print(f"  Sweep: {args.sweep_param} = {sv}")
        else:
            print(f"  Running default operating point")
        print(f"{'='*60}")

        result = run_sweep_point(
            args, ds, idx_start, idx_end,
            sv, label, out_dir, vis_this_point=(sv in vis_vals),
        )
        all_results.append(result)

    # Write metrics CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nMetrics saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*60}")
    param_name = args.sweep_param or "point"
    print(f"  {param_name:>12s} | t_dev  | IOU (mean±std)    | MSE(resist) | PEB ms")
    print(f"  {'-'*12} | ------ | ----------------- | ----------- | ------")
    for r in all_results:
        print(f"  {str(r['sweep_value']):>12s} | "
              f"{r['t_dev']:6.2f} | "
              f"{r['iou_mean']:.4f} ± {r['iou_std']:.4f}  | "
              f"{r['mse_resist_mean']:.5f}     | "
              f"{r['peb_ms_mean']:.0f}")
    print(f"{'='*72}")

    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
