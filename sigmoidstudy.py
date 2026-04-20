"""
sigmoidstudy.py
===============
Empirically recover the sigmoid parameters (alpha, Ith) from LithoBench
paired (aerial, resist) ground-truth tiles.

The resist image is defined as:
    Z(x,y) = sigma_Z(I(x,y)) = 1 / (1 + exp(-alpha * (I(x,y) - Ith)))

We fit this two-parameter sigmoid to paired pixel samples across tiles,
then visualise the quality of the fit and report the implied threshold
for printed binarisation (Ith, corresponding to Z=0.5).

Usage
-----
# Full run on MetalSet (aerial + resist)
python sigmoidstudy.py --data-root /path/to/lithobench --datasets MetalSet

# Resume a partial run
python sigmoidstudy.py --data-root /path/to/lithobench --datasets MetalSet --resume

# Multiple datasets
python sigmoidstudy.py --data-root /path/to/lithobench \
    --datasets MetalSet ViaSet --output-dir ./sigmoidstudy_results

# Limit pixels per tile for speed (subsampling)
python sigmoidstudy.py --data-root /path/to/lithobench \
    --datasets MetalSet --max-pixels-per-tile 1024
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = ["MetalSet", "ViaSet", "StdMetal", "StdContact"]

DATA_DICT = {
    "MetalSet":   {"aerial": "litho",   "resist": "resist"},
    "ViaSet":     {"aerial": "litho",   "resist": "resist"},
    "StdMetal":   {"aerial": "litho",   "resist": "resist"},
    "StdContact": {"aerial": "litho",   "resist": "resist"},
}

COMPLETED_CSV = "completed_tiles.csv"
RESULTS_JSON  = "sigmoid_results.json"

# ──────────────────────────────────────────────────────────────────────────────
# Sigmoid model
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(I, alpha, Ith):
    """LithoBench resist model: Z = 1 / (1 + exp(-alpha * (I - Ith)))"""
    return 1.0 / (1.0 + np.exp(-alpha * (I - Ith)))


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tile(path: Path):
    """Load a single tile. Supports PNG, NPY, PKL. Returns float32 array in [0,1] or None."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".png":
            from PIL import Image
            img = Image.open(path).convert("L")   # greyscale
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr
        elif suffix == ".npy":
            arr = np.load(path).astype(np.float32)
            return arr
        else:  # pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, np.ndarray):
                return data.astype(np.float32)
            if isinstance(data, dict):
                for key in ("image", "data", "array"):
                    if key in data:
                        return np.array(data[key], dtype=np.float32)
            return np.array(data, dtype=np.float32)
    except Exception:
        return None


def find_tile_pairs(data_root: Path, dataset: str):
    """
    Find matched (aerial, resist) tile file pairs for a dataset.
    Returns list of (aerial_path, resist_path, tile_id) tuples.
    """
    aerial_key  = DATA_DICT[dataset]["aerial"]
    resist_key  = DATA_DICT[dataset]["resist"]

    aerial_dir = data_root / dataset / aerial_key
    resist_dir = data_root / dataset / resist_key

    if not aerial_dir.exists():
        print(f"  [WARN] Aerial dir not found: {aerial_dir}")
        return []
    if not resist_dir.exists():
        print(f"  [WARN] Resist dir not found: {resist_dir}")
        return []

    aerial_files = {p.stem: p for p in sorted(aerial_dir.glob("*.pkl"))}
    resist_files = {p.stem: p for p in sorted(resist_dir.glob("*.pkl"))}

    common = sorted(set(aerial_files.keys()) & set(resist_files.keys()))
    pairs  = [(aerial_files[k], resist_files[k], k) for k in common]

    if not pairs:
        # Try .npy files
        aerial_files = {p.stem: p for p in sorted(aerial_dir.glob("*.npy"))}
        resist_files = {p.stem: p for p in sorted(resist_dir.glob("*.npy"))}
        common = sorted(set(aerial_files.keys()) & set(resist_files.keys()))
        pairs  = [(aerial_files[k], resist_files[k], k) for k in common]

    if not pairs:
        # Try .png files
        aerial_files = {p.stem: p for p in sorted(aerial_dir.glob("*.png"))}
        resist_files = {p.stem: p for p in sorted(resist_dir.glob("*.png"))}
        common = sorted(set(aerial_files.keys()) & set(resist_files.keys()))
        pairs  = [(aerial_files[k], resist_files[k], k) for k in common]

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Resume mechanism
# ──────────────────────────────────────────────────────────────────────────────

def load_completed(output_dir: Path, dataset: str):
    """Return set of already-processed tile IDs for this dataset."""
    csv_path = output_dir / dataset / COMPLETED_CSV
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["tile_id"].astype(str).tolist())


def save_completed(output_dir: Path, dataset: str, records: list):
    """Append tile records to the completed CSV."""
    csv_path = output_dir / dataset / COMPLETED_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Per-tile fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_tile(aerial: np.ndarray, resist: np.ndarray,
             max_pixels: int = None, rng=None):
    """
    Fit sigmoid parameters to one (aerial, resist) tile pair.
    Returns dict with alpha, Ith, r2, rmse, n_pixels, or None on failure.
    """
    I = aerial.ravel().astype(np.float64)
    Z = resist.ravel().astype(np.float64)

    # Remove border pixels that are exactly 0 in both (uninformative background)
    mask = ~((I < 1e-6) & (Z < 1e-6))
    I, Z = I[mask], Z[mask]

    if len(I) < 50:
        return None

    # Optional subsampling
    if max_pixels is not None and len(I) > max_pixels:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(I), size=max_pixels, replace=False)
        I, Z = I[idx], Z[idx]

    # Initial guess: alpha=100, Ith=median of aerial
    p0 = [100.0, float(np.median(I))]
    bounds = ([1.0, 0.0], [1e5, 1.0])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(sigmoid, I, Z, p0=p0, bounds=bounds,
                                maxfev=5000)
        alpha, Ith = popt

        Z_pred = sigmoid(I, alpha, Ith)
        ss_res = np.sum((Z - Z_pred) ** 2)
        ss_tot = np.sum((Z - Z.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse   = np.sqrt(np.mean((Z - Z_pred) ** 2))

        return {
            "alpha":    float(alpha),
            "Ith":      float(Ith),
            "r2":       float(r2),
            "rmse":     float(rmse),
            "n_pixels": int(len(I)),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_distributions(df: pd.DataFrame, dataset: str,
                                 output_dir: Path):
    """Histogram of alpha and Ith across all tiles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{dataset} — Sigmoid Parameter Distributions", fontsize=14)

    # Alpha
    ax = axes[0]
    ax.hist(df["alpha"], bins=50, color="#2196F3", edgecolor="white",
            linewidth=0.5)
    ax.axvline(df["alpha"].median(), color="red", linestyle="--",
               label=f"Median={df['alpha'].median():.1f}")
    ax.set_xlabel("α (steepness)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("α Distribution")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ith
    ax = axes[1]
    ax.hist(df["Ith"], bins=50, color="#4CAF50", edgecolor="white",
            linewidth=0.5)
    ax.axvline(df["Ith"].median(), color="red", linestyle="--",
               label=f"Median={df['Ith'].median():.4f}")
    ax.set_xlabel("I_th (intensity threshold)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("I_th Distribution")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[2]
    ax.hist(df["r2"], bins=50, color="#FF9800", edgecolor="white",
            linewidth=0.5)
    ax.axvline(df["r2"].median(), color="red", linestyle="--",
               label=f"Median R²={df['r2'].median():.4f}")
    ax.set_xlabel("R² (goodness of fit)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Per-tile R² Distribution")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / dataset / "parameter_distributions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_global_fit(all_aerial: np.ndarray, all_resist: np.ndarray,
                    alpha: float, Ith: float, dataset: str,
                    output_dir: Path):
    """Scatter plot of aerial vs resist pixels with fitted sigmoid overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{dataset} — Global Sigmoid Fit  "
                 f"(α={alpha:.1f}, I_th={Ith:.4f})", fontsize=14)

    # Subsample for display (max 50k points)
    rng = np.random.default_rng(0)
    n   = min(50_000, len(all_aerial))
    idx = rng.choice(len(all_aerial), size=n, replace=False)
    I_s = all_aerial[idx]
    Z_s = all_resist[idx]

    # Scatter
    ax = axes[0]
    ax.scatter(I_s, Z_s, s=0.5, alpha=0.3, color="#2196F3", rasterized=True)
    I_range = np.linspace(all_aerial.min(), all_aerial.max(), 500)
    Z_fit   = sigmoid(I_range, alpha, Ith)
    ax.plot(I_range, Z_fit, color="red", linewidth=2,
            label=f"Fit: α={alpha:.1f}, I_th={Ith:.4f}")
    ax.axvline(Ith, color="orange", linestyle=":", linewidth=1.5,
               label=f"I_th (Z=0.5 boundary)")
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Aerial intensity I", fontsize=12)
    ax.set_ylabel("Resist value Z", fontsize=12)
    ax.set_title("Aerial vs Resist (sampled pixels)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[1]
    Z_pred = sigmoid(all_aerial, alpha, Ith)
    residuals = all_resist - Z_pred
    ax.hist(residuals, bins=100, color="#9C27B0", edgecolor="white",
            linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Z_gt - Z_fit)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distribution\n"
                 f"RMSE={np.sqrt(np.mean(residuals**2)):.5f}, "
                 f"Bias={residuals.mean():.5f}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / dataset / "global_fit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_alpha_vs_Ith(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Scatter of alpha vs Ith coloured by R² to check parameter coupling."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["Ith"], df["alpha"], c=df["r2"], cmap="viridis",
                    s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="R²")
    ax.set_xlabel("I_th", fontsize=12)
    ax.set_ylabel("α", fontsize=12)
    ax.set_title(f"{dataset} — α vs I_th (coloured by R²)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / dataset / "alpha_vs_Ith.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_example_tiles(example_records: list, dataset: str, output_dir: Path):
    """
    Show a grid of example tiles: aerial | resist GT | sigmoid fit | residual.
    example_records: list of dicts with keys aerial, resist, alpha, Ith, r2
    """
    n = len(example_records)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"{dataset} — Example Tile Fits", fontsize=14)
    col_titles = ["Aerial (I)", "Resist GT (Z)", "Sigmoid Fit", "Residual"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")

    for i, rec in enumerate(example_records):
        aerial  = rec["aerial"]
        resist  = rec["resist"]
        Z_fit   = sigmoid(aerial, rec["alpha"], rec["Ith"]).astype(np.float32)
        residual = resist - Z_fit

        vmin_a, vmax_a = aerial.min(), aerial.max()

        im0 = axes[i, 0].imshow(aerial,   cmap="gray",
                                 vmin=vmin_a, vmax=vmax_a)
        im1 = axes[i, 1].imshow(resist,   cmap="gray", vmin=0, vmax=1)
        im2 = axes[i, 2].imshow(Z_fit,    cmap="gray", vmin=0, vmax=1)
        im3 = axes[i, 3].imshow(residual, cmap="RdBu",
                                 vmin=-0.2, vmax=0.2)

        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

        label = (f"α={rec['alpha']:.0f}  I_th={rec['Ith']:.4f}  "
                 f"R²={rec['r2']:.4f}")
        axes[i, 0].set_ylabel(label, fontsize=9)

        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    out = output_dir / dataset / "example_tiles.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cross_dataset_summary(all_results: dict, output_dir: Path):
    """Bar chart comparing alpha and Ith across datasets."""
    datasets = list(all_results.keys())
    if len(datasets) < 2:
        return

    alphas = [all_results[d]["global_alpha"]  for d in datasets]
    Iths   = [all_results[d]["global_Ith"]    for d in datasets]
    alpha_stds = [all_results[d]["std_alpha"] for d in datasets]
    Ith_stds   = [all_results[d]["std_Ith"]   for d in datasets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Dataset Sigmoid Parameter Comparison", fontsize=14)

    x = np.arange(len(datasets))
    axes[0].bar(x, alphas, yerr=alpha_stds, capsize=5,
                color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(datasets)],
                alpha=0.8, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, fontsize=11)
    axes[0].set_ylabel("α (steepness)", fontsize=12)
    axes[0].set_title("α per Dataset")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, Iths, yerr=Ith_stds, capsize=5,
                color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(datasets)],
                alpha=0.8, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, fontsize=11)
    axes[1].set_ylabel("I_th (intensity threshold)", fontsize=12)
    axes[1].set_title("I_th per Dataset")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_dir / "cross_dataset_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main per-dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset: str, data_root: Path, output_dir: Path,
                resume: bool, max_pixels: int, n_examples: int):

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    pairs = find_tile_pairs(data_root, dataset)
    if not pairs:
        print(f"  [SKIP] No tile pairs found for {dataset}")
        return None

    print(f"  Found {len(pairs)} tile pairs")

    completed = load_completed(output_dir, dataset) if resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} tiles already processed")

    rng = np.random.default_rng(42)
    batch_records = []
    all_results   = []
    example_data  = []   # store raw arrays for a few tiles

    # Accumulators for global fit (reservoir sample to keep memory bounded)
    MAX_GLOBAL_PIXELS = 5_000_000
    global_I = []
    global_Z = []
    global_count = 0

    todo = [(a, r, tid) for a, r, tid in pairs if tid not in completed]
    print(f"  To process: {len(todo)} tiles")

    for aerial_path, resist_path, tile_id in tqdm(
            todo, desc=f"{dataset}", unit="tile", dynamic_ncols=True):

        aerial = load_tile(aerial_path)
        resist = load_tile(resist_path)

        if aerial is None or resist is None:
            continue
        if aerial.shape != resist.shape:
            continue

        result = fit_tile(aerial, resist, max_pixels=max_pixels, rng=rng)

        if result is not None:
            result["tile_id"]  = tile_id
            result["dataset"]  = dataset
            all_results.append(result)
            batch_records.append(result)

            # Accumulate pixels for global fit (bounded)
            if global_count < MAX_GLOBAL_PIXELS:
                I_flat = aerial.ravel().astype(np.float64)
                Z_flat = resist.ravel().astype(np.float64)
                mask   = ~((I_flat < 1e-6) & (Z_flat < 1e-6))
                I_flat, Z_flat = I_flat[mask], Z_flat[mask]
                if max_pixels and len(I_flat) > max_pixels:
                    idx = rng.choice(len(I_flat), max_pixels, replace=False)
                    I_flat, Z_flat = I_flat[idx], Z_flat[idx]
                global_I.append(I_flat)
                global_Z.append(Z_flat)
                global_count += len(I_flat)

            # Collect example tiles (high R²)
            if len(example_data) < n_examples and result["r2"] > 0.95:
                example_data.append({
                    "aerial": aerial,
                    "resist": resist,
                    "alpha":  result["alpha"],
                    "Ith":    result["Ith"],
                    "r2":     result["r2"],
                })

        # Save progress every 500 tiles
        if len(batch_records) >= 500:
            save_completed(output_dir, dataset, batch_records)
            batch_records = []

    # Save remaining
    if batch_records:
        save_completed(output_dir, dataset, batch_records)

    # Load ALL completed records (including prior runs)
    csv_path = output_dir / dataset / COMPLETED_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        if not all_results:
            print("  [WARN] No results to analyse.")
            return None
        df = pd.DataFrame(all_results)

    # Filter bad fits
    df = df[df["r2"] > 0.5].copy()
    print(f"\n  Tiles with R²>0.5: {len(df)} / {len(pairs)}")

    if len(df) == 0:
        print("  [WARN] No good fits. Check data paths.")
        return None

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n  Per-tile statistics:")
    print(f"    alpha  — median: {df['alpha'].median():.2f}  "
          f"std: {df['alpha'].std():.2f}  "
          f"[{df['alpha'].quantile(0.05):.1f}, {df['alpha'].quantile(0.95):.1f}]")
    print(f"    Ith    — median: {df['Ith'].median():.5f}  "
          f"std: {df['Ith'].std():.5f}  "
          f"[{df['Ith'].quantile(0.05):.5f}, {df['Ith'].quantile(0.95):.5f}]")
    print(f"    R²     — median: {df['r2'].median():.4f}  "
          f"min: {df['r2'].min():.4f}")

    # ── Global fit ────────────────────────────────────────────────────────────
    global_alpha, global_Ith = None, None
    if global_I:
        all_I = np.concatenate(global_I)
        all_Z = np.concatenate(global_Z)
        try:
            p0     = [df["alpha"].median(), df["Ith"].median()]
            bounds = ([1.0, 0.0], [1e5, 1.0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(sigmoid, all_I, all_Z,
                                    p0=p0, bounds=bounds, maxfev=10000)
            global_alpha, global_Ith = float(popt[0]), float(popt[1])
            Z_pred_global = sigmoid(all_I, global_alpha, global_Ith)
            global_rmse   = float(np.sqrt(np.mean((all_Z - Z_pred_global)**2)))
            ss_res = np.sum((all_Z - Z_pred_global)**2)
            ss_tot = np.sum((all_Z - all_Z.mean())**2)
            global_r2 = float(1.0 - ss_res / ss_tot)

            print(f"\n  Global fit (pooled pixels: {len(all_I):,}):")
            print(f"    alpha = {global_alpha:.2f}")
            print(f"    Ith   = {global_Ith:.6f}")
            print(f"    R²    = {global_r2:.6f}")
            print(f"    RMSE  = {global_rmse:.6f}")
            print(f"\n  => Printed binarisation threshold on aerial: I > {global_Ith:.6f}")

            plot_global_fit(all_I, all_Z, global_alpha, global_Ith,
                            dataset, output_dir)
        except Exception as e:
            print(f"  [WARN] Global fit failed: {e}")
            global_alpha = float(df["alpha"].median())
            global_Ith   = float(df["Ith"].median())
            global_rmse  = float("nan")
            global_r2    = float("nan")
    else:
        global_alpha = float(df["alpha"].median())
        global_Ith   = float(df["Ith"].median())
        global_rmse  = float("nan")
        global_r2    = float("nan")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_parameter_distributions(df, dataset, output_dir)
    plot_alpha_vs_Ith(df, dataset, output_dir)

    if example_data:
        plot_example_tiles(example_data, dataset, output_dir)
    else:
        # Fall back: use best R² tiles from df
        print("  [INFO] No example tiles with R²>0.95 found for tile plots.")

    # ── Return summary ────────────────────────────────────────────────────────
    return {
        "dataset":       dataset,
        "n_tiles_total": len(pairs),
        "n_tiles_fit":   len(df),
        "median_alpha":  float(df["alpha"].median()),
        "std_alpha":     float(df["alpha"].std()),
        "median_Ith":    float(df["Ith"].median()),
        "std_Ith":       float(df["Ith"].std()),
        "median_r2":     float(df["r2"].median()),
        "global_alpha":  global_alpha,
        "global_Ith":    global_Ith,
        "global_r2":     global_r2 if "global_r2" in dir() else float("nan"),
        "global_rmse":   global_rmse if "global_rmse" in dir() else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sigmoid parameter study for LithoBench (aerial → resist)."
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root directory of LithoBench (contains MetalSet/, ViaSet/, etc.)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["MetalSet"],
        choices=DATASETS,
        help="Datasets to process (default: MetalSet)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./sigmoidstudy_results"),
        help="Directory for outputs (default: ./sigmoidstudy_results)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previously completed tiles"
    )
    parser.add_argument(
        "--max-pixels-per-tile", type=int, default=None,
        help="Subsample pixels per tile for speed (default: use all pixels)"
    )
    parser.add_argument(
        "--n-examples", type=int, default=4,
        help="Number of example tile visualisations (default: 4)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSigmoid Study")
    print(f"  Data root  : {args.data_root}")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Resume     : {args.resume}")
    print(f"  Max px/tile: {args.max_pixels_per_tile or 'all'}")

    all_results = {}
    t0 = time.time()

    for dataset in args.datasets:
        result = run_dataset(
            dataset      = dataset,
            data_root    = args.data_root,
            output_dir   = args.output_dir,
            resume       = args.resume,
            max_pixels   = args.max_pixels_per_tile,
            n_examples   = args.n_examples,
        )
        if result is not None:
            all_results[dataset] = result

    # ── Cross-dataset summary ─────────────────────────────────────────────────
    if len(all_results) >= 2:
        plot_cross_dataset_summary(all_results, args.output_dir)

    # ── Save JSON results ─────────────────────────────────────────────────────
    results_path = args.output_dir / RESULTS_JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<14} {'α (global)':>12} {'I_th (global)':>14} "
          f"{'R² (global)':>12} {'Tiles fit':>10}")
    print(f"  {'-'*65}")
    for ds, res in all_results.items():
        print(f"  {ds:<14} {res['global_alpha']:>12.2f} "
              f"{res['global_Ith']:>14.6f} "
              f"{res.get('global_r2', float('nan')):>12.6f} "
              f"{res['n_tiles_fit']:>10} / {res['n_tiles_total']}")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Outputs in: {args.output_dir}\n")


if __name__ == "__main__":
    main()
