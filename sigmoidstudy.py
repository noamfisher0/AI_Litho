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

Also supports --validate-threshold to verify that aerial > Ith reproduces
the ground truth printed images (IOU, pixel accuracy, visual examples).

Parallelism
-----------
Both the fitting loop and the validation loop use ProcessPoolExecutor for
CPU-bound parallelism. Use --num-workers to control concurrency (default: 4).
Set --num-workers 1 to disable multiprocessing (useful for debugging).

Usage
-----
# Full run on MetalSet with 8 workers
python sigmoidstudy.py --data-root /path/to/lithobench --datasets MetalSet --num-workers 8

# Resume a partial run
python sigmoidstudy.py --data-root /path/to/lithobench --datasets MetalSet --resume

# Validate threshold against printed ground truth
python sigmoidstudy.py --data-root /path/to/lithobench --datasets MetalSet --validate-threshold

# Sigmoid curve plot only (no refitting)
python sigmoidstudy.py --data-root /path/to/lithobench --datasets MetalSet --plot-sigmoid
"""

import argparse
import json
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = ["MetalSet", "ViaSet", "StdMetal", "StdContact"]

DATA_DICT = {
    "MetalSet":   {"aerial": "litho", "resist": "resist", "printed": "printed"},
    "ViaSet":     {"aerial": "litho", "resist": "resist", "printed": "printed"},
    "StdMetal":   {"aerial": "litho", "resist": "resist", "printed": "printed"},
    "StdContact": {"aerial": "litho", "resist": "resist", "printed": "printed"},
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
            img = Image.open(path).convert("L")
            return np.array(img, dtype=np.float32) / 255.0
        elif suffix == ".npy":
            return np.load(path).astype(np.float32)
        else:
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


def find_tile_pairs(data_root: Path, dataset: str, keys: list):
    """
    Find matched tile file tuples for given keys.
    Returns list of (tile_id, {key: path}) tuples.
    Only tiles present in ALL keys are returned.
    """
    dirs = {}
    for key in keys:
        subdir_name = DATA_DICT[dataset][key]
        d = data_root / dataset / subdir_name
        if not d.exists():
            print(f"  [WARN] Dir not found: {d}")
            return []
        dirs[key] = d

    files_per_key = {}
    for key, d in dirs.items():
        for ext in ("*.pkl", "*.npy", "*.png"):
            found = {p.stem: p for p in sorted(d.glob(ext))}
            if found:
                files_per_key[key] = found
                break
        if key not in files_per_key:
            print(f"  [WARN] No files found in {dirs[key]}")
            return []

    common = sorted(set.intersection(*[set(f.keys()) for f in files_per_key.values()]))
    return [(tid, {key: files_per_key[key][tid] for key in keys}) for tid in common]


# ──────────────────────────────────────────────────────────────────────────────
# Resume mechanism
# ──────────────────────────────────────────────────────────────────────────────

def load_completed(output_dir: Path, dataset: str):
    csv_path = output_dir / dataset / COMPLETED_CSV
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["tile_id"].astype(str).tolist())


def save_completed(output_dir: Path, dataset: str, records: list):
    csv_path = output_dir / dataset / COMPLETED_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Worker functions (module-level so they are picklable for multiprocessing)
# ──────────────────────────────────────────────────────────────────────────────

def _fit_worker(args_tuple):
    """
    Worker for ProcessPoolExecutor — fits sigmoid to one tile.
    args_tuple: (tile_id, aerial_path, resist_path, max_pixels, seed)
    Returns (tile_id, result_dict_or_None, I_flat, Z_flat)
    """
    tile_id, aerial_path, resist_path, max_pixels, seed = args_tuple

    aerial = load_tile(Path(aerial_path))
    resist = load_tile(Path(resist_path))

    if aerial is None or resist is None:
        return tile_id, None, None, None
    if aerial.shape != resist.shape:
        return tile_id, None, None, None

    I = aerial.ravel().astype(np.float64)
    Z = resist.ravel().astype(np.float64)
    mask = ~((I < 1e-6) & (Z < 1e-6))
    I, Z = I[mask], Z[mask]

    if len(I) < 50:
        return tile_id, None, None, None

    rng = np.random.default_rng(seed)
    if max_pixels is not None and len(I) > max_pixels:
        idx = rng.choice(len(I), size=max_pixels, replace=False)
        I, Z = I[idx], Z[idx]

    # Save a subsample for the global fit accumulator
    MAX_GLOBAL_SAMPLE = 2048
    if len(I) > MAX_GLOBAL_SAMPLE:
        idx2 = rng.choice(len(I), size=MAX_GLOBAL_SAMPLE, replace=False)
        I_global, Z_global = I[idx2], Z[idx2]
    else:
        I_global, Z_global = I.copy(), Z.copy()

    p0     = [100.0, float(np.median(I))]
    bounds = ([1.0, 0.0], [1e5, 1.0])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(sigmoid, I, Z, p0=p0, bounds=bounds, maxfev=5000)
        alpha, Ith = popt
        Z_pred = sigmoid(I, alpha, Ith)
        ss_res = np.sum((Z - Z_pred) ** 2)
        ss_tot = np.sum((Z - Z.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse   = float(np.sqrt(np.mean((Z - Z_pred) ** 2)))
        result = {"alpha": float(alpha), "Ith": float(Ith),
                  "r2": float(r2), "rmse": rmse, "n_pixels": int(len(I))}
        return tile_id, result, I_global, Z_global
    except Exception:
        return tile_id, None, None, None


def _validate_worker(args_tuple):
    """
    Worker for ProcessPoolExecutor — computes threshold metrics for one tile.
    args_tuple: (tile_id, aerial_path, printed_path, Ith)
    Returns (tile_id, metrics_dict_or_None, aerial_array, printed_array)
    """
    tile_id, aerial_path, printed_path, Ith = args_tuple

    aerial  = load_tile(Path(aerial_path))
    printed = load_tile(Path(printed_path))

    if aerial is None or printed is None:
        return tile_id, None, None, None
    if aerial.shape != printed.shape:
        return tile_id, None, None, None

    pred = (aerial > Ith).astype(np.uint8)
    gt   = (printed > 0.5).astype(np.uint8)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    iou       = tp / (tp + fp + fn + 1e-8)
    pa        = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fpr       = fp / (fp + tn + 1e-8)
    fnr       = fn / (fn + tp + 1e-8)

    metrics = {"iou": iou, "pa": pa, "f1": f1,
               "precision": precision, "recall": recall,
               "fpr": fpr, "fnr": fnr,
               "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    return tile_id, metrics, aerial, printed

def _consistency_worker(args_tuple):
    tid, resist_path, printed_path = args_tuple
    resist  = load_tile(Path(resist_path))
    printed = load_tile(Path(printed_path))
    if resist is None or printed is None:
        return tid, None
    if resist.shape != printed.shape:
        return tid, None

    resist_bin  = (resist  > 0.5).astype(np.float32)
    printed_bin = (printed > 0.5).astype(np.float32)

    mse = float(np.mean((resist_bin - printed_bin) ** 2))
    tp  = int(np.sum((resist_bin == 1) & (printed_bin == 1)))
    fp  = int(np.sum((resist_bin == 1) & (printed_bin == 0)))
    fn  = int(np.sum((resist_bin == 0) & (printed_bin == 1)))
    iou = tp / (tp + fp + fn + 1e-8)

    return tid, {"mse": mse, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations — fitting
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_distributions(df: pd.DataFrame, dataset: str, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{dataset} — Sigmoid Parameter Distributions", fontsize=14)
    specs = [
        ("alpha", "#2196F3", "alpha (steepness)", ".1f"),
        ("Ith",   "#4CAF50", "I_th (intensity threshold)", ".4f"),
        ("r2",    "#FF9800", "R^2 (goodness of fit)", ".4f"),
    ]
    for ax, (col, color, label, fmt) in zip(axes, specs):
        ax.hist(df[col], bins=50, color=color, edgecolor="white", linewidth=0.5)
        med = df[col].median()
        ax.axvline(med, color="red", linestyle="--", label=f"Median={med:{fmt}}")
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{col} Distribution")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / dataset / "parameter_distributions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_global_fit(all_I: np.ndarray, all_Z: np.ndarray,
                    alpha: float, Ith: float, dataset: str, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{dataset} — Global Sigmoid Fit  (alpha={alpha:.1f}, I_th={Ith:.4f})",
                 fontsize=14)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(all_I), size=min(50_000, len(all_I)), replace=False)
    ax = axes[0]
    ax.scatter(all_I[idx], all_Z[idx], s=0.5, alpha=0.3, color="#2196F3", rasterized=True)
    I_range = np.linspace(all_I.min(), all_I.max(), 500)
    ax.plot(I_range, sigmoid(I_range, alpha, Ith), color="red", linewidth=2,
            label=f"Fit: alpha={alpha:.1f}, I_th={Ith:.4f}")
    ax.axvline(Ith, color="orange", linestyle=":", linewidth=1.5, label="I_th (Z=0.5 boundary)")
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Aerial intensity I", fontsize=12)
    ax.set_ylabel("Resist value Z", fontsize=12)
    ax.set_title("Aerial vs Resist (sampled pixels)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    residuals = all_Z - sigmoid(all_I, alpha, Ith)
    ax.hist(residuals, bins=100, color="#9C27B0", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Z_gt - Z_fit)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distribution\n"
                 f"RMSE={np.sqrt(np.mean(residuals**2)):.5f}, Bias={residuals.mean():.5f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / dataset / "global_fit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_alpha_vs_Ith(df: pd.DataFrame, dataset: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["Ith"], df["alpha"], c=df["r2"], cmap="viridis",
                    s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="R^2")
    ax.set_xlabel("I_th", fontsize=12)
    ax.set_ylabel("alpha", fontsize=12)
    ax.set_title(f"{dataset} — alpha vs I_th (coloured by R^2)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / dataset / "alpha_vs_Ith.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_example_tiles(example_records: list, dataset: str, output_dir: Path):
    n = len(example_records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"{dataset} — Example Tile Fits", fontsize=14)
    for j, title in enumerate(["Aerial (I)", "Resist GT (Z)", "Sigmoid Fit", "Residual"]):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")
    for i, rec in enumerate(example_records):
        aerial   = rec["aerial"]
        resist   = rec["resist"]
        Z_fit    = sigmoid(aerial, rec["alpha"], rec["Ith"]).astype(np.float32)
        residual = resist - Z_fit
        ims = [
            axes[i, 0].imshow(aerial,   cmap="gray", vmin=aerial.min(), vmax=aerial.max()),
            axes[i, 1].imshow(resist,   cmap="gray", vmin=0, vmax=1),
            axes[i, 2].imshow(Z_fit,    cmap="gray", vmin=0, vmax=1),
            axes[i, 3].imshow(residual, cmap="RdBu", vmin=-0.2, vmax=0.2),
        ]
        for j, im in enumerate(ims):
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(
            f"alpha={rec['alpha']:.0f}  I_th={rec['Ith']:.4f}  R2={rec['r2']:.4f}",
            fontsize=9)
    plt.tight_layout()
    out = output_dir / dataset / "example_tiles.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_sigmoid_curve(alpha: float, Ith: float, dataset: str,
                       output_dir: Path, I_range: tuple = (0.0, 1.0),
                       show_annotations: bool = True,
                       figsize: tuple = (8, 5), dpi: int = 150):
    """Publication-quality sigmoid curve with threshold annotation."""
    I = np.linspace(I_range[0], I_range[1], 2000)
    Z = sigmoid(I, alpha, Ith)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(I, Z, color="#2196F3", linewidth=2.5, label="Z = sigmoid(alpha*(I - I_th))")
    ax.axvline(Ith, color="red", linestyle="--", linewidth=1.5,
               label=f"I_th = {Ith:.4f}")
    ax.axhline(0.5, color="orange", linestyle="--", linewidth=1.5,
               label="Z = 0.5 (print boundary)")
    ax.scatter([Ith], [0.5], color="red", s=80, zorder=5)
    if show_annotations:
        offset = (I_range[1] - I_range[0]) * 0.08
        ax.annotate(
            f"  (I_th, 0.5) = ({Ith:.4f}, 0.5)",
            xy=(Ith, 0.5), xytext=(Ith + offset, 0.35),
            fontsize=10, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )
        mid_left  = I_range[0] + (Ith - I_range[0]) * 0.4
        mid_right = Ith + (I_range[1] - Ith) * 0.5
        ax.text(mid_left,  0.08, "Not printed", fontsize=10, color="gray", ha="center")
        ax.text(mid_right, 0.92, "Printed",     fontsize=10, color="gray", ha="center")
        ax.fill_betweenx([0, 1], I_range[0], Ith,        alpha=0.05, color="blue")
        ax.fill_betweenx([0, 1], Ith, I_range[1],         alpha=0.05, color="green")
    ax.set_xlabel("Aerial Intensity I", fontsize=13)
    ax.set_ylabel("Resist Value Z", fontsize=13)
    ax.set_title(f"{dataset} Resist Model: alpha={alpha:.1f}, I_th={Ith:.4f}", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(I_range[0], I_range[1])
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    out = output_dir / dataset / "sigmoid_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_threshold_validation(metrics_df: pd.DataFrame, example_records: list,
                               dataset: str, Ith: float, output_dir: Path):
    # Metric distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{dataset} — Threshold Validation  (I_th={Ith:.4f})", fontsize=14)
    for ax, (col, color, label) in zip(axes, [
        ("iou", "#2196F3", "IOU"),
        ("f1",  "#4CAF50", "F1 Score"),
        ("pa",  "#FF9800", "Pixel Accuracy"),
    ]):
        ax.hist(metrics_df[col], bins=50, color=color, edgecolor="white", linewidth=0.5)
        med = metrics_df[col].median()
        ax.axvline(med, color="red", linestyle="--", label=f"Median={med:.4f}")
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{label} Distribution")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / dataset / "threshold_validation_metrics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    metric_names = ["IOU", "F1", "Pixel Acc", "Precision", "Recall"]
    metric_cols  = ["iou", "f1", "pa", "precision", "recall"]
    means  = [metrics_df[c].mean() for c in metric_cols]
    stds   = [metrics_df[c].std()  for c in metric_cols]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(metric_names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="black")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{dataset} — Mean Threshold Metrics\n"
                 f"(aerial > {Ith:.4f} => predicted printed)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / dataset / "threshold_validation_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Example tile grid
    n = len(example_records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"{dataset} — Threshold Validation Examples  (I_th={Ith:.4f})", fontsize=14)
    for j, title in enumerate(["Aerial (I)", "Threshold Pred (I > I_th)",
                                "Printed GT", "Error (FP=red, FN=blue)"]):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")
    for i, rec in enumerate(example_records):
        aerial  = rec["aerial"]
        gt      = rec["printed_gt"]
        pred    = (aerial > Ith).astype(np.float32)
        gt_bin  = (gt > 0.5).astype(np.float32)
        error   = pred - gt_bin
        ims = [
            axes[i, 0].imshow(aerial, cmap="gray", vmin=aerial.min(), vmax=aerial.max()),
            axes[i, 1].imshow(pred,   cmap="gray", vmin=0, vmax=1),
            axes[i, 2].imshow(gt,     cmap="gray", vmin=0, vmax=1),
            axes[i, 3].imshow(error,  cmap="RdBu", vmin=-1, vmax=1),
        ]
        for j, im in enumerate(ims):
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(f"IOU={rec['iou']:.4f}  F1={rec['f1']:.4f}", fontsize=9)
    plt.tight_layout()
    out = output_dir / dataset / "threshold_validation_examples.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cross_dataset_summary(all_results: dict, output_dir: Path):
    datasets = list(all_results.keys())
    if len(datasets) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Dataset Sigmoid Parameter Comparison", fontsize=14)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    x = np.arange(len(datasets))
    for ax, (key, std_key, ylabel, title) in zip(axes, [
        ("global_alpha", "std_alpha", "alpha (steepness)", "alpha per Dataset"),
        ("global_Ith",   "std_Ith",   "I_th (intensity threshold)", "I_th per Dataset"),
    ]):
        vals = [all_results[d][key]     for d in datasets]
        stds = [all_results[d][std_key] for d in datasets]
        ax.bar(x, vals, yerr=stds, capsize=5, color=colors[:len(datasets)],
               alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / "cross_dataset_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main per-dataset fitting pipeline (parallel)
# ──────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset: str, data_root: Path, output_dir: Path,
                resume: bool, max_pixels: int, n_examples: int,
                num_workers: int = 4, max_fit_tiles: int = None):

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  (workers={num_workers})")
    print(f"{'='*60}")

    pairs = find_tile_pairs(data_root, dataset, keys=["aerial", "resist"])
    if not pairs:
        print(f"  [SKIP] No tile pairs found for {dataset}")
        return None

    print(f"  Found {len(pairs)} tile pairs")
    completed = load_completed(output_dir, dataset) if resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} tiles already processed")

    todo = [(tid, paths) for tid, paths in pairs if tid not in completed]
    print(f"  To process: {len(todo)} tiles")

    if not todo:
        # All done — load existing CSV and go straight to analysis
        csv_path = output_dir / dataset / COMPLETED_CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df["r2"] > 0.5].copy()
        else:
            print("  [WARN] Nothing to process and no CSV found.")
            return None
        global_alpha = float(df["alpha"].median())
        global_Ith   = float(df["Ith"].median())
        return _finalise_dataset(df, [], [], dataset, output_dir,
                                 global_alpha, global_Ith, n_examples, len(pairs))

    # Build worker arg tuples — use tile index as seed for reproducibility
    worker_args = [
        (tid,
         str(paths["aerial"]),
         str(paths["resist"]),
         max_pixels,
         i)                          # seed = index
        for i, (tid, paths) in enumerate(todo)
    ]

    all_records  = []
    global_I     = []
    global_Z     = []
    example_data = []   # store raw arrays for n_examples tiles
    batch_buffer = []

    MAX_GLOBAL_PIXELS = 5_000_000

    executor_cls = ProcessPoolExecutor if num_workers > 1 else None

    if executor_cls is not None:
        executor = executor_cls(max_workers=num_workers)
        futures  = {executor.submit(_fit_worker, arg): arg for arg in worker_args}
        pbar     = tqdm(total=len(futures), desc=dataset, unit="tile", dynamic_ncols=True)

        for future in as_completed(futures):
            tile_id, result, I_g, Z_g = future.result()
            pbar.update(1)

            if result is None:
                continue

            result["tile_id"] = tile_id
            result["dataset"] = dataset
            all_records.append(result)
            batch_buffer.append(result)

            if I_g is not None and sum(len(x) for x in global_I) < MAX_GLOBAL_PIXELS:
                global_I.append(I_g)
                global_Z.append(Z_g)

            if len(example_data) < n_examples and result["r2"] > 0.95:
                aerial = load_tile(Path(futures[future][1]))
                resist = load_tile(Path(futures[future][2]))
                if aerial is not None and resist is not None:
                    example_data.append({
                        "aerial": aerial, "resist": resist,
                        "alpha": result["alpha"], "Ith": result["Ith"],
                        "r2": result["r2"],
                    })

            if len(batch_buffer) >= 500:
                save_completed(output_dir, dataset, batch_buffer)
                batch_buffer = []

        pbar.close()
        executor.shutdown(wait=False)

    else:
        # Single-process fallback (--num-workers 1)
        for arg in tqdm(worker_args, desc=dataset, unit="tile", dynamic_ncols=True):
            tile_id, result, I_g, Z_g = _fit_worker(arg)
            if result is None:
                continue
            result["tile_id"] = tile_id
            result["dataset"] = dataset
            all_records.append(result)
            batch_buffer.append(result)
            if I_g is not None and sum(len(x) for x in global_I) < MAX_GLOBAL_PIXELS:
                global_I.append(I_g)
                global_Z.append(Z_g)
            if len(example_data) < n_examples and result["r2"] > 0.95:
                aerial = load_tile(Path(arg[1]))
                resist = load_tile(Path(arg[2]))
                if aerial is not None and resist is not None:
                    example_data.append({
                        "aerial": aerial, "resist": resist,
                        "alpha": result["alpha"], "Ith": result["Ith"],
                        "r2": result["r2"],
                    })
            if len(batch_buffer) >= 500:
                save_completed(output_dir, dataset, batch_buffer)
                batch_buffer = []

    if batch_buffer:
        save_completed(output_dir, dataset, batch_buffer)

    # Reload full CSV (includes prior runs if resuming)
    csv_path = output_dir / dataset / COMPLETED_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        if not all_records:
            print("  [WARN] No results.")
            return None
        df = pd.DataFrame(all_records)

    df = df[df["r2"] > 0.5].copy()
    print(f"\n  Tiles with R2>0.5: {len(df)} / {len(pairs)}")
    if len(df) == 0:
        return None

    # Global fit using accumulated pixel samples
    global_alpha = float(df["alpha"].median())
    global_Ith   = float(df["Ith"].median())

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
        except Exception as e:
            print(f"  [WARN] Global fit failed: {e}")
    else:
        all_I, all_Z = np.array([]), np.array([])

    return _finalise_dataset(df, all_I, all_Z, dataset, output_dir,
                             global_alpha, global_Ith, n_examples, len(pairs),
                             example_data)


def _finalise_dataset(df, all_I, all_Z, dataset, output_dir,
                      global_alpha, global_Ith, n_examples, n_total,
                      example_data=None):
    """Compute global fit stats, print summary, generate all plots."""
    print(f"\n  Per-tile statistics:")
    print(f"    alpha — median: {df['alpha'].median():.2f}  std: {df['alpha'].std():.2f}")
    print(f"    Ith   — median: {df['Ith'].median():.5f}  std: {df['Ith'].std():.5f}")
    print(f"    R2    — median: {df['r2'].median():.4f}  min: {df['r2'].min():.4f}")

    global_r2   = float("nan")
    global_rmse = float("nan")

    if len(all_I) > 0:
        try:
            p0     = [df["alpha"].median(), df["Ith"].median()]
            bounds = ([1.0, 0.0], [1e5, 1.0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(sigmoid, all_I, all_Z,
                                    p0=p0, bounds=bounds, maxfev=10000)
            global_alpha, global_Ith = float(popt[0]), float(popt[1])
            Z_pred      = sigmoid(all_I, global_alpha, global_Ith)
            global_rmse = float(np.sqrt(np.mean((all_Z - Z_pred) ** 2)))
            ss_res = np.sum((all_Z - Z_pred) ** 2)
            ss_tot = np.sum((all_Z - all_Z.mean()) ** 2)
            global_r2 = float(1.0 - ss_res / ss_tot)

            print(f"\n  Global fit ({len(all_I):,} pixels):")
            print(f"    alpha = {global_alpha:.2f}")
            print(f"    Ith   = {global_Ith:.6f}")
            print(f"    R2    = {global_r2:.6f}")
            print(f"    RMSE  = {global_rmse:.6f}")
            print(f"\n  => Binarisation threshold: aerial > {global_Ith:.6f}")

            plot_global_fit(all_I, all_Z, global_alpha, global_Ith, dataset, output_dir)
        except Exception as e:
            print(f"  [WARN] Final global fit failed: {e}")

    plot_parameter_distributions(df, dataset, output_dir)
    plot_alpha_vs_Ith(df, dataset, output_dir)
    if example_data:
        plot_example_tiles(example_data, dataset, output_dir)

    return {
        "dataset":       dataset,
        "n_tiles_total": n_total,
        "n_tiles_fit":   len(df),
        "median_alpha":  float(df["alpha"].median()),
        "std_alpha":     float(df["alpha"].std()),
        "median_Ith":    float(df["Ith"].median()),
        "std_Ith":       float(df["Ith"].std()),
        "median_r2":     float(df["r2"].median()),
        "global_alpha":  global_alpha,
        "global_Ith":    global_Ith,
        "global_r2":     global_r2,
        "global_rmse":   global_rmse,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Threshold validation pipeline (parallel)
# ──────────────────────────────────────────────────────────────────────────────

def run_threshold_validation(dataset: str, data_root: Path, output_dir: Path,
                              Ith: float, n_examples: int,
                              max_tiles: int = None, num_workers: int = 4):
    print(f"\n  Threshold Validation — {dataset}  (I_th={Ith:.4f}, workers={num_workers})")

    pairs = find_tile_pairs(data_root, dataset, keys=["aerial", "printed"])
    if not pairs:
        print("  [SKIP] No aerial+printed pairs found.")
        return
    if max_tiles:
        pairs = pairs[:max_tiles]
    print(f"  Validating on {len(pairs)} tiles...")

    worker_args = [
        (tid, str(paths["aerial"]), str(paths["printed"]), Ith)
        for tid, paths in pairs
    ]

    records      = []
    example_data = []

    def _process_result(tile_id, metrics, aerial, printed):
        if metrics is None:
            return
        metrics["tile_id"] = tile_id
        records.append(metrics)
        if len(example_data) < n_examples and aerial is not None:
            example_data.append({
                "aerial":     aerial,
                "printed_gt": printed,
                "iou":        metrics["iou"],
                "f1":         metrics["f1"],
            })

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_validate_worker, arg): arg for arg in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Validating {dataset}", unit="tile",
                               dynamic_ncols=True):
                tile_id, metrics, aerial, printed = future.result()
                _process_result(tile_id, metrics, aerial, printed)
    else:
        for arg in tqdm(worker_args, desc=f"Validating {dataset}",
                        unit="tile", dynamic_ncols=True):
            tile_id, metrics, aerial, printed = _validate_worker(arg)
            _process_result(tile_id, metrics, aerial, printed)

    if not records:
        print("  [WARN] No valid tiles processed.")
        return

    df = pd.DataFrame(records)

    print(f"\n  Threshold Validation Results (I_th={Ith:.4f}):")
    for col, label in [("iou", "IOU"), ("f1", "F1"),
                       ("pa", "Pixel Acc"), ("precision", "Precision"),
                       ("recall", "Recall"), ("fpr", "FPR"), ("fnr", "FNR")]:
        print(f"    {label:<14} — mean: {df[col].mean():.4f}  "
              f"median: {df[col].median():.4f}  std: {df[col].std():.4f}")

    csv_out = output_dir / dataset / "threshold_validation_results.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    plot_threshold_validation(df, example_data, dataset, Ith, output_dir)


def run_gt_consistency_check(dataset: str, data_root: Path, output_dir: Path,
                              alpha: float, Ith: float, num_workers: int = 4):
    print(f"\n  GT Consistency Check — {dataset}")

    pairs = find_tile_pairs(data_root, dataset, keys=["resist", "printed"])
    if not pairs:
        print("  [SKIP] No resist+printed pairs found.")
        return

    print(f"  Checking {len(pairs)} tiles...")

    worker_args = [
        (tid, str(paths["resist"]), str(paths["printed"]))
        for tid, paths in pairs
    ]

    records = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_consistency_worker, arg): arg
                       for arg in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"GT check {dataset}", unit="tile",
                               dynamic_ncols=True):
                tid, metrics = future.result()
                if metrics is not None:
                    metrics["tile_id"] = tid
                    records.append(metrics)
    else:
        for arg in tqdm(worker_args, desc=f"GT check {dataset}",
                        unit="tile", dynamic_ncols=True):
            tid, metrics = _consistency_worker(arg)
            if metrics is not None:
                metrics["tile_id"] = tid
                records.append(metrics)

    if not records:
        print("  [WARN] No valid tiles.")
        return

    df = pd.DataFrame(records)
    inconsistent = df[(df["mse"] > 0) | (df["iou"] < 1.0 - 1e-6)]

    print(f"\n  GT Consistency Results:")
    print(f"    Tiles checked      : {len(df)}")
    print(f"    MSE  — mean: {df['mse'].mean():.2e}  max: {df['mse'].max():.2e}")
    print(f"    IOU  — mean: {df['iou'].mean():.6f}  min: {df['iou'].min():.6f}")
    print(f"    Inconsistent tiles : {len(inconsistent)} / {len(df)}")

    if len(inconsistent) == 0:
        print("    => PERFECT CONSISTENCY: resist GT thresholded at 0.5 "
              "exactly reproduces printed GT for all tiles.")
    else:
        print(f"    => WARNING: {len(inconsistent)} tiles show discrepancy.")
        print(inconsistent[["tile_id", "mse", "iou", "fp", "fn"]].to_string(index=False))

    csv_out = output_dir / dataset / "gt_consistency.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sigmoid parameter study + threshold validation for LithoBench."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["MetalSet"], choices=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=Path("./sigmoidstudy_results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pixels-per-tile", type=int, default=None)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel worker processes. Set to 1 to disable multiprocessing "
             "(default: 4). On HPC, match to available CPUs per task."
    )

    # Sigmoid curve plot
    parser.add_argument("--plot-sigmoid", action="store_true",
                        help="Plot sigmoid curve(s) from stored results (no refitting)")
    parser.add_argument("--plot-datasets", nargs="+", default=None, choices=DATASETS)
    parser.add_argument("--sigmoid-I-min", type=float, default=0.0)
    parser.add_argument("--sigmoid-I-max", type=float, default=1.0)
    parser.add_argument("--sigmoid-no-annotations", action="store_true")
    parser.add_argument("--sigmoid-dpi", type=int, default=150)

    # Threshold validation
    parser.add_argument("--validate-threshold", action="store_true",
                        help="Validate aerial > Ith against ground truth printed images")
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--validate-max-tiles", type=int, default=None)

    parser.add_argument("--gt-consistency-check", action="store_true",
                    help="Verify resist GT thresholded at 0.5 exactly reproduces printed GT")
    parser.add_argument("--max-fit-tiles", type=int, default=None,
                    help="Limit fitting to the first N tiles per dataset (default: all)")

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
    print(f"  Workers    : {args.num_workers}")

    t0 = time.time()

    all_results  = {}
    results_path = args.output_dir / RESULTS_JSON
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    skip_fit = (args.plot_sigmoid or args.validate_threshold or args.gt_consistency_check) and \
           all(ds in all_results for ds in args.datasets)

    if not skip_fit:
        for dataset in args.datasets:
            result = run_dataset(
                dataset     = dataset,
                data_root   = args.data_root,
                output_dir  = args.output_dir,
                resume      = args.resume,
                max_pixels  = args.max_pixels_per_tile,
                n_examples  = args.n_examples,
                num_workers   = args.num_workers,
                max_fit_tiles = args.max_fit_tiles,
            )
            if result is not None:
                all_results[dataset] = result

        if len(all_results) >= 2:
            plot_cross_dataset_summary(all_results, args.output_dir)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved: {results_path}")

    # Sigmoid curve plots
    plot_targets = args.plot_datasets or args.datasets
    if args.plot_sigmoid or not skip_fit:
        for ds in plot_targets:
            if ds in all_results:
                res = all_results[ds]
                plot_sigmoid_curve(
                    alpha            = res["global_alpha"],
                    Ith              = res["global_Ith"],
                    dataset          = ds,
                    output_dir       = args.output_dir,
                    I_range          = (args.sigmoid_I_min, args.sigmoid_I_max),
                    show_annotations = not args.sigmoid_no_annotations,
                    dpi              = args.sigmoid_dpi,
                )

    # Threshold validation
    if args.validate_threshold:
        for ds in args.datasets:
            if ds not in all_results:
                print(f"  [WARN] No fitted results for {ds}. Run study first.")
                continue
            Ith = (args.threshold_override
                   if args.threshold_override is not None
                   else all_results[ds]["global_Ith"])
            run_threshold_validation(
                dataset     = ds,
                data_root   = args.data_root,
                output_dir  = args.output_dir,
                Ith         = Ith,
                n_examples  = args.n_examples,
                max_tiles   = args.validate_max_tiles,
                num_workers = args.num_workers,
            )


    if args.gt_consistency_check:
        for ds in args.datasets:
            if ds not in all_results:
                print(f"  [WARN] No fitted results for {ds}. Run study first.")
                continue
            run_gt_consistency_check(
                dataset     = ds,
                data_root   = args.data_root,
                output_dir  = args.output_dir,
                alpha       = all_results[ds]["global_alpha"],
                Ith         = all_results[ds]["global_Ith"],
                num_workers = args.num_workers,
            )


    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<14} {'alpha':>8} {'I_th':>10} {'R2':>10} {'Tiles':>10}")
    print(f"  {'-'*55}")
    for ds, res in all_results.items():
        print(f"  {ds:<14} {res['global_alpha']:>8.2f} "
              f"{res['global_Ith']:>10.6f} "
              f"{res.get('global_r2', float('nan')):>10.6f} "
              f"{res['n_tiles_fit']:>5} / {res['n_tiles_total']}")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Outputs in: {args.output_dir}\n")


if __name__ == "__main__":
    main()
