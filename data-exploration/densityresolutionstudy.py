"""
densityresolutionstudy.py
=========================
Pixel density study under average downsampling for the LithoBench dataset.

Motivation
----------
Lithography target/ILT images are binary (pixels are exactly 0 or 1).
Average pooling blends boundary blocks, producing intermediate values at
feature edges.  Because pixel density is defined as the fraction of pixels
> 0, even a single non-zero value in an averaged block marks that pixel as
"active".  This study quantifies:

  1. How much density *inflates* as resolution decreases under average pooling.
  2. Whether the effect is uniform across datatypes (binary Target/PixelILT
     vs. continuous Resist/Litho) and datasets.
  3. The density inflation ratio at each scale, complementing the static
     expansion coefficient in densitystudy.py.

Architecture
------------
Uses a flat image-level work queue fed to a ProcessPoolExecutor — one task
per (image × resolution) pair, results streamed to a per-image CSV
immediately.  Aggregation (mean + std per subset × resolution) is computed
from that CSV at the end or on demand.

Two output CSVs
---------------
  dr_per_image.csv   — one row per (image × resolution)
                       fields: subset, dataset, datatype, filename,
                               resolution, density, density_ratio
  dr_averaged.csv    — one row per (subset × resolution) with mean/std
                       of density and density_ratio

Resume behaviour
----------------
On startup the completed (subset, filename, resolution) triples are read
from dr_per_image.csv and excluded from the work queue.  Stop at any time
with Ctrl+C — resume by re-running the same command.
Use --force to discard existing results and start fresh.

Flags
-----
  --evaluate              run the parallel evaluation
  --aggregate             re-aggregate dr_per_image.csv → dr_averaged.csv
  --plot                  plot density vs resolution curves per datatype
  --plot-ratio            plot density-inflation ratio vs resolution
  --tables                print per-subset tables to the terminal
  --workers N             number of worker processes (default: cpu_count-1)
  --batch-size N          tasks per worker batch (default: 32)
  --samples N             cap images per subset (default: all)
  --resolutions N [N ..   resolutions to evaluate (default: 1024 512 256 128)
  --datasets D [D ...]    restrict plots to these datasets
  --datatypes T [T ...]   restrict plots to these datatypes
  --force                 delete existing CSVs and start fresh
  --save-plots            save PNG figures to the output directory
  --timeout N             per-image timeout in seconds (default: 60)

Usage examples
--------------
  python densityresolutionstudy.py --evaluate
  python densityresolutionstudy.py --evaluate --workers 8 --samples 500
  python densityresolutionstudy.py --aggregate
  python densityresolutionstudy.py --plot --save-plots
  python densityresolutionstudy.py --plot-ratio --datasets MetalSet ViaSet --save-plots
  python densityresolutionstudy.py --evaluate --aggregate --plot --plot-ratio --save-plots
"""

import argparse
import concurrent.futures as cf
import csv
import logging
import logging.handlers
import math
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import downscale_local_mean


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "lithobench-main"
OUTPUT_DIR   = PROJECT_ROOT / "density_resolution_study_output"
LOG_FILE     = OUTPUT_DIR / "density_resolution_study.log"

TARGET_RESOLUTIONS = [1024, 512, 256, 128]

DATA_DICT = {
    "MetalSet-Printed":    str(DATA_ROOT / "MetalSet"       / "printed"),
    "MetalSet-Resist":     str(DATA_ROOT / "MetalSet"       / "resist"),
    "MetalSet-Target":     str(DATA_ROOT / "MetalSet"       / "target"),
    "MetalSet-LevelILT":   str(DATA_ROOT / "MetalSet"       / "levelsetILT"),
    "MetalSet-Litho":      str(DATA_ROOT / "MetalSet"       / "litho"),
    "MetalSet-PixelILT":   str(DATA_ROOT / "MetalSet"       / "pixelILT"),
    "ViaSet-Printed":      str(DATA_ROOT / "ViaSet"         / "printed"),
    "ViaSet-Resist":       str(DATA_ROOT / "ViaSet"         / "resist"),
    "ViaSet-Target":       str(DATA_ROOT / "ViaSet"         / "target"),
    "ViaSet-LevelILT":     str(DATA_ROOT / "ViaSet"         / "levelsetILT"),
    "ViaSet-Litho":        str(DATA_ROOT / "ViaSet"         / "litho"),
    "ViaSet-PixelILT":     str(DATA_ROOT / "ViaSet"         / "pixelILT"),
    "StdContact-Printed":  str(DATA_ROOT / "StdContactFull" / "printed"),
    "StdContact-Resist":   str(DATA_ROOT / "StdContactFull" / "resist"),
    "StdContact-Target":   str(DATA_ROOT / "StdContactFull" / "target"),
    "StdContact-Litho":    str(DATA_ROOT / "StdContactFull" / "litho"),
    "StdContact-PixelILT": str(DATA_ROOT / "StdContactFull" / "pixelILT"),
    "StdMetal-Printed":    str(DATA_ROOT / "StdMetal"       / "printed"),
    "StdMetal-Resist":     str(DATA_ROOT / "StdMetal"       / "resist"),
    "StdMetal-Target":     str(DATA_ROOT / "StdMetal"       / "target"),
    "StdMetal-Litho":      str(DATA_ROOT / "StdMetal"       / "litho"),
    "StdMetal-PixelILT":   str(DATA_ROOT / "StdMetal"       / "pixelILT"),
}

DATATYPE_ORDER = ["LevelILT", "Litho", "PixelILT", "Printed", "Resist", "Target"]

# -- CSV schemas ---------------------------------------------------------------

PER_IMAGE_FIELDS = [
    "subset", "dataset", "datatype", "filename",
    "resolution", "density", "density_ratio",
]

AVERAGED_FIELDS = [
    "subset", "dataset", "datatype", "resolution",
    "num_samples",
    "mean_density", "std_density",
    "mean_ratio", "std_ratio",
]


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("densityresolutionstudy")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | PID %(process)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3,
        encoding="utf-8", mode="a",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("densityresolutionstudy")


# ------------------------------------------------------------------------------
# Resume helpers
# ------------------------------------------------------------------------------

def load_completed(csv_path: Path) -> set:
    """Return set of (subset, filename, resolution) triples already in the CSV."""
    done = set()
    if not csv_path.exists():
        return done
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["subset"], row["filename"], int(row["resolution"])))
    except Exception:
        pass
    return done


def append_rows(csv_path: Path, rows: list):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------
# Image loading
# ------------------------------------------------------------------------------

def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = img[..., ::-1]
        elif img.shape[2] == 4:
            img = img[..., [2, 1, 0, 3]]
    img = img.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0
    return img


# ------------------------------------------------------------------------------
# Downsampling (average method only)
# ------------------------------------------------------------------------------

def downsample_average(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    fh, fw = h // target_size, w // target_size
    factors = (fh, fw, 1) if image.ndim == 3 else (fh, fw)
    return downscale_local_mean(image, factors).astype(image.dtype)


# ------------------------------------------------------------------------------
# Pixel density
# ------------------------------------------------------------------------------

def compute_pixel_density(image: np.ndarray) -> float:
    """Fraction of pixels with value > 0 (any channel for RGB)."""
    if image.ndim == 3:
        active = np.any(image > 0, axis=-1)
    else:
        active = image > 0
    return float(np.mean(active))


# ------------------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------------------

def _process_image(task: tuple):
    """
    Compute pixel density at each resolution for one image.

    task: (subset, dataset, datatype, file_path_str, resolutions_tuple, log_file)

    Returns list of row dicts (one per resolution) on success, empty list on failure.
    """
    import matplotlib
    matplotlib.use("Agg")

    subset, dataset, datatype, file_path_str, resolutions, log_file = task
    logger = setup_logging(Path(log_file))
    file_path = Path(file_path_str)

    img = load_image(file_path)
    if img is None:
        logger.warning(f"Failed to load {file_path} — skipping")
        return []

    original_density = compute_pixel_density(img)

    rows = []
    for res in resolutions:
        try:
            h, w = img.shape[:2]
            if h < res or w < res:
                continue
            downsampled = downsample_average(img, res)
            ds_density = compute_pixel_density(downsampled)
            ratio = (
                round(ds_density / original_density, 8)
                if original_density > 1e-9
                else float("nan")
            )
            rows.append({
                "subset":        subset,
                "dataset":       dataset,
                "datatype":      datatype,
                "filename":      file_path.name,
                "resolution":    res,
                "density":       round(ds_density, 8),
                "density_ratio": ratio,
            })
        except Exception as exc:
            logger.warning(f"ERROR {subset}/{file_path.name} res={res}: {exc}")

    return rows


def _process_batch(batch_tasks: list):
    all_rows = []
    failed = 0
    for task in batch_tasks:
        result = _process_image(task)
        if not result:
            failed += 1
        else:
            all_rows.extend(result)
    return all_rows, failed


# ------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------

def aggregate(per_image_csv: Path, averaged_csv: Path):
    """Compute mean/std of density and density_ratio per (subset, resolution)."""
    from collections import defaultdict

    data = defaultdict(list)    # (subset, resolution) → list of (density, ratio)
    meta = {}                   # (subset, resolution) → (dataset, datatype)

    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                d = float(row["density"])
                r = row["density_ratio"]
                res = int(row["resolution"])
                key = (row["subset"], res)
                if math.isnan(d) or math.isinf(d):
                    continue
                ratio_val = float(r) if r not in ("", "nan") else float("nan")
                data[key].append((d, ratio_val))
                meta[key] = (row["dataset"], row["datatype"])
            except (ValueError, KeyError):
                continue

    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AVERAGED_FIELDS)
        w.writeheader()
        for (subset, res) in sorted(data.keys()):
            pairs = data[(subset, res)]
            densities = [p[0] for p in pairs]
            ratios = [p[1] for p in pairs if not math.isnan(p[1])]
            dataset, datatype = meta[(subset, res)]
            w.writerow({
                "subset":       subset,
                "dataset":      dataset,
                "datatype":     datatype,
                "resolution":   res,
                "num_samples":  len(densities),
                "mean_density": round(float(np.mean(densities)), 8),
                "std_density":  round(float(np.std(densities)),  8),
                "mean_ratio":   round(float(np.mean(ratios)), 8) if ratios else float("nan"),
                "std_ratio":    round(float(np.std(ratios)),  8) if ratios else float("nan"),
            })

    return averaged_csv


# ------------------------------------------------------------------------------
# Evaluation orchestrator
# ------------------------------------------------------------------------------

def run_evaluation(
    data_dict:    dict,
    output_dir:   Path,
    resolutions:  list   = None,
    num_workers:  int    = None,
    num_samples:  int    = None,
    batch_size:   int    = 32,
    force:        bool   = False,
    timeout:      int    = 60,
):
    import tqdm

    if resolutions is None:
        resolutions = TARGET_RESOLUTIONS

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "dr_per_image.csv"
    averaged_csv  = output_dir / "dr_averaged.csv"
    logger        = get_logger()

    if force:
        for p in (per_image_csv, averaged_csv):
            if p.exists():
                p.unlink()
        logger.info("--force: cleared existing CSVs")
        print("--force: cleared existing results, starting fresh.\n")

    completed = load_completed(per_image_csv)
    logger.info(f"Already completed: {len(completed)} (image × resolution) pairs")

    print("Scanning dataset directories ...")
    tasks = []
    res_tuple = tuple(resolutions)

    for subset_key, image_dir in data_dict.items():
        parts    = subset_key.split("-", 1)
        dataset  = parts[0]
        datatype = parts[1] if len(parts) > 1 else ""
        path     = Path(image_dir)

        if not path.exists():
            logger.warning(f"Directory not found, skipping: {path}")
            print(f"  WARNING: {path} not found — skipping {subset_key}")
            continue

        files = sorted(f for f in path.iterdir() if f.is_file())
        if num_samples is not None:
            files = random.sample(files, min(num_samples, len(files)))

        for f in files:
            # Only add task if at least one resolution is incomplete
            pending_res = [
                r for r in resolutions
                if (subset_key, f.name, r) not in completed
            ]
            if pending_res:
                tasks.append((
                    subset_key, dataset, datatype,
                    str(f), tuple(pending_res), str(LOG_FILE),
                ))

    pending = len(tasks)
    total_pairs = sum(len(t[4]) for t in tasks)

    if completed:
        print(f"\nResuming — {len(completed)} pairs already done, "
              f"{pending} images / {total_pairs} (image × resolution) pairs remaining.\n")
    else:
        print(f"\nStarting fresh — {pending} images / {total_pairs} pairs to process.\n")

    if not pending:
        print("Nothing to do — all images already processed.")
        aggregate(per_image_csv, averaged_csv)
        print(f"dr_averaged.csv written to: {averaged_csv}")
        return per_image_csv, averaged_csv

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, pending)
    batch_size  = max(1, int(batch_size))

    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    logger.info(
        f"Starting | images={pending} | pairs={total_pairs} | "
        f"workers={num_workers} | batch_size={batch_size}"
    )
    print(
        f"Workers: {num_workers}  |  Images: {pending}  |  "
        f"Pairs: {total_pairs}  |  Batches: {len(batches)}\n"
    )

    failed   = 0
    timedout = 0

    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(_process_batch, batch): batch
            for batch in batches
        }

        with tqdm.tqdm(
            total=total_pairs,
            desc="Density × resolution",
            unit="pair",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for future in cf.as_completed(future_to_batch):
                batch = future_to_batch[future]
                batch_pairs = sum(len(t[4]) for t in batch)
                first = Path(batch[0][3]).name

                try:
                    rows, batch_failed = future.result(timeout=max(timeout, timeout * len(batch)))
                    if rows:
                        append_rows(per_image_csv, rows)
                    failed += batch_failed
                except cf.TimeoutError:
                    timedout += len(batch)
                    logger.error(f"TIMEOUT batch starting {first} — skipped {len(batch)} images")
                except Exception as exc:
                    failed += len(batch)
                    logger.error(f"ERROR batch starting {first}: {exc}")

                pbar.update(batch_pairs)

    issues = failed + timedout
    if issues:
        print(f"\n  {failed} failed, {timedout} timed out. See {LOG_FILE} for details.")
    else:
        print("\n  All images processed successfully.")

    print("\nAggregating results ...")
    aggregate(per_image_csv, averaged_csv)
    logger.info(f"Aggregation complete: {averaged_csv}")
    print(f"  dr_per_image.csv : {per_image_csv}")
    print(f"  dr_averaged.csv  : {averaged_csv}")

    return per_image_csv, averaged_csv


# ------------------------------------------------------------------------------
# Terminal tables
# ------------------------------------------------------------------------------

def print_tables(averaged_csv: Path):
    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"], {}).setdefault(row["datatype"], []).append(row)

    for dataset in sorted(groups):
        resolutions = sorted({int(r["resolution"]) for dt_rows in groups[dataset].values() for r in dt_rows})
        header_parts = ["Datatype".ljust(12)] + [f"{r:>8}" for r in resolutions]
        header = "  ".join(header_parts)
        width  = len(header)
        print(f"\n{'=' * width}")
        print(f"  {dataset}  — mean density (std) by resolution")
        print(f"{'=' * width}")
        print(header)
        print("-" * width)
        for datatype in DATATYPE_ORDER:
            if datatype not in groups[dataset]:
                continue
            dt_rows = {int(r["resolution"]): r for r in groups[dataset][datatype]}
            parts = [datatype.ljust(12)]
            for res in resolutions:
                if res in dt_rows:
                    row = dt_rows[res]
                    parts.append(f"{float(row['mean_density']):>8.4f}")
                else:
                    parts.append("       —")
            print("  ".join(parts))
        print("-" * width)


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

DATASET_COLORS = {
    "MetalSet":   "#2E86AB",
    "ViaSet":     "#E07B39",
    "StdContact": "#6A994E",
    "StdMetal":   "#9B5DE5",
}
DATASET_ORDER  = ["MetalSet", "ViaSet", "StdMetal", "StdContact"]
DEFAULT_COLOR  = "#888888"

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]

_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _load_averaged(averaged_csv: Path):
    import pandas as pd
    return pd.read_csv(averaged_csv)


def _filter_df(df, datasets=None, datatypes=None):
    if datasets:
        ds_lower = [d.lower() for d in datasets]
        df = df[df["dataset"].str.lower().isin(ds_lower)]
    if datatypes:
        dt_lower = [d.lower() for d in datatypes]
        df = df[df["datatype"].str.lower().isin(dt_lower)]
    return df


def plot_density_vs_resolution(
    averaged_csv: Path,
    output_dir:   Path,
    datasets:     list = None,
    datatypes:    list = None,
    save:         bool = False,
):
    """
    One subplot per dataset, lines = datatypes.
    X: resolution (ascending), Y: mean density ± 1 std.
    """
    df = _load_averaged(averaged_csv)
    df = _filter_df(df, datasets, datatypes)
    if df.empty:
        print("No data matches the given filters.")
        return

    all_datasets = sorted(df["dataset"].unique())
    n_ds = len(all_datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, all_datasets):
        ds_df = df[df["dataset"] == dataset]
        present_types = [dt for dt in DATATYPE_ORDER if dt in ds_df["datatype"].unique()]
        present_types += [dt for dt in ds_df["datatype"].unique() if dt not in DATATYPE_ORDER]

        for i, datatype in enumerate(present_types):
            dt_df = ds_df[ds_df["datatype"] == datatype].sort_values("resolution")
            resolutions = dt_df["resolution"].values
            means = dt_df["mean_density"].values
            stds  = dt_df["std_density"].values
            color  = _COLORS[i % len(_COLORS)]
            marker = _MARKERS[i % len(_MARKERS)]
            ax.errorbar(
                resolutions, means, yerr=stds,
                label=datatype, color=color, marker=marker,
                linewidth=1.5, markersize=5, capsize=3,
            )

        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("Mean Pixel Density")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["resolution"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.invert_xaxis()   # higher resolution on the left (less downsampled)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Pixel density vs. resolution (average downsampling)", fontsize=13)
    plt.tight_layout()

    if save:
        out = output_dir / "density_vs_resolution.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_density_ratio_vs_resolution(
    averaged_csv: Path,
    output_dir:   Path,
    datasets:     list = None,
    datatypes:    list = None,
    save:         bool = False,
):
    """
    One subplot per dataset, lines = datatypes.
    Y: mean density_ratio (downsampled / original density) ± 1 std.
    A ratio > 1 means density inflated after downsampling.
    """
    df = _load_averaged(averaged_csv)
    df = _filter_df(df, datasets, datatypes)
    df = df.dropna(subset=["mean_ratio"])
    if df.empty:
        print("No ratio data available.")
        return

    all_datasets = sorted(df["dataset"].unique())
    n_ds = len(all_datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, all_datasets):
        ds_df = df[df["dataset"] == dataset]
        present_types = [dt for dt in DATATYPE_ORDER if dt in ds_df["datatype"].unique()]
        present_types += [dt for dt in ds_df["datatype"].unique() if dt not in DATATYPE_ORDER]

        for i, datatype in enumerate(present_types):
            dt_df = ds_df[ds_df["datatype"] == datatype].sort_values("resolution")
            resolutions = dt_df["resolution"].values
            means = dt_df["mean_ratio"].values
            stds  = dt_df["std_ratio"].values
            color  = _COLORS[i % len(_COLORS)]
            marker = _MARKERS[i % len(_MARKERS)]
            ax.errorbar(
                resolutions, means, yerr=stds,
                label=datatype, color=color, marker=marker,
                linewidth=1.5, markersize=5, capsize=3,
            )

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="ratio = 1")
        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("Density ratio (downsampled / original)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["resolution"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.invert_xaxis()
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Density inflation ratio vs. resolution (average downsampling)", fontsize=13)
    plt.tight_layout()

    if save:
        out = output_dir / "density_ratio_vs_resolution.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_density_histograms(
    per_image_csv: Path,
    resolution:    int,
    output_dir:    Path,
    bins:          int  = 40,
    samples:       int  = None,
    datasets:      list = None,
    datatypes:     list = None,
    save:          bool = False,
):
    """
    One figure per datatype; one subplot per dataset.
    Shows the distribution of pixel densities at the given downsampled resolution,
    with mean and ±1σ marked as vertical lines.

    samples: if set, randomly draw at most this many values per (dataset, datatype).
    """
    ds_filter = {d.lower() for d in datasets}  if datasets  else None
    dt_filter = {d.lower() for d in datatypes} if datatypes else None

    records = {}   # (dataset, datatype) → list of density floats
    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["resolution"]) != resolution:
                continue
            v = row.get("density", "")
            if v == "":
                continue
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            ds, dt = row["dataset"], row["datatype"]
            if ds_filter and ds.lower() not in ds_filter:
                continue
            if dt_filter and dt.lower() not in dt_filter:
                continue
            records.setdefault((ds, dt), []).append(fv)

    if samples is not None:
        records = {
            k: random.sample(v, min(samples, len(v)))
            for k, v in records.items()
        }

    if not records:
        print(f"No data found for resolution={resolution}. "
              "Check --hist-resolution and that --evaluate has been run.")
        return

    all_datatypes = [dt for dt in DATATYPE_ORDER if any(dt == k[1] for k in records)]
    all_datatypes += sorted(
        dt for dt in set(k[1] for k in records) if dt not in DATATYPE_ORDER
    )

    known_ds   = [ds for ds in DATASET_ORDER if any(ds == k[0] for k in records)]
    unknown_ds = sorted(ds for ds in set(k[0] for k in records) if ds not in DATASET_ORDER)
    all_datasets = known_ds + unknown_ds

    for datatype in all_datatypes:
        active = [ds for ds in all_datasets if (ds, datatype) in records]
        if not active:
            continue

        n_cols = len(active)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4),
                                 constrained_layout=False)
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.97, wspace=0.35)
        if n_cols == 1:
            axes = [axes]

        for ax, dataset in zip(axes, active):
            vals  = records[(dataset, datatype)]
            color = DATASET_COLORS.get(dataset, DEFAULT_COLOR)
            mean  = float(np.mean(vals))
            std   = float(np.std(vals))

            ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(mean,       color="#222222", linewidth=1.8,
                       linestyle="-",  label=f"Mean {mean:.4f}")
            ax.axvline(mean - std, color="#222222", linewidth=1.2,
                       linestyle="--", label=f"±1σ  {std:.4f}")
            ax.axvline(mean + std, color="#222222", linewidth=1.2,
                       linestyle="--")

            ax.set_title(f"{dataset}\nn={len(vals):,}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Pixel Density", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            ax.legend(fontsize=10, framealpha=0.85)
            ax.grid(True, alpha=0.2, linestyle="--")

        fig.suptitle(
            f"Datatype: {datatype}  —  resolution: {resolution}px (average downsampling)",
            fontsize=14, fontweight="bold", y=0.97,
        )

        if save:
            out = output_dir / f"density_hist_{datatype}_{resolution}px.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
        else:
            plt.show()
        plt.close(fig)


# ------------------------------------------------------------------------------
# Argument parser + main
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pixel density under average downsampling — LithoBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--evaluate",        action="store_true", help="run parallel evaluation")
    p.add_argument("--aggregate",       action="store_true", help="re-aggregate per-image CSV")
    p.add_argument("--plot",            action="store_true", help="plot density vs resolution")
    p.add_argument("--plot-ratio",      action="store_true", help="plot density ratio vs resolution")
    p.add_argument("--plot-hist",       action="store_true", help="plot density histograms at a specific resolution")
    p.add_argument("--hist-resolution", type=int, default=None, metavar="N",
                   help="resolution to use for --plot-hist (e.g. 512)")
    p.add_argument("--bins",            type=int, default=40, metavar="N",
                   help="number of histogram bins (default: 40)")
    p.add_argument("--tables",          action="store_true", help="print tables to terminal")
    p.add_argument("--workers",         type=int,   default=None,  metavar="N")
    p.add_argument("--batch-size",      type=int,   default=32,    metavar="N")
    p.add_argument("--samples",         type=int,   default=None,  metavar="N")
    p.add_argument("--resolutions",     type=int,   nargs="+",     default=TARGET_RESOLUTIONS, metavar="N")
    p.add_argument("--datasets",        type=str,   nargs="+",     default=None, metavar="D")
    p.add_argument("--datatypes",       type=str,   nargs="+",     default=None, metavar="T")
    p.add_argument("--force",           action="store_true")
    p.add_argument("--save-plots",      action="store_true")
    p.add_argument("--timeout",         type=int,   default=60,    metavar="N")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)

    per_image_csv = OUTPUT_DIR / "dr_per_image.csv"
    averaged_csv  = OUTPUT_DIR / "dr_averaged.csv"

    if args.evaluate:
        run_evaluation(
            data_dict   = DATA_DICT,
            output_dir  = OUTPUT_DIR,
            resolutions = args.resolutions,
            num_workers = args.workers,
            num_samples = args.samples,
            batch_size  = args.batch_size,
            force       = args.force,
            timeout     = args.timeout,
        )

    if args.aggregate:
        if not per_image_csv.exists():
            print(f"ERROR: {per_image_csv} not found — run --evaluate first.")
            sys.exit(1)
        aggregate(per_image_csv, averaged_csv)
        print(f"Aggregated → {averaged_csv}")

    if args.tables:
        if not averaged_csv.exists():
            print(f"ERROR: {averaged_csv} not found — run --evaluate or --aggregate first.")
            sys.exit(1)
        print_tables(averaged_csv)

    if args.plot:
        if not averaged_csv.exists():
            print(f"ERROR: {averaged_csv} not found — run --evaluate or --aggregate first.")
            sys.exit(1)
        plot_density_vs_resolution(
            averaged_csv, OUTPUT_DIR,
            datasets=args.datasets, datatypes=args.datatypes,
            save=args.save_plots,
        )

    if args.plot_ratio:
        if not averaged_csv.exists():
            print(f"ERROR: {averaged_csv} not found — run --evaluate or --aggregate first.")
            sys.exit(1)
        plot_density_ratio_vs_resolution(
            averaged_csv, OUTPUT_DIR,
            datasets=args.datasets, datatypes=args.datatypes,
            save=args.save_plots,
        )

    if args.plot_hist:
        if not per_image_csv.exists():
            print(f"ERROR: {per_image_csv} not found — run --evaluate first.")
            sys.exit(1)
        if args.hist_resolution is None:
            print("ERROR: --plot-hist requires --hist-resolution N (e.g. --hist-resolution 512).")
            sys.exit(1)
        plot_density_histograms(
            per_image_csv, args.hist_resolution, OUTPUT_DIR,
            bins=args.bins,
            samples=args.samples,
            datasets=args.datasets, datatypes=args.datatypes,
            save=args.save_plots,
        )

    if not any([args.evaluate, args.aggregate, args.tables, args.plot, args.plot_ratio, args.plot_hist]):
        print("No action specified. Use --help to see available flags.")


if __name__ == "__main__":
    main()
