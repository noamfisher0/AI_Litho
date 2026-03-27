"""
densitystudy.py
===============
Pixel density study for the LithoBench dataset.

Pixel density is defined as the fraction of pixels with value > 0 in an
image.  For binary masks (PixelILT, Target) this equals the fraction of
active pixels.  For continuous images (Litho, Printed, Resist) it measures
the fraction of non-zero pixels.

Architecture
------------
Uses a flat image-level work queue fed to a ProcessPoolExecutor — one task
per image, results streamed to a per-image CSV immediately.  Aggregation
(mean + std per subset) is computed from that CSV at the end or on demand.

Two output CSVs
---------------
  density_per_image.csv   — one row per image with its pixel density score
  density_averaged.csv    — one row per subset with mean and std deviation

Resume behaviour
----------------
On startup the completed (subset, filename) pairs are read from
density_per_image.csv and excluded from the work queue.  Stop at any time
with Ctrl+C and resume by re-running the same command.
Use --force to discard existing results and start fresh.

Flags
-----
  --evaluate              run the parallel pixel density evaluation
  --aggregate             re-aggregate density_per_image.csv → density_averaged.csv
  --plot                  generate histograms from density_per_image.csv
  --plot-mean-std         generate mean±std comparison plots from density_averaged.csv
  --tables                print per-subset mean/std tables to terminal
  --snr                   calculate and print SNR (mean/std) per subset
  --snr-plot              generate bar plot of SNR from density_averaged.csv
  --expansion             calculate and print per-tile expansion coefficient summary
  --expansion-plot        generate mean±std expansion coefficient plot
  --workers N             number of worker processes (default: cpu_count - 1)
  --batch-size N          images per worker task batch (default: 32)
  --samples N             cap images per subset (default: None = all images)
  --force                 delete existing CSVs and start fresh
  --save-plots            save plot PNGs to the output directory
  --bins N                number of histogram bins (default: 40)
  --datatypes A B ...     restrict all plots to these datatypes (case-insensitive)
                            e.g. --datatypes PixelILT Resist
  --datasets A B ...      restrict all plots to these datasets (case-insensitive)
                            e.g. --datasets MetalSet ViaSet
  --min-target-density F  floor for Target density when computing expansion
                            coefficient — tiles at or below this are excluded
                            (default: 1e-4)

Usage examples
--------------
  python densitystudy.py --evaluate
  python densitystudy.py --evaluate --workers 8 --samples 500
  python densitystudy.py --aggregate
  python densitystudy.py --plot --save-plots
  python densitystudy.py --plot-mean-std --save-plots
  python densitystudy.py --plot --datatypes PixelILT Resist --datasets MetalSet ViaSet
  python densitystudy.py --snr-plot --datasets MetalSet StdMetal
  python densitystudy.py --expansion --expansion-plot --save-plots
  python densitystudy.py --expansion-plot --datasets MetalSet StdMetal --save-plots
  python densitystudy.py --evaluate --aggregate --plot
  python densitystudy.py --evaluate --force --workers 8 --batch-size 64
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
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "lithobench-main"
OUTPUT_DIR   = PROJECT_ROOT / "density_study_output"
LOG_FILE     = OUTPUT_DIR / "density_study.log"

# Cap images per subset for test runs — set to None for the full dataset
NUM_SAMPLES = None

# Floor for Target density when computing expansion coefficient.
# Tiles at or below this value are excluded to avoid division instability.
DEFAULT_MIN_TARGET_DENSITY = 1e-4

DATA_DICT = {
    "MetalSet-Printed":    str(DATA_ROOT / "MetalSet"   / "printed"),
    "MetalSet-Resist":     str(DATA_ROOT / "MetalSet"   / "resist"),
    "MetalSet-Target":     str(DATA_ROOT / "MetalSet"   / "target"),
    "MetalSet-LevelILT":   str(DATA_ROOT / "MetalSet"   / "levelsetILT"),
    "MetalSet-Litho":      str(DATA_ROOT / "MetalSet"   / "litho"),
    "MetalSet-PixelILT":   str(DATA_ROOT / "MetalSet"   / "pixelILT"),
    "ViaSet-Printed":      str(DATA_ROOT / "ViaSet"     / "printed"),
    "ViaSet-Resist":       str(DATA_ROOT / "ViaSet"     / "resist"),
    "ViaSet-Target":       str(DATA_ROOT / "ViaSet"     / "target"),
    "ViaSet-LevelILT":     str(DATA_ROOT / "ViaSet"     / "levelsetILT"),
    "ViaSet-Litho":        str(DATA_ROOT / "ViaSet"     / "litho"),
    "ViaSet-PixelILT":     str(DATA_ROOT / "ViaSet"     / "pixelILT"),
    "StdContact-Printed":  str(DATA_ROOT / "StdContact" / "printed"),
    "StdContact-Resist":   str(DATA_ROOT / "StdContact" / "resist"),
    "StdContact-Target":   str(DATA_ROOT / "StdContact" / "target"),
    "StdContact-Litho":    str(DATA_ROOT / "StdContact" / "litho"),
    "StdContact-PixelILT": str(DATA_ROOT / "StdContact" / "pixelILT"),
    "StdMetal-Printed":    str(DATA_ROOT / "StdMetal"   / "printed"),
    "StdMetal-Resist":     str(DATA_ROOT / "StdMetal"   / "resist"),
    "StdMetal-Target":     str(DATA_ROOT / "StdMetal"   / "target"),
    "StdMetal-Litho":      str(DATA_ROOT / "StdMetal"   / "litho"),
    "StdMetal-PixelILT":   str(DATA_ROOT / "StdMetal"   / "pixelILT"),
}

# ── CSV schemas ───────────────────────────────────────────────────────────────

PER_IMAGE_FIELDS = [
    "subset", "dataset", "datatype", "filename", "pixel_density",
]

AVERAGED_FIELDS = [
    "subset", "dataset", "datatype",
    "num_samples", "mean_density", "std_density",
]


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("densitystudy")
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
    return logging.getLogger("densitystudy")


# ──────────────────────────────────────────────────────────────────────────────
# Resume helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_completed_images(csv_path: Path) -> set:
    """Return set of (subset, filename) pairs already in the per-image CSV."""
    done = set()
    if not csv_path.exists():
        return done
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["subset"], row["filename"]))
    except Exception:
        pass
    return done


def append_image_rows(csv_path: Path, rows: list):
    """Append per-image result rows — called from main process only."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Pixel density computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_pixel_density(image: np.ndarray) -> float:
    """
    Fraction of pixels with value > 0.
    For RGB/RGBA images a pixel is considered active if any channel is > 0.
    Input should be a float32 array normalised to [0, 1].
    """
    if image.ndim == 3:
        active = np.any(image > 0, axis=-1)
    else:
        active = image > 0
    return float(np.mean(active))


# ──────────────────────────────────────────────────────────────────────────────
# Image loading (cv2 — no GUI involvement)
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: Path):
    """Load image with cv2, return float32 array in [0, 1]. Returns None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = img[..., ::-1]           # BGR  → RGB
        elif img.shape[2] == 4:
            img = img[..., [2, 1, 0, 3]]   # BGRA → RGBA
    img = img.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Worker function  (one call per image)
# ──────────────────────────────────────────────────────────────────────────────

def _process_image(task: tuple):
    """
    Compute pixel density for a single image.

    Parameters
    ----------
    task : (subset, dataset, datatype, file_path_str, log_file)

    Returns
    -------
    dict row on success, None on failure.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-GUI — safe in spawned worker processes

    subset, dataset, datatype, file_path_str, log_file = task

    logger = setup_logging(Path(log_file))
    file_path = Path(file_path_str)
    logger.debug(f"Processing {subset}/{file_path.name}")

    img = load_image(file_path)
    if img is None:
        logger.warning(f"Failed to load {file_path} — skipping")
        return None

    try:
        density = compute_pixel_density(img)
        logger.debug(f"DONE {subset}/{file_path.name} density={density:.6f}")
        return {
            "subset":        subset,
            "dataset":       dataset,
            "datatype":      datatype,
            "filename":      file_path.name,
            "pixel_density": round(density, 8),
        }
    except Exception as exc:
        logger.warning(f"ERROR {subset}/{file_path.name}: {exc}")
        return None


def _process_image_batch(batch_tasks: list):
    """
    Compute pixel density for a batch of images in one worker call.

    Parameters
    ----------
    batch_tasks : list of task tuples expected by _process_image

    Returns
    -------
    tuple[list[dict], int]
        Successful rows and number of failed items in the batch.
    """
    rows = []
    failed = 0
    for task in batch_tasks:
        row = _process_image(task)
        if row is None:
            failed += 1
        else:
            rows.append(row)
    return rows, failed


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_to_averaged_csv(per_image_csv: Path, averaged_csv: Path):
    """
    Read density_per_image.csv and compute mean + std per subset,
    writing results to density_averaged.csv.
    """
    densities = {}   # subset → list of float
    meta      = {}   # subset → (dataset, datatype)

    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row["subset"]
            v = row["pixel_density"]
            if v == "" or v is None:
                continue
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            densities.setdefault(s, []).append(fv)
            meta[s] = (row["dataset"], row["datatype"])

    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AVERAGED_FIELDS)
        w.writeheader()
        for subset in sorted(densities.keys()):
            vals = densities[subset]
            dataset, datatype = meta[subset]
            w.writerow({
                "subset":        subset,
                "dataset":       dataset,
                "datatype":      datatype,
                "num_samples":   len(vals),
                "mean_density":  round(float(np.mean(vals)),  8),
                "std_density":   round(float(np.std(vals)),   8),
            })

    return averaged_csv


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    data_dict:   dict,
    output_dir:  Path,
    num_workers: int  = None,
    num_samples: int  = None,
    batch_size:  int  = 32,
    force:       bool = False,
    timeout:     int  = 60,
):
    """
    Build a flat image-level work queue across all subsets, dispatch to a
    ProcessPoolExecutor, stream results to density_per_image.csv, then
    aggregate to density_averaged.csv.
    """
    import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "density_per_image.csv"
    averaged_csv  = output_dir / "density_averaged.csv"
    logger        = get_logger()

    # ── Force: clear existing results ─────────────────────────────────────────
    if force:
        for p in (per_image_csv, averaged_csv):
            if p.exists():
                p.unlink()
        logger.info("--force: cleared existing CSVs")
        print("--force: cleared existing results, starting fresh.\n")

    # ── Load already-completed (subset, filename) pairs ───────────────────────
    completed = load_completed_images(per_image_csv)
    logger.info(f"Already completed: {len(completed)} images")

    # ── Build flat work queue ─────────────────────────────────────────────────
    print("Scanning dataset directories ...")
    tasks = []

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
            if (subset_key, f.name) in completed:
                continue
            tasks.append((
                subset_key, dataset, datatype,
                str(f), str(LOG_FILE),
            ))

    pending = len(tasks)

    if completed:
        print(f"\nResuming — {len(completed)} images already done, "
              f"{pending} remaining.\n")
    else:
        print(f"\nStarting fresh — {pending} images to process.\n")

    if not pending:
        print("Nothing to do — all images already processed.")
        logger.info("All images complete, aggregating.")
        aggregate_to_averaged_csv(per_image_csv, averaged_csv)
        print(f"density_averaged.csv written to: {averaged_csv}")
        return per_image_csv, averaged_csv

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, pending)
    batch_size = max(1, int(batch_size))

    batched_tasks = [
        tasks[i:i + batch_size]
        for i in range(0, len(tasks), batch_size)
    ]

    logger.info(
        f"Starting evaluation | images={pending} | "
        f"workers={num_workers} | batch_size={batch_size} | timeout={timeout}s"
    )
    print(
        f"Workers: {num_workers}  |  Images: {pending}  |  "
        f"Batch size: {batch_size} ({len(batched_tasks)} batches)\n"
    )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    failed   = 0
    timedout = 0

    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(_process_image_batch, batch): batch
            for batch in batched_tasks
        }

        with tqdm.tqdm(
            total=pending,
            desc="Computing pixel densities",
            unit="img",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for future in cf.as_completed(future_to_batch):
                batch = future_to_batch[future]
                first_subset = batch[0][0]
                first_file = Path(batch[0][3]).name
                batch_timeout = max(timeout, timeout * len(batch))

                try:
                    rows, batch_failed = future.result(timeout=batch_timeout)
                    if rows:
                        append_image_rows(per_image_csv, rows)
                    failed += batch_failed
                except cf.TimeoutError:
                    timedout += len(batch)
                    logger.error(
                        f"TIMEOUT ({batch_timeout}s) batch starting "
                        f"{first_subset}/{first_file} — skipped {len(batch)} images"
                    )
                    pbar.set_postfix_str(
                        f"timeout batch: {first_file[:20]}",
                        refresh=True,
                    )
                except Exception as exc:
                    failed += len(batch)
                    logger.error(
                        f"ERROR batch starting {first_subset}/{first_file}: {exc}"
                    )

                pbar.update(len(batch))

    # ── Summary ───────────────────────────────────────────────────────────────
    issues = failed + timedout
    if issues:
        print(f"\n  {failed} failed, {timedout} timed out. "
              f"See {LOG_FILE} for details.")
    else:
        print("\n  All images processed successfully.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\nAggregating results ...")
    aggregate_to_averaged_csv(per_image_csv, averaged_csv)
    logger.info(f"Aggregation complete: {averaged_csv}")
    print(f"  density_per_image.csv : {per_image_csv}")
    print(f"  density_averaged.csv  : {averaged_csv}")

    return per_image_csv, averaged_csv


# ──────────────────────────────────────────────────────────────────────────────
# Terminal table
# ──────────────────────────────────────────────────────────────────────────────

def print_tables(averaged_csv: Path):
    """Print mean and std per subset, grouped by dataset."""
    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by dataset
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)

    cw = {"datatype": 12, "n": 8, "mean": 12, "std": 12}
    header = (
        f"{'Datatype':<{cw['datatype']}}"
        f"  {'N':>{cw['n']}}"
        f"  {'Mean Density':>{cw['mean']}}"
        f"  {'Std Density':>{cw['std']}}"
    )
    width = len(header)

    for dataset, dataset_rows in sorted(groups.items()):
        print(f"\n{'=' * width}")
        print(f"  {dataset}")
        print(f"{'=' * width}")
        print(header)
        print("-" * width)
        for row in sorted(dataset_rows, key=lambda r: r["datatype"]):
            print(
                f"{row['datatype']:<{cw['datatype']}}"
                f"  {row['num_samples']:>{cw['n']}}"
                f"  {float(row['mean_density']):>{cw['mean']}.6f}"
                f"  {float(row['std_density']):>{cw['std']}.6f}"
            )
        print("-" * width)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers — filter & ordering
# ──────────────────────────────────────────────────────────────────────────────

# Consistent colour per dataset
DATASET_COLORS = {
    "MetalSet":   "#2E86AB",
    "ViaSet":     "#E07B39",
    "StdContact": "#6A994E",
    "StdMetal":   "#9B5DE5",
}
DEFAULT_COLOR = "#888888"

# Canonical display order for datasets across all plots
DATASET_ORDER = ["MetalSet", "ViaSet", "StdMetal", "StdContact"]


def _resolve_filters(datatypes_arg, datasets_arg):
    """
    Normalise --datatypes / --datasets CLI values to lowercase sets.
    Returns (dt_filter, ds_filter) where either may be None (= no filter).
    """
    dt_filter = {d.lower() for d in datatypes_arg} if datatypes_arg else None
    ds_filter = {d.lower() for d in datasets_arg}  if datasets_arg  else None
    return dt_filter, ds_filter


def _apply_dataset_filter_and_order(datasets: list, ds_filter) -> list:
    """
    Apply optional dataset filter then sort by DATASET_ORDER.
    Any dataset not in DATASET_ORDER is appended at the end, alphabetically.
    """
    if ds_filter is not None:
        datasets = [ds for ds in datasets if ds.lower() in ds_filter]
    known   = [ds for ds in DATASET_ORDER if ds in datasets]
    unknown = sorted(ds for ds in datasets if ds not in DATASET_ORDER)
    return known + unknown


def _apply_datatype_filter(datatypes: list, dt_filter) -> list:
    """Apply optional datatype filter (case-insensitive). Preserves existing order."""
    if dt_filter is None:
        return datatypes
    return [dt for dt in datatypes if dt.lower() in dt_filter]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (reads from density_per_image.csv)
# ──────────────────────────────────────────────────────────────────────────────

def plot_density_histograms(
    per_image_csv: str,
    bins:          int  = 40,
    save_dir:      str  = None,
    dt_filter:     set  = None,
    ds_filter:     set  = None,
):
    """
    Generate one figure per datatype.  Each figure contains one subplot per
    dataset that has data for that datatype.

    Within each subplot the histogram for that (dataset, datatype) pair is
    shown, with mean and ±1 std marked as vertical lines.

    save_dir=None  → show only
    save_dir=<str> → save PNG files to that directory
    dt_filter      → set of lowercase datatype strings to include, or None for all
    ds_filter      → set of lowercase dataset strings to include, or None for all
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-image CSV ────────────────────────────────────────────────────
    records = {}   # (dataset, datatype) → list of float densities
    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("pixel_density", "")
            if v == "":
                continue
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            key = (row["dataset"], row["datatype"])
            records.setdefault(key, []).append(fv)

    # ── Derive filtered + ordered dimension lists ─────────────────────────────
    datatypes = _apply_datatype_filter(
        sorted(set(dt for _, dt in records.keys())), dt_filter)
    datasets  = _apply_dataset_filter_and_order(
        list(set(ds for ds, _ in records.keys())), ds_filter)

    for datatype in datatypes:
        active = [ds for ds in datasets if (ds, datatype) in records]
        if not active:
            continue

        n_cols = len(active)
        fig, axes = plt.subplots(
            1, n_cols, figsize=(5 * n_cols, 4),
            constrained_layout=False,
        )
        fig.subplots_adjust(top=0.88, bottom=0.12,
                            left=0.07, right=0.97, wspace=0.35)

        if n_cols == 1:
            axes = [axes]

        for ax, dataset in zip(axes, active):
            vals  = records[(dataset, datatype)]
            color = DATASET_COLORS.get(dataset, DEFAULT_COLOR)
            n     = len(vals)
            mean  = float(np.mean(vals))
            std   = float(np.std(vals))

            ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white")

            # Mean and ±1 std vertical lines
            ax.axvline(mean,       color="#222222", linewidth=1.8,
                       linestyle="-",  label=f"Mean {mean:.4f}")
            ax.axvline(mean - std, color="#222222", linewidth=1.2,
                       linestyle="--", label=f"±1σ  {std:.4f}")
            ax.axvline(mean + std, color="#222222", linewidth=1.2,
                       linestyle="--")

            ax.set_title(f"{dataset}\nn={n:,}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Pixel Density", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(fontsize=7, framealpha=0.85)
            ax.grid(True, alpha=0.2, linestyle="--")

        fig.suptitle(
            f"Pixel Density Distribution  |  Datatype: {datatype}",
            fontsize=12, fontweight="bold", y=0.97,
        )

        if save_dir is not None:
            out = save_dir / f"density_{datatype}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

        plt.show()
        plt.close(fig)


def plot_mean_std_by_datatype(
    averaged_csv: str,
    save_dir:     str = None,
    dt_filter:    set = None,
    ds_filter:    set = None,
):
    """
    Generate one figure per datatype with dataset mean density and std error bars.

    Each datatype gets its own plot, where x-axis is dataset and y-axis is
    mean density with ±std shown as error bars.

    dt_filter / ds_filter: sets of lowercase strings to include, or None for all.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "subset":   row["subset"],
                    "dataset":  row["dataset"],
                    "datatype": row["datatype"],
                    "mean":     float(row["mean_density"]),
                    "std":      float(row["std_density"]),
                    "n":        int(row["num_samples"]),
                }
            )

    if not records:
        print("No averaged rows available to plot.")
        return

    datatypes = _apply_datatype_filter(
        sorted({r["datatype"] for r in records}), dt_filter)

    for datatype in datatypes:
        rows = [r for r in records if r["datatype"] == datatype]
        if not rows:
            continue

        present_datasets = _apply_dataset_filter_and_order(
            [r["dataset"] for r in rows], ds_filter)
        rows = [r for r in rows if r["dataset"] in present_datasets]
        rows.sort(key=lambda r: present_datasets.index(r["dataset"]))

        if not rows:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(rows))

        for i, row in enumerate(rows):
            color = DATASET_COLORS.get(row["dataset"], DEFAULT_COLOR)
            ax.errorbar(
                x[i],
                row["mean"],
                yerr=row["std"],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=6,
                markersize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([r["dataset"] for r in rows], rotation=25, ha="right")
        ax.set_ylabel("Mean Pixel Density")
        ax.set_xlabel("Dataset")
        ax.set_title(f"Mean ± Std Pixel Density by Dataset | Datatype: {datatype}")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.margins(x=0.08)

        for i, row in enumerate(rows):
            ax.text(
                x[i],
                row["mean"] + row["std"] + 0.005,
                f"n={row['n']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        if save_dir is not None:
            out = Path(save_dir) / f"mean_std_{datatype}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

        plt.show()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# SNR
# ──────────────────────────────────────────────────────────────────────────────

def calculate_snr(output_dir: Path) -> list:
    """Compute SNR = mean/std per (dataset, datatype) from density_averaged.csv.

    Returns a list of dicts with keys: dataset, datatype, snr.
    Rows where std == 0 are skipped to avoid division by zero.
    Only (dataset, datatype) combinations present in the CSV are included.
    """
    averaged_csv = output_dir / "density_averaged.csv"
    df = pd.read_csv(averaged_csv)

    snr_results = []
    for _, row in df.iterrows():
        std = float(row["std_density"])
        if std == 0:
            continue
        snr_results.append({
            "dataset":  row["dataset"],
            "datatype": row["datatype"],
            "snr":      float(row["mean_density"]) / std,
        })

    return snr_results


def bar_plot_snr(
    snr_results: list,
    save_dir:    str = None,
    dt_filter:   set = None,
    ds_filter:   set = None,
):
    """
    Bar chart of SNR grouped by datatype, with one bar per dataset.
    Datasets are ordered by DATASET_ORDER; bars are placed side-by-side.

    dt_filter / ds_filter: sets of lowercase strings to include, or None for all.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if not snr_results:
        print("No SNR results to plot.")
        return

    df_snr = pd.DataFrame(snr_results)

    datatypes = _apply_datatype_filter(
        sorted(df_snr["datatype"].unique()), dt_filter)
    datasets  = _apply_dataset_filter_and_order(
        list(df_snr["dataset"].unique()), ds_filter)

    if not datatypes or not datasets:
        print("No data remaining after applying filters.")
        return

    n_ds   = len(datasets)
    n_dt   = len(datatypes)
    width  = 0.8 / n_ds
    x      = np.arange(n_dt)

    fig, ax = plt.subplots(figsize=(max(8, n_dt * 1.5), 6))

    for i, ds in enumerate(datasets):
        df_ds  = df_snr[df_snr["dataset"] == ds].set_index("datatype")
        values = [float(df_ds.loc[dt, "snr"]) if dt in df_ds.index else 0.0
                  for dt in datatypes]
        offsets = x - 0.4 + (i + 0.5) * width
        color   = DATASET_COLORS.get(ds, DEFAULT_COLOR)
        bars    = ax.bar(offsets, values, width=width * 0.9,
                         label=ds, color=color, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datatypes, rotation=40, ha="right")
    ax.set_xlabel("Datatype")
    ax.set_ylabel("SNR  (mean / std)")
    ax.set_title("Signal-to-Noise Ratio by Dataset and Datatype")
    ax.legend(title="Dataset", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()

    if save_dir is not None:
        out = Path(save_dir) / "snr_bar_plot.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Expansion coefficient  (PixelILT density / Target density, per tile)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_expansion_coefficient(
    per_image_csv:      Path,
    min_target_density: float = DEFAULT_MIN_TARGET_DENSITY,
) -> pd.DataFrame:
    """
    Compute per-tile expansion coefficient = PixelILT density / Target density.

    Tiles where Target density <= min_target_density are excluded to avoid
    division instability (near-empty tiles where ILT has no meaningful input).

    Parameters
    ----------
    per_image_csv       : path to density_per_image.csv
    min_target_density  : exclusion floor for Target density (default 1e-4)

    Returns
    -------
    df_avg : DataFrame with columns:
                 dataset, n, mean_exp, std_exp, median_exp, p25, p75
             one row per dataset, ordered by DATASET_ORDER.
    """
    all_data = pd.read_csv(per_image_csv)

    df_pixelilt = (
        all_data[all_data["datatype"] == "PixelILT"]
        [["dataset", "filename", "pixel_density"]]
        .rename(columns={"pixel_density": "density_pixelilt"})
    )
    df_target = (
        all_data[all_data["datatype"] == "Target"]
        [["dataset", "filename", "pixel_density"]]
        .rename(columns={"pixel_density": "density_target"})
    )

    # Join on (dataset, filename) — the only safe pairing key
    df_paired = pd.merge(df_pixelilt, df_target,
                         on=["dataset", "filename"], how="inner")

    # Drop near-zero Target tiles
    df_valid = df_paired[
        df_paired["density_target"] > min_target_density
    ].copy()

    n_total   = len(df_paired)
    n_dropped = n_total - len(df_valid)
    print(f"  Expansion: {n_total:,} paired tiles, "
          f"dropped {n_dropped:,} with Target density ≤ {min_target_density} "
          f"({100 * n_dropped / max(n_total, 1):.1f}%)")

    df_valid["exp_coeff"] = (
        df_valid["density_pixelilt"] / df_valid["density_target"]
    )

    df_avg = (
        df_valid.groupby("dataset")["exp_coeff"]
        .agg(
            n          = "count",
            mean_exp   = "mean",
            std_exp    = "std",
            median_exp = "median",
            p25        = lambda x: x.quantile(0.25),
            p75        = lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    # Apply canonical dataset ordering
    order  = [ds for ds in DATASET_ORDER if ds in df_avg["dataset"].values]
    df_avg = (
        df_avg.set_index("dataset")
        .reindex(order)
        .reset_index()
    )

    return df_avg


def print_expansion_table(df_avg: pd.DataFrame):
    """Print the expansion coefficient summary table to the terminal."""
    cw = {"dataset": 12, "n": 8, "mean": 10, "std": 10,
          "median": 10, "p25": 10, "p75": 10}
    header = (
        f"{'Dataset':<{cw['dataset']}}"
        f"  {'N':>{cw['n']}}"
        f"  {'Mean':>{cw['mean']}}"
        f"  {'Std':>{cw['std']}}"
        f"  {'Median':>{cw['median']}}"
        f"  {'P25':>{cw['p25']}}"
        f"  {'P75':>{cw['p75']}}"
    )
    width = len(header)
    print(f"\n{'=' * width}")
    print("  Expansion Coefficient  (PixelILT density / Target density)")
    print(f"{'=' * width}")
    print(header)
    print("-" * width)
    for _, row in df_avg.iterrows():
        print(
            f"{row['dataset']:<{cw['dataset']}}"
            f"  {int(row['n']):>{cw['n']},}"
            f"  {row['mean_exp']:>{cw['mean']}.4f}"
            f"  {row['std_exp']:>{cw['std']}.4f}"
            f"  {row['median_exp']:>{cw['median']}.4f}"
            f"  {row['p25']:>{cw['p25']}.4f}"
            f"  {row['p75']:>{cw['p75']}.4f}"
        )
    print("-" * width)


def plot_expansion_coefficient(
    df_avg:    pd.DataFrame,
    save_dir:  str = None,
    ds_filter: set = None,
):
    """
    Plot mean ± std of the per-tile expansion coefficient for each dataset.

    Parameters
    ----------
    df_avg    : output of calculate_expansion_coefficient()
    save_dir  : directory to save PNG, or None to show only
    ds_filter : set of lowercase dataset names to include, or None for all
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Apply optional dataset filter then re-order
    datasets = _apply_dataset_filter_and_order(
        list(df_avg["dataset"].values), ds_filter)
    df_plt = (
        df_avg[df_avg["dataset"].isin(datasets)]
        .set_index("dataset")
        .reindex(datasets)
        .reset_index()
    )

    if df_plt.empty:
        print("No data remaining after applying dataset filter.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_plt))

    for i, row in df_plt.iterrows():
        color = DATASET_COLORS.get(row["dataset"], DEFAULT_COLOR)
        ax.errorbar(
            x[i], row["mean_exp"],
            yerr=row["std_exp"],
            fmt="o",
            color=color, ecolor=color,
            elinewidth=2, capsize=6, markersize=9,
            label=row["dataset"],
        )
        ax.text(
            x[i],
            row["mean_exp"] + row["std_exp"] + 0.05,
            f"μ={row['mean_exp']:.2f}\nσ={row['std_exp']:.2f}\nn={int(row['n']):,}",
            ha="center", va="bottom", fontsize=8, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plt["dataset"], fontsize=11)
    ax.set_ylabel("Expansion Coefficient  (PixelILT density / Target density)",
                  fontsize=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_title("Per-Tile ILT Expansion Coefficient  |  Mean ± Std by Dataset",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.margins(x=0.15)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_dir is not None:
        out = Path(save_dir) / "expansion_coefficient.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pixel density study for LithoBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--evaluate",           action="store_true",
                   help="Run the parallel pixel density evaluation.")
    p.add_argument("--aggregate",          action="store_true",
                   help="Re-aggregate density_per_image.csv → density_averaged.csv.")
    p.add_argument("--plot",               action="store_true",
                   help="Generate density histograms from density_per_image.csv.")
    p.add_argument("--plot-mean-std",      action="store_true",
                   help="Generate mean±std comparison plots per datatype.")
    p.add_argument("--tables",             action="store_true",
                   help="Print per-subset mean/std tables to terminal.")
    p.add_argument("--snr",                action="store_true",
                   help="Calculate and print SNR (mean/std) from density_averaged.csv.")
    p.add_argument("--snr-plot",           action="store_true",
                   help="Generate bar plot of SNR from density_averaged.csv.")
    p.add_argument("--expansion",          action="store_true",
                   help="Calculate and print per-tile expansion coefficient summary "
                        "(PixelILT density / Target density).")
    p.add_argument("--expansion-plot",     action="store_true",
                   help="Generate mean±std expansion coefficient plot.")
    p.add_argument("--workers",            type=int, default=None,
                   help="Worker processes (default: cpu_count - 1).")
    p.add_argument("--batch-size",         type=int, default=32,
                   help="Images per worker task batch (default: 32).")
    p.add_argument("--samples",            type=int, default=None,
                   help="Cap images per subset — None means all images.")
    p.add_argument("--timeout",            type=int, default=60,
                   help="Per-image timeout in seconds (default: 60).")
    p.add_argument("--force",              action="store_true",
                   help="Delete existing CSVs and start fresh.")
    p.add_argument("--save-plots",         action="store_true",
                   help="Save plot PNGs to the output directory.")
    p.add_argument("--bins",               type=int, default=40,
                   help="Number of histogram bins (default: 40).")
    p.add_argument("--datatypes",          nargs="+", default=None, metavar="DATATYPE",
                   help="Restrict all plots to these datatypes (case-insensitive). "
                        "E.g. --datatypes PixelILT Resist")
    p.add_argument("--datasets",           nargs="+", default=None, metavar="DATASET",
                   help="Restrict all plots to these datasets (case-insensitive). "
                        "E.g. --datasets MetalSet ViaSet")
    p.add_argument("--min-target-density", type=float,
                   default=DEFAULT_MIN_TARGET_DENSITY,
                   help="Exclusion floor for Target density when computing the "
                        f"expansion coefficient (default: {DEFAULT_MIN_TARGET_DENSITY}).")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not any([args.evaluate, args.aggregate, args.plot, args.plot_mean_std,
                args.tables, args.snr, args.snr_plot,
                args.expansion, args.expansion_plot]):
        print("No action specified. Use --evaluate, --aggregate, --plot, "
              "--plot-mean-std, --tables, --snr, --snr-plot, "
              "--expansion, or --expansion-plot.")
        print("Run with --help for full usage.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)
    logger = get_logger()

    per_image_csv = OUTPUT_DIR / "density_per_image.csv"
    averaged_csv  = OUTPUT_DIR / "density_averaged.csv"

    # Resolve filters once; passed to every plot/print function
    dt_filter, ds_filter = _resolve_filters(args.datatypes, args.datasets)

    logger.info(
        f"Session start | evaluate={args.evaluate} | aggregate={args.aggregate} | "
        f"plot={args.plot} | plot_mean_std={args.plot_mean_std} | "
        f"tables={args.tables} | snr={args.snr} | snr_plot={args.snr_plot} | "
        f"expansion={args.expansion} | expansion_plot={args.expansion_plot} | "
        f"workers={args.workers} | batch_size={args.batch_size} | "
        f"samples={args.samples} | force={args.force} | "
        f"datatypes={args.datatypes} | datasets={args.datasets} | "
        f"min_target_density={args.min_target_density}"
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate:
        num_samples = args.samples if args.samples is not None else NUM_SAMPLES
        per_image_csv, averaged_csv = run_evaluation(
            data_dict   = DATA_DICT,
            output_dir  = OUTPUT_DIR,
            num_workers = args.workers,
            num_samples = num_samples,
            batch_size  = args.batch_size,
            force       = args.force,
            timeout     = args.timeout,
        )

    # ── Standalone aggregation ────────────────────────────────────────────────
    if args.aggregate:
        if not per_image_csv.exists():
            print("density_per_image.csv not found. Run --evaluate first.")
        else:
            print("Aggregating density_per_image.csv ...")
            aggregate_to_averaged_csv(per_image_csv, averaged_csv)
            print(f"Done. density_averaged.csv written to: {averaged_csv}")
            logger.info(f"Standalone aggregation complete: {averaged_csv}")

    # ── Terminal tables ───────────────────────────────────────────────────────
    if args.tables:
        if not averaged_csv.exists():
            print("density_averaged.csv not found. "
                  "Run --evaluate or --aggregate first.")
        else:
            print_tables(averaged_csv)

    # ── Plotting ──────────────────────────────────────────────────────────────
    if args.plot:
        if not per_image_csv.exists():
            print("density_per_image.csv not found. Run --evaluate first.")
        else:
            print("\nGenerating density histograms ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_density_histograms(
                str(per_image_csv),
                bins      = args.bins,
                save_dir  = save_dir,
                dt_filter = dt_filter,
                ds_filter = ds_filter,
            )

    # ── Mean/std comparison plotting ──────────────────────────────────────────
    if args.plot_mean_std:
        if not averaged_csv.exists():
            print("density_averaged.csv not found. "
                  "Run --evaluate or --aggregate first.")
        else:
            print("\nGenerating mean±std comparison plots ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_mean_std_by_datatype(
                str(averaged_csv),
                save_dir  = save_dir,
                dt_filter = dt_filter,
                ds_filter = ds_filter,
            )

    # ── SNR ───────────────────────────────────────────────────────────────────
    if args.snr:
        if not averaged_csv.exists():
            print("density_averaged.csv not found. "
                  "Run --evaluate or --aggregate first.")
        else:
            print("\nCalculating SNR from density_averaged.csv ...")
            snr_results = calculate_snr(OUTPUT_DIR)
            for res in snr_results:
                if dt_filter and res["datatype"].lower() not in dt_filter:
                    continue
                if ds_filter and res["dataset"].lower() not in ds_filter:
                    continue
                print(f"Dataset: {res['dataset']:12s}  "
                      f"Datatype: {res['datatype']:12s}  "
                      f"SNR: {res['snr']:.4f}")

    # ── SNR plot ──────────────────────────────────────────────────────────────
    if args.snr_plot:
        if not averaged_csv.exists():
            print("density_averaged.csv not found. "
                  "Run --evaluate or --aggregate first.")
        else:
            snr_results = calculate_snr(OUTPUT_DIR)
            print("\nGenerating SNR bar plot ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            bar_plot_snr(
                snr_results,
                save_dir  = save_dir,
                dt_filter = dt_filter,
                ds_filter = ds_filter,
            )

    # ── Expansion coefficient ─────────────────────────────────────────────────
    # Both --expansion and --expansion-plot share a single calculation call
    # so the CSV is only read once even when both flags are passed together.
    if args.expansion or args.expansion_plot:
        if not per_image_csv.exists():
            print("density_per_image.csv not found. Run --evaluate first.")
        else:
            print("\nCalculating expansion coefficients ...")
            df_exp = calculate_expansion_coefficient(
                per_image_csv,
                min_target_density=args.min_target_density,
            )

            if args.expansion:
                print_expansion_table(df_exp)

            if args.expansion_plot:
                print("\nGenerating expansion coefficient plot ...")
                save_dir = str(OUTPUT_DIR) if args.save_plots else None
                plot_expansion_coefficient(
                    df_exp,
                    save_dir  = save_dir,
                    ds_filter = ds_filter,
                )

    logger.info("Session end")
