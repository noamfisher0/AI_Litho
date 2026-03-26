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
  --evaluate            run the parallel pixel density evaluation
  --aggregate           re-aggregate density_per_image.csv → density_averaged.csv
  --plot                generate histograms from density_per_image.csv
  --tables              print per-subset mean/std tables to terminal
  --workers N           number of worker processes (default: cpu_count - 1)
  --samples N           cap images per subset (default: None = all images)
  --force               delete existing CSVs and start fresh
  --save-plots          save histogram PNGs to the output directory
  --bins N              number of histogram bins (default: 40)

Usage examples
--------------
  python densitystudy.py --evaluate
  python densitystudy.py --evaluate --workers 8 --samples 500
  python densitystudy.py --aggregate
  python densitystudy.py --plot --save-plots
  python densitystudy.py --evaluate --aggregate --plot
  python densitystudy.py --evaluate --force --workers 8
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

    logger.info(
        f"Starting evaluation | images={pending} | "
        f"workers={num_workers} | timeout={timeout}s"
    )
    print(f"Workers: {num_workers}  |  Images to process: {pending}\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    failed   = 0
    timedout = 0

    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(_process_image, task): task
            for task in tasks
        }

        with tqdm.tqdm(
            total=pending,
            desc="Computing pixel densities",
            unit="img",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for future in cf.as_completed(future_to_task):
                task      = future_to_task[future]
                subset_key = task[0]
                filename   = Path(task[3]).name

                try:
                    row = future.result(timeout=timeout)
                    if row:
                        append_image_rows(per_image_csv, [row])
                    else:
                        failed += 1
                        logger.warning(f"No result for {subset_key}/{filename}")
                except cf.TimeoutError:
                    timedout += 1
                    logger.error(
                        f"TIMEOUT ({timeout}s) {subset_key}/{filename} — skipped"
                    )
                    pbar.set_postfix_str(f"timeout: {filename[:20]}", refresh=True)
                except Exception as exc:
                    failed += 1
                    logger.error(f"ERROR {subset_key}/{filename}: {exc}")

                pbar.update(1)

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
# Plotting  (reads from density_per_image.csv)
# ──────────────────────────────────────────────────────────────────────────────

# Consistent colour per dataset
DATASET_COLORS = {
    "MetalSet":   "#2E86AB",
    "ViaSet":     "#E07B39",
    "StdContact": "#6A994E",
    "StdMetal":   "#9B5DE5",
}
DEFAULT_COLOR = "#888888"


def plot_density_histograms(
    per_image_csv: str,
    bins:      int  = 40,
    save_dir:  str  = None,
):
    """
    Generate one figure per datatype.  Each figure contains one subplot per
    dataset that has data for that datatype.

    Within each subplot the histogram for that (dataset, datatype) pair is
    shown, with mean and ±1 std marked as vertical lines.

    save_dir=None  → show only
    save_dir=<str> → save PNG files to that directory
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

    # ── Derive ordered dimension lists ────────────────────────────────────────
    datatypes = sorted(set(dt for _, dt in records.keys()))
    datasets  = sorted(set(ds for ds, _ in records.keys()))

    for datatype in datatypes:
        # Collect datasets that have data for this datatype
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pixel density study for LithoBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--evaluate",   action="store_true",
                   help="Run the parallel pixel density evaluation.")
    p.add_argument("--aggregate",  action="store_true",
                   help="Re-aggregate density_per_image.csv → density_averaged.csv.")
    p.add_argument("--plot",       action="store_true",
                   help="Generate density histograms from density_per_image.csv.")
    p.add_argument("--tables",     action="store_true",
                   help="Print per-subset mean/std tables to terminal.")
    p.add_argument("--workers",    type=int, default=None,
                   help="Worker processes (default: cpu_count - 1).")
    p.add_argument("--samples",    type=int, default=None,
                   help="Cap images per subset — None means all images.")
    p.add_argument("--timeout",    type=int, default=60,
                   help="Per-image timeout in seconds (default: 60).")
    p.add_argument("--force",      action="store_true",
                   help="Delete existing CSVs and start fresh.")
    p.add_argument("--save-plots", action="store_true",
                   help="Save histogram PNGs to the output directory.")
    p.add_argument("--bins",       type=int, default=40,
                   help="Number of histogram bins (default: 40).")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not any([args.evaluate, args.aggregate, args.plot, args.tables]):
        print("No action specified. Use --evaluate, --aggregate, --plot, or --tables.")
        print("Run with --help for full usage.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)
    logger = get_logger()

    per_image_csv = OUTPUT_DIR / "density_per_image.csv"
    averaged_csv  = OUTPUT_DIR / "density_averaged.csv"

    logger.info(
        f"Session start | evaluate={args.evaluate} | aggregate={args.aggregate} | "
        f"plot={args.plot} | tables={args.tables} | workers={args.workers} | "
        f"samples={args.samples} | force={args.force}"
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate:
        num_samples = args.samples if args.samples is not None else NUM_SAMPLES
        per_image_csv, averaged_csv = run_evaluation(
            data_dict   = DATA_DICT,
            output_dir  = OUTPUT_DIR,
            num_workers = args.workers,
            num_samples = num_samples,
            force       = args.force,
            timeout     = args.timeout,
        )

    # ── Standalone aggregation ────────────────────────────────────────────────
    if args.aggregate:
        if not per_image_csv.exists():
            print(f"density_per_image.csv not found. Run --evaluate first.")
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
                bins=args.bins,
                save_dir=save_dir,
            )

    logger.info("Session end")
