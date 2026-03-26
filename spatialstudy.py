"""
spatialstudy.py
===============
Spatial resolution study for the LithoBench dataset.

Architecture
------------
Evaluation uses a flat image-level work queue fed to a
ProcessPoolExecutor.  Each worker processes one image at a time and
returns one result row.  The main process writes each result to
results_per_image.csv immediately, giving image-level checkpointing.
At the end the main process aggregates the per-image CSV into
results_average.csv, which the plotting functions consume.

Two output CSVs
---------------
  results_per_image.csv   — one row per (image × method × resolution)
                            written continuously; used for checkpointing
                            and provides full metric distributions
  results_average.csv    — one row per (subset × method × resolution)
                            computed from results_per_image.csv at the
                            end of a completed or resumed run

Resume behaviour
----------------
On startup the main process reads results_per_image.csv and builds the
set of already-completed (subset, filename) pairs.  Only the remaining
images are added to the work queue.  Stop the run at any time with
Ctrl+C — the next run resumes from exactly where it left off.
Use --force to discard existing results and start fresh.

Flags
-----
  --evaluate            run the image-level parallel evaluation
  --plot                generate figures from results_average.csv
  --tables              print per-subset metric tables to the terminal
  --workers N           number of worker processes (default: cpu_count-1)
  --force               delete existing CSVs and start fresh
  --save-plots          save PNG figures to the output directory
  --timeout N           per-image timeout in seconds (default: 120)

Usage examples
--------------
  python spatialstudy.py --evaluate --workers 8
  python spatialstudy.py --evaluate --workers 8 --timeout 60
  python spatialstudy.py --plot --save-plots
  python spatialstudy.py --evaluate --plot --force
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
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
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean, resize


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "lithobench-main"
OUTPUT_DIR   = PROJECT_ROOT / "resolution_study_output_fourrier"
LOG_FILE     = OUTPUT_DIR / "spatial_study.log"

# Set to an integer to cap images per subset (useful for test runs).
# Set to None to process every image in every subset.
NUM_SAMPLES = 500

TARGET_RESOLUTIONS = [1024, 512, 256, 128]

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

# One row per (image × method × resolution)
PER_IMAGE_FIELDS = [
    "subset", "dataset", "datatype",
    "filename", "resolution", "method",
    "psnr", "ssim", "mse", "hf_ratio",
]

# One row per (subset × method × resolution) — aggregated
AVERAGED_FIELDS = [
    "subset", "dataset", "datatype", "num_samples",
    "resolution", "method",
    "psnr", "psnr_std",
    "ssim", "ssim_std",
    "mse",  "mse_std",
    "hf_ratio", "hf_ratio_std",
]


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Configure the 'spatialstudy' logger to write to a rotating file.
    Safe to call from both the main process and worker processes —
    each call is idempotent (handlers are not duplicated).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("spatialstudy")
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
    return logging.getLogger("spatialstudy")


# ──────────────────────────────────────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: Path):
    """
    Load an image with cv2 (no GUI involvement) and return a float32 array
    in RGB channel order normalised to [0, 1].
    Returns None on failure.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = img[..., ::-1]          # BGR  → RGB
        elif img.shape[2] == 4:
            img = img[..., [2, 1, 0, 3]]  # BGRA → RGBA
    img = img.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────

def to_grayscale(image):
    if image.ndim == 3:
        return image[..., :3].mean(axis=-1)
    return image


# ──────────────────────────────────────────────────────────────────────────────
# Downsampling methods
# ──────────────────────────────────────────────────────────────────────────────

def downsample_pointwise(image, target_size: int):
    """
    Point-Wise (Nearest-Neighbour) Downsampling
    -------------------------------------------
    Selects every N-th pixel (stride = original / target).  Fast and
    zero blurring, but susceptible to aliasing when scale factor is large.
    """
    h, w = image.shape[:2]
    sh, sw = h // target_size, w // target_size
    return image[::sh, ::sw][:target_size, :target_size]


def downsample_average(image, target_size: int):
    """
    Average (Block-Mean) Downsampling
    ----------------------------------
    Replaces each non-overlapping block with its mean — acts as a box
    low-pass filter, suppressing aliasing at the cost of slight blurring.
    Requires the original size to be divisible by the target size.
    """
    h, w = image.shape[:2]
    fh, fw = h // target_size, w // target_size
    factors = (fh, fw, 1) if image.ndim == 3 else (fh, fw)
    return downscale_local_mean(image, factors).astype(image.dtype)


def downsample_fourier(image, target_size: int):
    """
    Fourier (Frequency-Domain) Downsampling
    ----------------------------------------
    Crops the 2-D FFT spectrum to the central target×target coefficients,
    enforcing the Nyquist limit exactly.  Theoretically optimal
    anti-aliasing; most computationally expensive of the three methods.
    """
    def _channel(ch):
        F    = np.fft.fftshift(np.fft.fft2(ch.astype(np.float64)))
        cy, cx = F.shape[0] // 2, F.shape[1] // 2
        half = target_size // 2
        Fc   = F[cy - half:cy + half, cx - half:cx + half]
        scale = (target_size ** 2) / (F.shape[0] * F.shape[1])
        return (np.fft.ifft2(np.fft.ifftshift(Fc)).real * scale).astype(np.float32)

    result = (np.stack([_channel(image[..., c]) for c in range(image.shape[2])], axis=-1)
              if image.ndim == 3 else _channel(image))

    lo, hi   = image.min(), image.max()
    rlo, rhi = result.min(), result.max()
    if rhi - rlo > 1e-8:
        result = (result - rlo) / (rhi - rlo) * (hi - lo) + lo
    return result.astype(np.float32)


METHODS = {
    "PointWise": downsample_pointwise,
    "Average":   downsample_average,
    "Fourier":   downsample_fourier,
}


# ──────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def upsample_to_original(image, original_size: int):
    """Nearest-neighbour upscale back to original_size — introduces no
    additional smoothing so only the downsampling artefacts are measured."""
    return resize(image, (original_size, original_size),
                  order=0, preserve_range=True,
                  anti_aliasing=False).astype("float32")


def compute_mse(orig, recon) -> float:
    return float(np.mean(
        (orig.astype(np.float64) - recon.astype(np.float64)) ** 2
    ))


def compute_psnr(orig, recon) -> float:
    """Returns nan for identical or flat images — these appear as gaps on
    plots, which is the correct behaviour (no meaningful PSNR to report)."""
    mse = compute_mse(orig, recon)
    if mse == 0:
        return float("nan")
    data_range = float(orig.max() - orig.min())
    if data_range == 0:
        return float("nan")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def compute_ssim(orig, recon) -> float:
    o, r = orig.astype(np.float64), recon.astype(np.float64)
    L    = float(o.max() - o.min()) or 1.0
    C1, C2   = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_o, mu_r   = o.mean(), r.mean()
    sig_o  = np.sqrt(np.mean((o - mu_o) ** 2))
    sig_r  = np.sqrt(np.mean((r - mu_r) ** 2))
    sig_or = np.mean((o - mu_o) * (r - mu_r))
    num = (2 * mu_o * mu_r + C1) * (2 * sig_or  + C2)
    den = (mu_o**2 + mu_r**2 + C1) * (sig_o**2 + sig_r**2 + C2)
    return float(num / den)


def compute_hf_ratio(orig, recon) -> float:
    """Fraction of high-frequency energy (outside central 50% of spectrum)
    retained after downsampling+upsampling.  1 = perfect retention."""
    def hf_energy(img):
        F = np.abs(np.fft.fftshift(np.fft.fft2(
            to_grayscale(img).astype(np.float64)
        )))
        h, w  = F.shape
        mask  = np.ones((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = False
        return float(np.sum(F[mask] ** 2))

    orig_hf = hf_energy(orig)
    return float(hf_energy(recon) / orig_hf) if orig_hf > 0 else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Worker function  (one call per image — runs in a subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _process_image(task: tuple):
    """
    Process a single image across all methods and resolutions.

    Parameters
    ----------
    task : (subset, dataset, datatype, file_path_str, resolutions, log_file)

    Returns
    -------
    List of per-image CSV row dicts on success, or None on failure.
    """
    subset, dataset, datatype, file_path_str, resolutions, log_file = task

    import matplotlib
    matplotlib.use("Agg")   # non-GUI — must be set before pyplot in each worker

    logger = setup_logging(Path(log_file))
    file_path = Path(file_path_str)
    logger.debug(f"Processing {subset}/{file_path.name}")

    img = load_image(file_path)
    if img is None:
        logger.warning(f"Failed to load {file_path} — skipping")
        return None

    native_h  = img.shape[0]
    orig_gray = to_grayscale(img)
    rows      = []

    for method_name, method_fn in METHODS.items():
        for res in resolutions:

            # Skip degenerate case: target >= native resolution
            if res >= native_h:
                logger.debug(
                    f"SKIP {subset} | {method_name} | res={res} "
                    f"(native={native_h}) | {file_path.name}"
                )
                continue

            try:
                ds      = method_fn(img, res)
                up      = upsample_to_original(ds, native_h)
                up_gray = to_grayscale(up)

                psnr     = compute_psnr(orig_gray, up_gray)
                ssim     = compute_ssim(orig_gray, up_gray)
                mse      = compute_mse(orig_gray, up_gray)
                hf_ratio = compute_hf_ratio(img, up)

                rows.append({
                    "subset":     subset,
                    "dataset":    dataset,
                    "datatype":   datatype,
                    "filename":   file_path.name,
                    "resolution": res,
                    "method":     method_name,
                    "psnr":       "" if math.isnan(psnr)  else round(psnr,     6),
                    "ssim":       round(ssim,     6),
                    "mse":        round(mse,      6),
                    "hf_ratio":   round(hf_ratio, 6),
                })

            except Exception as exc:
                logger.warning(
                    f"ERROR {subset} | {method_name} | res={res} | "
                    f"{file_path.name}: {exc}"
                )

    logger.info(f"DONE {subset}/{file_path.name} — {len(rows)} rows")
    return rows if rows else None


# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_completed_images(csv_path: Path) -> set:
    """
    Return the set of (subset, filename) pairs already present in the
    per-image CSV.  Used to build the resume queue.
    """
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
    """
    Append per-image result rows to the per-image CSV.
    Called from the main process only — no locking required.
    """
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def aggregate_to_averaged_csv(per_image_csv: Path, averaged_csv: Path):
    """
    Read results_per_image.csv and write results_average.csv.
    Computes mean and std of each metric independently per
    (subset × method × resolution), ignoring nan/inf/empty values.
    Each metric is accumulated independently so a nan in one metric
    (e.g. PSNR for a perfect reconstruction) does not discard valid
    values of other metrics for the same image.
    """
    METRICS = ("psnr", "ssim", "mse", "hf_ratio")

    # Store all valid values per (key, metric) to compute both mean and std
    values = {}  # (subset, method, res) → {metric: [float, ...]}
    meta   = {}  # subset → (dataset, datatype)

    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["subset"], row["method"], int(row["resolution"]))
            if key not in values:
                values[key] = {m: [] for m in METRICS}
                meta[row["subset"]] = (row["dataset"], row["datatype"])

            for m in METRICS:
                raw = row.get(m, "")
                if raw == "" or raw is None:
                    continue
                v = float(raw)
                if math.isnan(v) or math.isinf(v):
                    continue
                values[key][m].append(v)

    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AVERAGED_FIELDS)
        w.writeheader()
        for (subset, method, res), vals in sorted(values.items()):
            dataset, datatype = meta[subset]
            n = max((len(vals[m]) for m in METRICS), default=0)

            def avg(m):
                v = vals[m]
                return round(float(np.mean(v)), 6) if v else ""

            def std(m):
                v = vals[m]
                # ddof=1 for sample std; fall back to 0.0 for single-image subsets
                return round(float(np.std(v, ddof=1 if len(v) > 1 else 0)), 6) if v else ""

            w.writerow({
                "subset":       subset,
                "dataset":      dataset,
                "datatype":     datatype,
                "num_samples":  n,
                "resolution":   res,
                "method":       method,
                "psnr":         avg("psnr"),
                "psnr_std":     std("psnr"),
                "ssim":         avg("ssim"),
                "ssim_std":     std("ssim"),
                "mse":          avg("mse"),
                "mse_std":      std("mse"),
                "hf_ratio":     avg("hf_ratio"),
                "hf_ratio_std": std("hf_ratio"),
            })

    return averaged_csv


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    data_dict:   dict,
    output_dir:  Path,
    num_workers: int  = None,
    force:       bool = False,
    timeout:     int  = 120,
):
    """
    Build a flat image-level work queue, dispatch to a ProcessPoolExecutor,
    stream results to results_per_image.csv, then aggregate to
    results_average.csv.

    - Perfect load balancing: all workers consume from the same queue
      regardless of which subset an image belongs to.
    - Image-level resume: on restart only images not yet in
      results_per_image.csv are re-queued.
    - Per-image timeout: a hung image is cancelled after `timeout` seconds
      and the worker is replaced automatically.
    - Single tqdm bar counting images, not subsets.
    """
    import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "results_per_image.csv"
    averaged_csv  = output_dir / "results_average.csv"
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
    logger.info(f"Already completed: {len(completed)} (subset, image) pairs")

    # ── Build flat work queue ─────────────────────────────────────────────────
    print("Scanning dataset directories ...")
    tasks      = []
    skipped    = 0
    subset_sizes = {}

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

        if NUM_SAMPLES is not None:
            # On resume: always include files already completed for this subset
            # so they are recognised and skipped, then top up with new samples
            # only if needed to reach NUM_SAMPLES total.
            already_done_names = {
                fname for (sub, fname) in completed if sub == subset_key
            }
            already_done_files = [f for f in files if f.name in already_done_names]
            remaining_files    = [f for f in files if f.name not in already_done_names]

            still_needed = max(0, NUM_SAMPLES - len(already_done_files))
            new_sample   = random.sample(
                remaining_files, min(still_needed, len(remaining_files))
            )
            files = already_done_files + new_sample

        subset_sizes[subset_key] = len(files)

        for f in files:
            if (subset_key, f.name) in completed:
                skipped += 1
                continue
            tasks.append((
                subset_key, dataset, datatype,
                str(f), TARGET_RESOLUTIONS, str(LOG_FILE),
            ))

    total_images   = sum(subset_sizes.values())
    pending_images = len(tasks)
    done_images    = total_images - pending_images - skipped  # already in CSV but outside sample

    if completed:
        print(
            f"\nResuming — {len(completed)} images already done, "
            f"{pending_images} remaining.\n"
        )
    else:
        print(f"\nStarting fresh — {pending_images} images to process.\n")

    if not tasks:
        print("Nothing to do — all images already processed.")
        logger.info("All images already complete, aggregating CSV.")
        aggregate_to_averaged_csv(per_image_csv, averaged_csv)
        print(f"Averaged CSV written to: {averaged_csv}")
        return averaged_csv

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, pending_images)

    logger.info(
        f"Starting evaluation | images={pending_images} | "
        f"workers={num_workers} | timeout={timeout}s"
    )
    print(f"Workers: {num_workers}  |  Images to process: {pending_images}\n")

    # ── Dispatch with ProcessPoolExecutor ─────────────────────────────────────
    failed_images = 0
    timed_out     = 0

    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks up front — executor manages the queue internally
        future_to_task = {
            executor.submit(_process_image, task): task
            for task in tasks
        }

        with tqdm.tqdm(
            total=pending_images,
            desc="Processing images",
            unit="img",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for future in cf.as_completed(future_to_task, timeout=None):
                task = future_to_task[future]
                subset_key = task[0]
                filename   = Path(task[3]).name

                try:
                    rows = future.result(timeout=timeout)
                    if rows:
                        append_image_rows(per_image_csv, rows)
                    else:
                        failed_images += 1
                        logger.warning(
                            f"No rows returned for {subset_key}/{filename}"
                        )
                except cf.TimeoutError:
                    timed_out += 1
                    logger.error(
                        f"TIMEOUT ({timeout}s) {subset_key}/{filename} — skipped"
                    )
                    pbar.set_postfix_str(f"timeout: {filename[:20]}", refresh=True)
                except Exception as exc:
                    failed_images += 1
                    logger.error(
                        f"ERROR {subset_key}/{filename}: {exc}"
                    )

                pbar.update(1)

    # ── Final summary ─────────────────────────────────────────────────────────
    issues = failed_images + timed_out
    if issues:
        print(
            f"\n  {failed_images} image(s) failed, {timed_out} timed out. "
            f"See {LOG_FILE} for details."
        )
    else:
        print(f"\n  All images processed successfully.")

    # ── Aggregate per-image → averaged ────────────────────────────────────────
    print("\nAggregating results ...")
    aggregate_to_averaged_csv(per_image_csv, averaged_csv)
    logger.info(f"Averaged CSV written: {averaged_csv}")
    print(f"  results_per_image.csv : {per_image_csv}")
    print(f"  results_average.csv  : {averaged_csv}")

    return averaged_csv


# ──────────────────────────────────────────────────────────────────────────────
# Terminal table  (reads from results_average.csv)
# ──────────────────────────────────────────────────────────────────────────────

def _best_marker(values, higher_is_better: bool):
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return [" "] * len(values)
    best = max(finite) if higher_is_better else min(finite)
    return ["*" if (not math.isnan(v) and v == best) else " " for v in values]


def print_detailed_tables(averaged_csv: Path):
    """Print one formatted table per subset to the terminal."""
    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    subsets     = list(dict.fromkeys(r["subset"]          for r in rows))
    methods     = list(dict.fromkeys(r["method"]          for r in rows))
    resolutions = sorted(set(int(r["resolution"])         for r in rows))

    cw  = {"res": 6, "method": 10, "psnr": 10, "ssim": 8, "mse": 12, "hf": 10}
    sep = "-"
    header = (
        f"{'Res':>{cw['res']}}  {'Method':<{cw['method']}}"
        f"  {'PSNR(dB)':>{cw['psnr']}}"
        f"  {'SSIM':>{cw['ssim']}}"
        f"  {'MSE':>{cw['mse']}}"
        f"  {'HF Retain':>{cw['hf']}}"
    )
    width = len(header)

    def _get(subset, method, res, key):
        for r in rows:
            if (r["subset"] == subset and r["method"] == method
                    and int(r["resolution"]) == res):
                v = r.get(key, "")
                return float(v) if v else float("nan")
        return float("nan")

    for subset in sorted(subsets):
        n = next((r["num_samples"] for r in rows if r["subset"] == subset), "?")
        print(f"\n{'=' * width}")
        print(f"  {subset}  (n={n})")
        print(f"{'=' * width}")
        print(header)
        print(sep * width)

        for res in resolutions:
            pv = [_get(subset, m, res, "psnr")     for m in methods]
            sv = [_get(subset, m, res, "ssim")     for m in methods]
            mv = [_get(subset, m, res, "mse")      for m in methods]
            hv = [_get(subset, m, res, "hf_ratio") for m in methods]

            pm = _best_marker(pv, True)
            sm = _best_marker(sv, True)
            mm = _best_marker(mv, False)
            hm = _best_marker(hv, True)

            for i, method in enumerate(methods):
                rc = str(res) if i == 0 else ""
                p  = f"{pv[i]:.4f}" if not math.isnan(pv[i]) else "  nan  "
                s  = f"{sv[i]:.4f}" if not math.isnan(sv[i]) else "  nan"
                m  = f"{mv[i]:.6f}" if not math.isnan(mv[i]) else "     nan   "
                h  = f"{hv[i]:.4f}" if not math.isnan(hv[i]) else "  nan  "
                print(
                    f"{rc:>{cw['res']}}  {method:<{cw['method']}}"
                    f"  {p:>{cw['psnr'] - 1}}{pm[i]}"
                    f"  {s:>{cw['ssim'] - 1}}{sm[i]}"
                    f"  {m:>{cw['mse']  - 1}}{mm[i]}"
                    f"  {h:>{cw['hf']   - 1}}{hm[i]}"
                )
            print(sep * width)

    print("\n* = best method for that metric at that resolution")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (reads from results_average.csv — fully decoupled)
# ──────────────────────────────────────────────────────────────────────────────

METHOD_MARKERS     = {"PointWise": "o", "Average": "s", "Fourier": "^"}
METHOD_LINESTYLES  = {"PointWise": "-", "Average": "--", "Fourier": "-."}
DATASET_COLORS     = ["#2E86AB", "#E07B39", "#6A994E", "#9B5DE5"]
DATASET_LINESTYLES = ["-", "--", "-.", ":"]   # kept for legacy use

# Shorthand aliases accepted by --methods CLI flag
METHOD_SHORTHAND = {
    "pw":        "PointWise",
    "pointwise": "PointWise",
    "avg":       "Average",
    "average":   "Average",
    "fft":       "Fourier",
    "fourier":   "Fourier",
}

METRIC_LABELS = {
    "psnr":     "PSNR (dB) ↑",
    "ssim":     "SSIM ↑",
    "mse":      "MSE ↓",
    "hf_ratio": "HF Energy Retention ↑",
}

DATASET_GROUPS = {
    "MetalSet_ViaSet":     ["MetalSet", "ViaSet"],
    "StdContact_StdMetal": ["StdContact", "StdMetal"],
}


def plot_metrics_from_csv(
    averaged_csv:      str,
    save_dir:          str  = None,
    filter_resolutions: list = None,
    filter_methods:     list = None,
    filter_datasets:    list = None,
):
    """
    Generate one figure per (datatype × dataset-group) from results_average.csv.

    Visual encoding
    ---------------
    Colour      → dataset   (primary comparison axis)
    Marker      → method    (circle / square / triangle)
    Line style  → method    (solid / dashed / dash-dot)

    Parameters
    ----------
    filter_resolutions : list of ints to include, or None for all
    filter_methods     : list of canonical method names to include, or None for all
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "dataset":     row["dataset"],
                "datatype":    row["datatype"],
                "resolution":  int(row["resolution"]),
                "method":      row["method"],
                "num_samples": row["num_samples"],
                **{
                    m: (float(row[m]) if row[m] else float("nan"))
                    for m in ("psnr", "ssim", "mse", "hf_ratio")
                },
            })

    datatypes   = list(dict.fromkeys(r["datatype"] for r in rows))
    methods     = [m for m in dict.fromkeys(r["method"] for r in rows)
                   if filter_methods is None or m in filter_methods]
    resolutions = sorted(r for r in set(r["resolution"] for r in rows)
                         if filter_resolutions is None
                         or r in filter_resolutions)
    metrics     = list(METRIC_LABELS.keys())
    num_samples = rows[0]["num_samples"] if rows else "?"

    # 4 datasets → 4 colours, consistent across all figures
    all_datasets  = list(dict.fromkeys(r["dataset"] for r in rows))
    if filter_datasets is not None:
        all_datasets = [d for d in all_datasets if d in filter_datasets]
    dataset_color = {ds: DATASET_COLORS[i % len(DATASET_COLORS)]
                     for i, ds in enumerate(all_datasets)}

    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        for group_name, group_datasets in DATASET_GROUPS.items():
            # Apply dataset filter within this group
            group_datasets = [d for d in group_datasets
                              if filter_datasets is None or d in filter_datasets]
            if not any(r["dataset"] in group_datasets for r in dt_rows):
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                                     constrained_layout=False)
            fig.subplots_adjust(top=0.91, bottom=0.18,
                                hspace=0.38, wspace=0.28)
            axes = axes.flatten()

            for ax, metric in zip(axes, metrics):
                for dataset in group_datasets:
                    color = dataset_color[dataset]
                    for method in methods:
                        ls     = METHOD_LINESTYLES.get(method, "-")
                        marker = METHOD_MARKERS.get(method, "o")

                        values = []
                        for res in resolutions:
                            matching = [
                                r[metric] for r in dt_rows
                                if r["dataset"]     == dataset
                                and r["method"]     == method
                                and r["resolution"] == res
                                and not math.isnan(r[metric])
                            ]
                            values.append(
                                float(np.mean(matching))
                                if matching else float("nan")
                            )

                        # Data line with markers
                        ax.plot(
                            resolutions, values,
                            color=color, linestyle=ls,
                            marker=marker, linewidth=2,
                            markersize=6, label="_nolegend_",
                        )

                ax.set_title(METRIC_LABELS[metric], fontsize=11,
                             fontweight="bold", pad=6)
                ax.set_xlabel("Resolution (px)", fontsize=9)
                ax.set_xticks(resolutions)
                ax.set_xticklabels([str(r) for r in resolutions], fontsize=8)
                ax.grid(True, alpha=0.25, linestyle="--")

            # Legend 1: Dataset (colour, solid line)
            dataset_handles = [
                plt.Line2D([0], [0], color=dataset_color[ds],
                           linewidth=3, label=ds)
                for ds in group_datasets
            ]
            legend_ds = fig.legend(
                handles=dataset_handles, title="Dataset",
                title_fontsize=10, fontsize=9,
                loc="upper left", bbox_to_anchor=(0.04, 0.13),
                ncol=len(group_datasets), framealpha=0.9,
                edgecolor="#aaaaaa",
            )

            # Legend 2: Method (marker + linestyle, neutral colour)
            method_handles = [
                plt.Line2D([0], [0], color="#444444",
                           linestyle=METHOD_LINESTYLES.get(m, "-"),
                           marker=METHOD_MARKERS.get(m, "o"),
                           linewidth=2, markersize=7, label=m)
                for m in methods
            ]
            fig.legend(
                handles=method_handles, title="Downsampling Method",
                title_fontsize=10, fontsize=9,
                loc="upper right", bbox_to_anchor=(0.96, 0.13),
                ncol=len(methods), framealpha=0.9,
                edgecolor="#aaaaaa",
            )
            fig.add_artist(legend_ds)

            pretty_group = " & ".join(group_datasets)
            fig.suptitle(
                f"Downsampling Study  |  Datatype: {datatype}  |  "
                f"Datasets: {pretty_group}  |  n = {num_samples} per subset",
                fontsize=12, fontweight="bold", y=0.97,
            )

            if save_dir is not None:
                out = save_dir / f"metrics_{datatype}_{group_name}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"  Saved: {out}")

            plt.show()
            plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Single-metric plot with error bars
# ──────────────────────────────────────────────────────────────────────────────

# Consistent colour per dataset — used by both plot functions
DATASET_COLOR_MAP = {
    "MetalSet":   DATASET_COLORS[0],
    "ViaSet":     DATASET_COLORS[1],
    "StdContact": DATASET_COLORS[2],
    "StdMetal":   DATASET_COLORS[3],
}

VALID_METRICS = ("psnr", "ssim", "mse", "hf_ratio")

METRIC_AXIS_LABELS = {
    "psnr":     "PSNR (dB) \u2191",
    "ssim":     "SSIM \u2191",
    "mse":      "MSE \u2193",
    "hf_ratio": "HF Energy Retention \u2191",
}


def plot_single_metric(
    averaged_csv:       str,
    metric:             str,
    save_dir:           str  = None,
    filter_resolutions: list = None,
    filter_methods:     list = None,
    filter_datasets:    list = None,
):
    """
    One figure per datatype, two panels side by side — one per dataset group
    (MetalSet & ViaSet | StdContact & StdMetal).  All datasets in a panel
    are plotted together.

    Visual encoding
    ---------------
    Colour     -> dataset   (4 distinct colours, consistent with plot_metrics_from_csv)
    Marker     -> method    (circle / square / triangle)
    Line style -> method    (solid / dashed / dash-dot)
    Error bars -> +/-1 std deviation across images in the subset

    Parameters
    ----------
    filter_resolutions : list of ints to include, or None for all
    filter_methods     : list of canonical method names, or None for all
    """
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}'. Choose from: {VALID_METRICS}"
        )

    std_col = f"{metric}_std"

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mean_val = row.get(metric, "")
            std_val  = row.get(std_col, "")
            rows.append({
                "subset":      row["subset"],
                "dataset":     row["dataset"],
                "datatype":    row["datatype"],
                "method":      row["method"],
                "resolution":  int(row["resolution"]),
                "num_samples": row.get("num_samples", "?"),
                "mean": float(mean_val) if mean_val else float("nan"),
                "std":  float(std_val)  if std_val  else float("nan"),
            })

    datatypes   = list(dict.fromkeys(r["datatype"] for r in rows))
    methods     = [m for m in dict.fromkeys(r["method"] for r in rows)
                   if filter_methods is None or m in filter_methods]
    resolutions = sorted(r for r in set(r["resolution"] for r in rows)
                         if filter_resolutions is None
                         or r in filter_resolutions)

    axis_label = METRIC_AXIS_LABELS.get(metric, metric)

    # All 4 datasets drawn on a single axes per datatype figure
    all_datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    if filter_datasets is not None:
        all_datasets = [d for d in all_datasets if d in filter_datasets]

    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=False)
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.75)

        for dataset in all_datasets:
            color   = DATASET_COLOR_MAP.get(dataset, "#888888")
            ds_rows = [r for r in dt_rows if r["dataset"] == dataset]
            if not ds_rows:
                continue

            for method in methods:
                ls     = METHOD_LINESTYLES.get(method, "-")
                marker = METHOD_MARKERS.get(method, "o")

                means, stds = [], []
                for res in resolutions:
                    match = [r for r in ds_rows
                             if r["method"] == method
                             and r["resolution"] == res]
                    if match and not math.isnan(match[0]["mean"]):
                        means.append(match[0]["mean"])
                        stds.append(
                            match[0]["std"]
                            if not math.isnan(match[0]["std"])
                            else 0.0
                        )
                    else:
                        means.append(float("nan"))
                        stds.append(float("nan"))

                plot_res   = [r for r, m in zip(resolutions, means)
                              if not math.isnan(m)]
                plot_means = [m for m in means if not math.isnan(m)]
                plot_stds  = [s for s, m in zip(stds, means)
                              if not math.isnan(m)]

                if not plot_res:
                    continue

                ax.errorbar(
                    plot_res, plot_means,
                    yerr=plot_stds,
                    fmt=f"{marker}{ls}",
                    color=color,
                    markersize=7,
                    linewidth=1.8,
                    capsize=4,
                    capthick=1.2,
                    elinewidth=1.0,
                    label="_nolegend_",
                )

        ax.set_xlabel("Resolution (px)", fontsize=9)
        ax.set_ylabel(axis_label, fontsize=9)
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(r) for r in resolutions], fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")

        # Legends placed outside the axes to the right — never overlap data
        colour_handles = [
            plt.Line2D([0], [0], color=DATASET_COLOR_MAP.get(ds, "#888"),
                       linewidth=3, label=ds)
            for ds in all_datasets
            if any(r["dataset"] == ds for r in dt_rows)
        ]
        method_handles = [
            plt.Line2D([0], [0], color="#444444",
                       linestyle=METHOD_LINESTYLES.get(m, "-"),
                       marker=METHOD_MARKERS.get(m, "o"),
                       linewidth=1.8, markersize=6, label=m)
            for m in methods
        ]
        legend_ds = fig.legend(
            handles=colour_handles,
            title="Dataset", title_fontsize=9, fontsize=8,
            loc="upper left", bbox_to_anchor=(0.77, 0.88),
            framealpha=0.9, edgecolor="#aaaaaa",
        )
        fig.legend(
            handles=method_handles,
            title="Method", title_fontsize=9, fontsize=8,
            loc="upper left", bbox_to_anchor=(0.77, 0.52),
            framealpha=0.9, edgecolor="#aaaaaa",
        )
        fig.add_artist(legend_ds)

        fig.suptitle(
            f"{axis_label}  |  Datatype: {datatype}",
            fontsize=12, fontweight="bold", y=0.97,
        )

        if save_dir is not None:
            out = save_dir / f"metric_{metric}_{datatype}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

        plt.show()
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Spatial resolution study for LithoBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--evaluate",   action="store_true",
                   help="Run the parallel image-level evaluation.")
    p.add_argument("--plot",       action="store_true",
                   help="Generate plots from results_average.csv.")
    p.add_argument("--tables",     action="store_true",
                   help="Print per-subset metric tables to terminal.")
    p.add_argument("--workers",    type=int, default=None,
                   help="Worker processes (default: cpu_count - 1).")
    p.add_argument("--timeout",    type=int, default=120,
                   help="Per-image timeout in seconds (default: 120).")
    p.add_argument("--force",      action="store_true",
                   help="Delete existing CSVs and start fresh.")
    p.add_argument("--save-plots", action="store_true",
                   help="Save plot PNGs to the output directory.")
    p.add_argument("--aggregate",  action="store_true",
                   help="Re-aggregate results_per_image.csv into results_average.csv "
                        "without re-running the evaluation.")
    p.add_argument("--plot-metric", type=str, default=None,
                   metavar="METRIC",
                   help=f"Plot a single metric with error bars. "
                        f"Choose from: {VALID_METRICS}.")
    p.add_argument("--resolutions", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Resolutions to include in plots (e.g. --resolutions 128 256 512). "
                        "Default: all available.")
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   metavar="M",
                   help="Methods to include in plots. Accepts: pw, avg, fft "
                        "(or full names PointWise, Average, Fourier). "
                        "Default: all.")
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                   metavar="D",
                   help="Datasets to include in plots. Accepts: metalset, viaset, "
                        "stdmetal, stdcontact (case-insensitive). Default: all.")
    p.add_argument("--csv",        type=str, default=None,
                   help="Override path to results_average.csv for "
                        "--plot / --tables / --plot-metric.")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not any([args.evaluate, args.plot, args.tables,
                args.aggregate, args.plot_metric]):
        print("No action specified. Use --evaluate, --plot, --plot-metric,\n"
              "--aggregate, or --tables.")
        print("Run with --help for full usage.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)
    logger = get_logger()
    averaged_csv = Path(args.csv) if args.csv else OUTPUT_DIR / "results_average.csv"

    logger.info(
        f"Session start | evaluate={args.evaluate} | plot={args.plot} | "
        f"tables={args.tables} | workers={args.workers} | "
        f"timeout={args.timeout} | force={args.force} | "
        f"averaged_csv={averaged_csv}"
    )

    # ── Resolve method shorthands ─────────────────────────────────────────────
    resolved_methods = None
    if args.methods:
        resolved_methods = []
        for m in args.methods:
            canonical = METHOD_SHORTHAND.get(m.lower(), m)
            if canonical not in METHOD_MARKERS:
                print(f"Unknown method '{m}'. "
                      f"Valid options: pw, avg, fft (or PointWise, Average, Fourier)")
                sys.exit(1)
            resolved_methods.append(canonical)

    filter_resolutions = args.resolutions   # None or list of ints
    filter_methods     = resolved_methods   # None or list of canonical names

    # ── Resolve dataset names (case-insensitive) ──────────────────────────────
    DATASET_SHORTHAND = {
        "metalset":   "MetalSet",
        "viaset":     "ViaSet",
        "stdmetal":   "StdMetal",
        "stdcontact": "StdContact",
    }
    filter_datasets = None
    if args.datasets:
        filter_datasets = []
        for d in args.datasets:
            canonical = DATASET_SHORTHAND.get(d.lower(), d)
            if canonical not in DATASET_SHORTHAND.values():
                print(f"Unknown dataset '{d}'. "
                      f"Valid options: metalset, viaset, stdmetal, stdcontact")
                sys.exit(1)
            filter_datasets.append(canonical)

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate:
        averaged_csv = run_evaluation(
            data_dict   = DATA_DICT,
            output_dir  = OUTPUT_DIR,
            num_workers = args.workers,
            force       = args.force,
            timeout     = args.timeout,
        )

    # ── Standalone aggregation ────────────────────────────────────────────────
    if args.aggregate:
        per_image_csv = OUTPUT_DIR / "results_per_image.csv"
        if not per_image_csv.exists():
            print(f"results_per_image.csv not found at {per_image_csv}. "
                  f"Run --evaluate first.")
        else:
            print("Aggregating results_per_image.csv ...")
            aggregate_to_averaged_csv(per_image_csv, averaged_csv)
            print(f"Done. results_average.csv written to: {averaged_csv}")
            logger.info(f"Standalone aggregation complete: {averaged_csv}")

    # ── Terminal tables ───────────────────────────────────────────────────────
    if args.tables:
        if not averaged_csv.exists():
            print(f"results_average.csv not found. Run --evaluate first.")
        else:
            print_detailed_tables(averaged_csv)

    # ── Plotting ──────────────────────────────────────────────────────────────
    if args.plot:
        if not averaged_csv.exists():
            print(f"results_average.csv not found. Run --evaluate first.")
        else:
            print("\nGenerating plots ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_metrics_from_csv(str(averaged_csv), save_dir=save_dir,
                                  filter_resolutions=filter_resolutions,
                                  filter_methods=filter_methods,
                                  filter_datasets=filter_datasets)

    # ── Single-metric plot with error bars ───────────────────────────────────
    if args.plot_metric:
        metric = args.plot_metric.lower().strip()
        if metric not in VALID_METRICS:
            print(f"Unknown metric '{metric}'. "
                  f"Choose from: {VALID_METRICS}")
        elif not averaged_csv.exists():
            print(f"results_average.csv not found. "
                  f"Run --evaluate or --aggregate first.")
        else:
            print(f"\nGenerating single-metric plot for: {metric} ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_single_metric(str(averaged_csv),
                               metric=metric,
                               save_dir=save_dir,
                               filter_resolutions=filter_resolutions,
                               filter_methods=filter_methods,
                               filter_datasets=filter_datasets)

    logger.info("Session end")
