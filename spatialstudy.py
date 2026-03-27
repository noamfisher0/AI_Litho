"""
spatialstudy.py
===============
Spatial resolution study for the LithoBench dataset.

Architecture
------------
Evaluation uses a flat image-level work queue fed to a
ProcessPoolExecutor. Each worker processes one image at a time and
returns one result row. The main process writes each result to
results_per_image.csv immediately, giving image-level checkpointing.
At the end the main process aggregates the per-image CSV into
results_average.csv, which the plotting functions consume.

Two output CSVs
---------------
  results_per_image.csv   -- one row per (image x method x resolution)
                             written continuously; used for checkpointing
                             and provides full metric distributions
  results_average.csv     -- one row per (subset x method x resolution)
                             computed from results_per_image.csv at the
                             end of a completed or resumed run

Resume behaviour
----------------
On startup the main process reads results_per_image.csv and builds the
set of already-completed (subset, filename) pairs. Only the remaining
images are added to the work queue. Stop the run at any time with
Ctrl+C -- the next run resumes from exactly where it left off.
Use --force to discard existing results and start fresh.

Flags
-----
  --evaluate            run the image-level parallel evaluation
  --plot                generate figures from results_average.csv
  --plot-metric METRIC  plot a single metric with error bars
  --tables              print per-subset metric tables to the terminal
  --aggregate           re-aggregate per-image CSV without re-evaluating
  --workers N           number of worker processes (default: cpu_count-1)
  --samples N           cap images per subset; omit for all images
  --methods M [M ...]   evaluation methods: pw/avg/fft (default: all three)
  --resolutions N [N ..]resolutions to plot (default: all)
  --datasets D [D ...]  datasets to plot (default: all)
  --force               delete existing CSVs and start fresh
  --save-plots          save PNG figures to the output directory
  --timeout N           per-image timeout in seconds (default: 120)

Usage examples
--------------
  python spatialstudy.py --evaluate --workers 8
  python spatialstudy.py --evaluate --workers 8 --methods avg pw
  python spatialstudy.py --evaluate --workers 8 --samples 500
  python spatialstudy.py --evaluate --workers 8 --methods avg pw --samples None
  python spatialstudy.py --plot --save-plots
  python spatialstudy.py --plot-metric psnr --methods avg pw
  python spatialstudy.py --evaluate --plot --force
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
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
from skimage.transform import downscale_local_mean, resize


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "lithobench-main"
OUTPUT_DIR = PROJECT_ROOT / "resolution_study_output_final"
LOG_FILE = OUTPUT_DIR / "spatial_study.log"

# Default sample cap — overridden at runtime by --samples flag.
# None means process every image in every subset.
NUM_SAMPLES = None

TARGET_RESOLUTIONS = [1024, 512, 256, 128]

DATA_DICT = {
    "MetalSet-Printed": str(DATA_ROOT / "MetalSet" / "printed"),
    "MetalSet-Resist": str(DATA_ROOT / "MetalSet" / "resist"),
    "MetalSet-Target": str(DATA_ROOT / "MetalSet" / "target"),
    "MetalSet-LevelILT": str(DATA_ROOT / "MetalSet" / "levelsetILT"),
    "MetalSet-Litho": str(DATA_ROOT / "MetalSet" / "litho"),
    "MetalSet-PixelILT": str(DATA_ROOT / "MetalSet" / "pixelILT"),
    "ViaSet-Printed": str(DATA_ROOT / "ViaSet" / "printed"),
    "ViaSet-Resist": str(DATA_ROOT / "ViaSet" / "resist"),
    "ViaSet-Target": str(DATA_ROOT / "ViaSet" / "target"),
    "ViaSet-LevelILT": str(DATA_ROOT / "ViaSet" / "levelsetILT"),
    "ViaSet-Litho": str(DATA_ROOT / "ViaSet" / "litho"),
    "ViaSet-PixelILT": str(DATA_ROOT / "ViaSet" / "pixelILT"),
    "StdContact-Printed": str(DATA_ROOT / "StdContact" / "printed"),
    "StdContact-Resist": str(DATA_ROOT / "StdContact" / "resist"),
    "StdContact-Target": str(DATA_ROOT / "StdContact" / "target"),
    "StdContact-Litho": str(DATA_ROOT / "StdContact" / "litho"),
    "StdContact-PixelILT": str(DATA_ROOT / "StdContact" / "pixelILT"),
    "StdMetal-Printed": str(DATA_ROOT / "StdMetal" / "printed"),
    "StdMetal-Resist": str(DATA_ROOT / "StdMetal" / "resist"),
    "StdMetal-Target": str(DATA_ROOT / "StdMetal" / "target"),
    "StdMetal-Litho": str(DATA_ROOT / "StdMetal" / "litho"),
    "StdMetal-PixelILT": str(DATA_ROOT / "StdMetal" / "pixelILT"),
}

# -- CSV schemas ---------------------------------------------------------------

PER_IMAGE_FIELDS = [
    "subset",
    "dataset",
    "datatype",
    "filename",
    "resolution",
    "method",
    "psnr",
    "ssim",
    "mse",
    "hf_ratio",
]

AVERAGED_FIELDS = [
    "subset",
    "dataset",
    "datatype",
    "num_samples",
    "resolution",
    "method",
    "psnr",
    "psnr_std",
    "ssim",
    "ssim_std",
    "mse",
    "mse_std",
    "hf_ratio",
    "hf_ratio_std",
]


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------


def setup_logging(log_file: Path) -> logging.Logger:
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
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        mode="a",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("spatialstudy")


def _worker_initializer(log_file: str):
    setup_logging(Path(log_file))


# ------------------------------------------------------------------------------
# Image loading
# ------------------------------------------------------------------------------


def load_image(path: Path):
    """Load with cv2 (no GUI), return float32 [0,1] in RGB order. None on failure."""
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


def to_grayscale(image):
    if image.ndim == 3:
        return image[..., :3].mean(axis=-1)
    return image


# ------------------------------------------------------------------------------
# Downsampling methods
# ------------------------------------------------------------------------------


def downsample_pointwise(image, target_size: int):
    """Nearest-neighbour: selects every N-th pixel. Fast, zero blurring,
    susceptible to aliasing at large scale factors."""
    h, w = image.shape[:2]
    sh, sw = h // target_size, w // target_size
    return image[::sh, ::sw][:target_size, :target_size]


def downsample_average(image, target_size: int):
    """Block-mean: replaces each NxN block with its mean. Acts as a box
    low-pass filter suppressing aliasing at the cost of slight blurring."""
    h, w = image.shape[:2]
    fh, fw = h // target_size, w // target_size
    factors = (fh, fw, 1) if image.ndim == 3 else (fh, fw)
    return downscale_local_mean(image, factors).astype(image.dtype)


def downsample_fourier(image, target_size: int):
    """Frequency-domain: crops 2-D FFT to central target x target coefficients.
    Theoretically optimal anti-aliasing; most computationally expensive."""

    def _channel(ch):
        F = np.fft.fftshift(np.fft.fft2(ch.astype(np.float64)))
        cy, cx = F.shape[0] // 2, F.shape[1] // 2
        half = target_size // 2
        Fc = F[cy - half : cy + half, cx - half : cx + half]
        scale = (target_size**2) / (F.shape[0] * F.shape[1])
        return (np.fft.ifft2(np.fft.ifftshift(Fc)).real * scale).astype(np.float32)

    result = (
        np.stack([_channel(image[..., c]) for c in range(image.shape[2])], axis=-1)
        if image.ndim == 3
        else _channel(image)
    )
    lo, hi = image.min(), image.max()
    rlo, rhi = result.min(), result.max()
    if rhi - rlo > 1e-8:
        result = (result - rlo) / (rhi - rlo) * (hi - lo) + lo
    return result.astype(np.float32)


# All available methods — workers receive a filtered subset via eval_methods
ALL_METHODS = {
    "PointWise": downsample_pointwise,
    "Average": downsample_average,
    "Fourier": downsample_fourier,
}


# ------------------------------------------------------------------------------
# Quality metrics
# ------------------------------------------------------------------------------


def upsample_to_original(image, original_size: int):
    return resize(
        image,
        (original_size, original_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype("float32")


def compute_mse(orig, recon) -> float:
    return float(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2))


def compute_psnr_from_mse(mse: float, data_range: float) -> float:
    """Returns nan for identical/flat images (shown as gaps on plots)."""
    if mse == 0 or data_range == 0:
        return float("nan")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def compute_psnr(orig, recon) -> float:
    mse = compute_mse(orig, recon)
    data_range = float(orig.max() - orig.min())
    return compute_psnr_from_mse(mse, data_range)


def compute_ssim(orig, recon) -> float:
    o, r = orig.astype(np.float64), recon.astype(np.float64)
    L = float(o.max() - o.min()) or 1.0
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_o, mu_r = o.mean(), r.mean()
    sig_o = np.sqrt(np.mean((o - mu_o) ** 2))
    sig_r = np.sqrt(np.mean((r - mu_r) ** 2))
    sig_or = np.mean((o - mu_o) * (r - mu_r))
    num = (2 * mu_o * mu_r + C1) * (2 * sig_or + C2)
    den = (mu_o**2 + mu_r**2 + C1) * (sig_o**2 + sig_r**2 + C2)
    return float(num / den)


def compute_hf_ratio(orig, recon, fft_size: int = 512) -> float:
    """Fraction of high-frequency energy retained after downsampling+upsampling.
    Images are resized to fft_size x fft_size before the FFT to reduce cost —
    512x512 is 16x cheaper than 2048x2048 with negligible accuracy difference."""

    def hf_energy(img):
        gray = to_grayscale(img)
        # Downsample to fft_size for speed before FFT
        if gray.shape[0] != fft_size:
            from skimage.transform import resize as sk_resize

            gray = sk_resize(
                gray,
                (fft_size, fft_size),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float64)
        else:
            gray = gray.astype(np.float64)
        F = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        h, w = F.shape
        mask = np.ones((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = False
        return float(np.sum(F[mask] ** 2))

    orig_hf = hf_energy(orig)
    return float(hf_energy(recon) / orig_hf) if orig_hf > 0 else 1.0


# ------------------------------------------------------------------------------
# Worker function  (one call per batch, runs in a subprocess)
# ------------------------------------------------------------------------------

# Number of images processed per worker task — reduces IPC and CSV overhead
BATCH_SIZE = 64
ROW_BUFFER_LIMIT = 2000  # flush buffer to disk every N rows


def _process_batch(task: tuple):
    """
    Process a batch of images, returning all rows in one IPC round-trip.

    Parameters
    ----------
    task : (batch_items, resolutions, eval_methods, eval_metrics)
        batch_items : list of (subset, dataset, datatype, file_path_str)

    Returns
    -------
    all_rows : list of CSV row dicts for all images in the batch
    """
    batch_items, resolutions, eval_methods, eval_metrics = task

    logger = get_logger()
    active_methods = {m: ALL_METHODS[m] for m in eval_methods if m in ALL_METHODS}

    eval_metric_set = set(eval_metrics)
    do_psnr = "psnr" in eval_metric_set
    do_ssim = "ssim" in eval_metric_set
    do_mse = "mse" in eval_metric_set
    do_hf = "hf_ratio" in eval_metric_set
    need_mse_for_anything = do_psnr or do_mse

    all_rows = []
    append_row = all_rows.append

    for subset, dataset, datatype, file_path_str in batch_items:
        file_path = Path(file_path_str)

        img = load_image(file_path)
        if img is None:
            logger.warning(f"Failed to load {file_path} -- skipping")
            continue

        native_h = img.shape[0]
        orig_gray = to_grayscale(img)
        data_range = float(orig_gray.max() - orig_gray.min())

        for method_name, method_fn in active_methods.items():
            for res in resolutions:
                if res >= native_h:
                    continue
                try:
                    ds = method_fn(img, res)
                    up = upsample_to_original(ds, native_h)
                    up_gray = to_grayscale(up)

                    mse_val = compute_mse(orig_gray, up_gray) if need_mse_for_anything else float("nan")
                    psnr = compute_psnr_from_mse(mse_val, data_range) if do_psnr else float("nan")
                    ssim = compute_ssim(orig_gray, up_gray) if do_ssim else float("nan")
                    mse = mse_val if do_mse else float("nan")
                    hf_ratio = compute_hf_ratio(img, up) if do_hf else float("nan")

                    append_row(
                        {
                            "subset": subset,
                            "dataset": dataset,
                            "datatype": datatype,
                            "filename": file_path.name,
                            "resolution": res,
                            "method": method_name,
                            "psnr": "" if math.isnan(psnr) else round(psnr, 6),
                            "ssim": "" if math.isnan(ssim) else round(ssim, 6),
                            "mse": "" if math.isnan(mse) else round(mse, 6),
                            "hf_ratio": "" if math.isnan(hf_ratio) else round(hf_ratio, 6),
                        }
                    )

                except Exception as exc:
                    logger.warning(
                        f"ERROR {subset} | {method_name} | res={res} | "
                        f"{file_path.name}: {exc}"
                    )

    logger.info(f"Batch done: {len(batch_items)} images, {len(all_rows)} rows")
    return all_rows


# ------------------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------------------


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
    """Append per-image result rows. Called from main process only."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def aggregate_to_averaged_csv(per_image_csv: Path, averaged_csv: Path):
    """
    Read results_per_image.csv and write results_average.csv.
    Computes mean and std per metric independently per
    (subset x method x resolution), ignoring nan/inf/empty values.
    """
    METRICS = ("psnr", "ssim", "mse", "hf_ratio")

    values = {}  # (subset, method, res) -> {metric: [float, ...]}
    meta = {}  # subset -> (dataset, datatype)

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
                return round(float(np.std(v, ddof=1 if len(v) > 1 else 0)), 6) if v else ""

            w.writerow(
                {
                    "subset": subset,
                    "dataset": dataset,
                    "datatype": datatype,
                    "num_samples": n,
                    "resolution": res,
                    "method": method,
                    "psnr": avg("psnr"),
                    "psnr_std": std("psnr"),
                    "ssim": avg("ssim"),
                    "ssim_std": std("ssim"),
                    "mse": avg("mse"),
                    "mse_std": std("mse"),
                    "hf_ratio": avg("hf_ratio"),
                    "hf_ratio_std": std("hf_ratio"),
                }
            )

    return averaged_csv


# ------------------------------------------------------------------------------
# Evaluation orchestrator
# ------------------------------------------------------------------------------

ALL_METRICS = ("psnr", "ssim", "mse", "hf_ratio")

METRIC_SHORTHAND = {
    "psnr": "psnr",
    "ssim": "ssim",
    "mse": "mse",
    "hf": "hf_ratio",
    "hf_ratio": "hf_ratio",
}


def run_evaluation(
    data_dict: dict,
    output_dir: Path,
    num_workers: int = None,
    num_samples: int = None,
    eval_methods: list = None,
    eval_metrics: list = None,
    force: bool = False,
    timeout: int = 120,
):
    """
    Build a flat image-level work queue, dispatch to a ProcessPoolExecutor,
    stream results to results_per_image.csv, then aggregate.

    Parameters
    ----------
    eval_methods : list of canonical method names to evaluate.
                   Defaults to all three (PointWise, Average, Fourier).
    eval_metrics : list of metric keys to compute per image.
                   Defaults to all four (psnr, ssim, mse, hf_ratio).
    num_samples  : cap per subset, or None for all images.
    """
    import tqdm

    if eval_methods is None:
        eval_methods = list(ALL_METHODS.keys())
    if eval_metrics is None:
        eval_metrics = list(ALL_METRICS)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "results_per_image.csv"
    averaged_csv = output_dir / "results_average.csv"
    logger = get_logger()

    if force:
        for p in (per_image_csv, averaged_csv):
            if p.exists():
                p.unlink()
        logger.info("--force: cleared existing CSVs")
        print("--force: cleared existing results, starting fresh.\n")

    completed = load_completed_images(per_image_csv)
    logger.info(
        f"Already completed: {len(completed)} images | "
        f"eval_methods={eval_methods} | eval_metrics={eval_metrics} | "
        f"num_samples={num_samples}"
    )

    print("Scanning dataset directories ...")
    image_items = []

    for subset_key, image_dir in data_dict.items():
        parts = subset_key.split("-", 1)
        dataset = parts[0]
        datatype = parts[1] if len(parts) > 1 else ""
        path = Path(image_dir)

        if not path.exists():
            logger.warning(f"Directory not found, skipping: {path}")
            print(f"  WARNING: {path} not found -- skipping {subset_key}")
            continue

        files = sorted(f for f in path.iterdir() if f.is_file())

        if num_samples is not None:
            # On resume: keep already-completed files, top up to num_samples
            already_done_names = {fname for (sub, fname) in completed if sub == subset_key}
            already_done_files = [f for f in files if f.name in already_done_names]
            remaining_files = [f for f in files if f.name not in already_done_names]
            still_needed = max(0, num_samples - len(already_done_files))
            new_sample = random.sample(remaining_files, min(still_needed, len(remaining_files)))
            files = already_done_files + new_sample

        for f in files:
            if (subset_key, f.name) in completed:
                continue
            image_items.append((subset_key, dataset, datatype, str(f)))

    # Chunk image_items into batches for reduced IPC overhead
    tasks = [
        (image_items[i : i + BATCH_SIZE], TARGET_RESOLUTIONS, eval_methods, eval_metrics)
        for i in range(0, len(image_items), BATCH_SIZE)
    ]
    pending_images = len(image_items)
    pending = len(tasks)

    if completed:
        print(
            f"\nResuming -- {len(completed)} images already done, "
            f"{pending_images} remaining ({pending} batches of {BATCH_SIZE}).\n"
        )
    else:
        print(
            f"\nStarting fresh -- {pending_images} images to process "
            f"({pending} batches of {BATCH_SIZE}).\n"
        )

    print(f"Methods:  {eval_methods}")
    print(f"Metrics:  {eval_metrics}")
    print(f"Samples per subset: {'all' if num_samples is None else num_samples}\n")

    if not tasks:
        print("Nothing to do -- all images already processed.")
        logger.info("All images complete, aggregating.")
        aggregate_to_averaged_csv(per_image_csv, averaged_csv)
        print(f"Averaged CSV written to: {averaged_csv}")
        return averaged_csv

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, pending)

    logger.info(
        f"Starting evaluation | images={pending_images} | batches={pending} | "
        f"batch_size={BATCH_SIZE} | workers={num_workers} | timeout={timeout}s"
    )
    print(f"Workers: {num_workers}  |  Batches: {pending}  |  Images: {pending_images}\n")

    failed_images = 0
    timed_out = 0

    def _flush_buffer(buf, path):
        """Write buffered rows to disk and clear the buffer."""
        if buf:
            append_image_rows(path, buf)
            buf.clear()

    with cf.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_initializer,
        initargs=(str(LOG_FILE),),
    ) as executor:
        # Submit in chunks so the executor queue doesn't hold all futures
        # in memory at once — important for very large datasets
        future_to_task = {executor.submit(_process_batch, task): task for task in tasks}

        row_buffer = []  # accumulate rows here before flushing to disk

        with tqdm.tqdm(
            total=pending_images,
            desc="Processing images",
            unit="img",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for future in cf.as_completed(future_to_task, timeout=None):
                batch_task = future_to_task[future]
                n_in_batch = len(batch_task[0])

                try:
                    all_rows = future.result(timeout=timeout * BATCH_SIZE)
                    if all_rows:
                        row_buffer.extend(all_rows)
                        # Flush to disk once buffer is large enough,
                        # keeping individual write operations efficient
                        if len(row_buffer) >= ROW_BUFFER_LIMIT:
                            _flush_buffer(row_buffer, per_image_csv)
                    else:
                        failed_images += n_in_batch
                        logger.warning("Batch returned no results")
                except cf.TimeoutError:
                    timed_out += n_in_batch
                    logger.error(f"TIMEOUT batch of {n_in_batch} images -- skipped")
                    pbar.set_postfix_str("batch timeout", refresh=True)
                    # Flush whatever we have so far on timeout — preserves progress
                    _flush_buffer(row_buffer, per_image_csv)
                except Exception as exc:
                    failed_images += n_in_batch
                    logger.error(f"ERROR in batch: {exc}")

                pbar.update(n_in_batch)

        # Final flush — write any remaining buffered rows
        _flush_buffer(row_buffer, per_image_csv)
        logger.info("Final buffer flush complete")

    issues = failed_images + timed_out
    if issues:
        print(f"\n  {failed_images} failed, {timed_out} timed out. See {LOG_FILE} for details.")
    else:
        print("\n  All images processed successfully.")

    print("\nAggregating results ...")
    aggregate_to_averaged_csv(per_image_csv, averaged_csv)
    logger.info(f"Averaged CSV written: {averaged_csv}")
    print(f"  results_per_image.csv : {per_image_csv}")
    print(f"  results_average.csv   : {averaged_csv}")

    return averaged_csv


# ------------------------------------------------------------------------------
# Terminal table
# ------------------------------------------------------------------------------


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

    subsets = list(dict.fromkeys(r["subset"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    resolutions = sorted(set(int(r["resolution"]) for r in rows))

    cw = {"res": 6, "method": 10, "psnr": 10, "ssim": 8, "mse": 12, "hf": 10}
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
            if r["subset"] == subset and r["method"] == method and int(r["resolution"]) == res:
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
            pv = [_get(subset, m, res, "psnr") for m in methods]
            sv = [_get(subset, m, res, "ssim") for m in methods]
            mv = [_get(subset, m, res, "mse") for m in methods]
            hv = [_get(subset, m, res, "hf_ratio") for m in methods]

            pm = _best_marker(pv, True)
            sm = _best_marker(sv, True)
            mm = _best_marker(mv, False)
            hm = _best_marker(hv, True)

            for i, method in enumerate(methods):
                rc = str(res) if i == 0 else ""
                p = f"{pv[i]:.4f}" if not math.isnan(pv[i]) else "  nan  "
                s = f"{sv[i]:.4f}" if not math.isnan(sv[i]) else "  nan"
                m = f"{mv[i]:.6f}" if not math.isnan(mv[i]) else "     nan   "
                h = f"{hv[i]:.4f}" if not math.isnan(hv[i]) else "  nan  "
                print(
                    f"{rc:>{cw['res']}}  {method:<{cw['method']}}"
                    f"  {p:>{cw['psnr'] - 1}}{pm[i]}"
                    f"  {s:>{cw['ssim'] - 1}}{sm[i]}"
                    f"  {m:>{cw['mse']  - 1}}{mm[i]}"
                    f"  {h:>{cw['hf']   - 1}}{hm[i]}"
                )
            print(sep * width)

    print("\n* = best method for that metric at that resolution")


# ------------------------------------------------------------------------------
# Plotting constants  (shared by both plot functions)
# ------------------------------------------------------------------------------

METHOD_MARKERS = {"PointWise": "o", "Average": "s", "Fourier": "^"}
METHOD_LINESTYLES = {"PointWise": "-", "Average": "--", "Fourier": "-."}
DATASET_COLORS = ["#2E86AB", "#E07B39", "#6A994E", "#9B5DE5"]

# Shorthand aliases for --methods and --datasets CLI flags
METHOD_SHORTHAND = {
    "pw": "PointWise",
    "pointwise": "PointWise",
    "avg": "Average",
    "average": "Average",
    "fft": "Fourier",
    "fourier": "Fourier",
}

DATASET_SHORTHAND = {
    "metalset": "MetalSet",
    "viaset": "ViaSet",
    "stdmetal": "StdMetal",
    "stdcontact": "StdContact",
}

METRIC_LABELS = {
    "psnr": "PSNR (dB) \u2191",
    "ssim": "SSIM \u2191",
    "mse": "MSE \u2193",
    "hf_ratio": "HF Energy Retention \u2191",
}

DATASET_GROUPS = {
    "MetalSet_ViaSet": ["MetalSet", "ViaSet"],
    "StdContact_StdMetal": ["StdContact", "StdMetal"],
}

DATASET_COLOR_MAP = {
    "MetalSet": DATASET_COLORS[0],
    "ViaSet": DATASET_COLORS[1],
    "StdContact": DATASET_COLORS[2],
    "StdMetal": DATASET_COLORS[3],
}

VALID_METRICS = ("psnr", "ssim", "mse", "hf_ratio")


# ------------------------------------------------------------------------------
# plot_metrics_from_csv  -- 4-metric overview, one figure per datatype x group
# ------------------------------------------------------------------------------


def plot_metrics_from_csv(
    averaged_csv: str,
    save_dir: str = None,
    filter_resolutions: list = None,
    filter_methods: list = None,
    filter_datasets: list = None,
):
    """
    Visual encoding: Colour -> dataset | Marker+linestyle -> method
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "datatype": row["datatype"],
                    "resolution": int(row["resolution"]),
                    "method": row["method"],
                    "num_samples": row["num_samples"],
                    **{m: (float(row[m]) if row[m] else float("nan")) for m in ("psnr", "ssim", "mse", "hf_ratio")},
                }
            )

    datatypes = list(dict.fromkeys(r["datatype"] for r in rows))
    methods = [m for m in dict.fromkeys(r["method"] for r in rows) if filter_methods is None or m in filter_methods]
    resolutions = sorted(r for r in set(r["resolution"] for r in rows) if filter_resolutions is None or r in filter_resolutions)
    metrics = list(METRIC_LABELS.keys())
    num_samples = rows[0]["num_samples"] if rows else "?"

    all_datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    if filter_datasets is not None:
        all_datasets = [d for d in all_datasets if d in filter_datasets]
    dataset_color = {ds: DATASET_COLORS[i % len(DATASET_COLORS)] for i, ds in enumerate(all_datasets)}

    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        for group_name, group_datasets in DATASET_GROUPS.items():
            group_datasets = [d for d in group_datasets if filter_datasets is None or d in filter_datasets]
            if not any(r["dataset"] in group_datasets for r in dt_rows):
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=False)
            fig.subplots_adjust(top=0.91, bottom=0.18, hspace=0.38, wspace=0.28)
            axes = axes.flatten()

            for ax, metric in zip(axes, metrics):
                for dataset in group_datasets:
                    color = dataset_color[dataset]
                    for method in methods:
                        ls = METHOD_LINESTYLES.get(method, "-")
                        marker = METHOD_MARKERS.get(method, "o")
                        values = []
                        for res in resolutions:
                            matching = [
                                r[metric]
                                for r in dt_rows
                                if r["dataset"] == dataset
                                and r["method"] == method
                                and r["resolution"] == res
                                and not math.isnan(r[metric])
                            ]
                            values.append(float(np.mean(matching)) if matching else float("nan"))
                        ax.plot(
                            resolutions,
                            values,
                            color=color,
                            linestyle=ls,
                            marker=marker,
                            linewidth=2,
                            markersize=6,
                            label="_nolegend_",
                        )

                ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold", pad=6)
                ax.set_xlabel("Resolution (px)", fontsize=9)
                ax.set_xticks(resolutions)
                ax.set_xticklabels([str(r) for r in resolutions], fontsize=8)
                ax.grid(True, alpha=0.25, linestyle="--")

            dataset_handles = [plt.Line2D([0], [0], color=dataset_color[ds], linewidth=3, label=ds) for ds in group_datasets]
            legend_ds = fig.legend(
                handles=dataset_handles,
                title="Dataset",
                title_fontsize=10,
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(0.04, 0.13),
                ncol=len(group_datasets),
                framealpha=0.9,
                edgecolor="#aaaaaa",
            )
            method_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color="#444444",
                    linestyle=METHOD_LINESTYLES.get(m, "-"),
                    marker=METHOD_MARKERS.get(m, "o"),
                    linewidth=2,
                    markersize=7,
                    label=m,
                )
                for m in methods
            ]
            fig.legend(
                handles=method_handles,
                title="Downsampling Method",
                title_fontsize=10,
                fontsize=9,
                loc="upper right",
                bbox_to_anchor=(0.96, 0.13),
                ncol=len(methods),
                framealpha=0.9,
                edgecolor="#aaaaaa",
            )
            fig.add_artist(legend_ds)

            pretty_group = " & ".join(group_datasets)
            fig.suptitle(
                f"Downsampling Study  |  Datatype: {datatype}  |  "
                f"Datasets: {pretty_group}  |  n = {num_samples} per subset",
                fontsize=12,
                fontweight="bold",
                y=0.97,
            )

            if save_dir is not None:
                out = save_dir / f"metrics_{datatype}_{group_name}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"  Saved: {out}")

            plt.show()
            plt.close(fig)


# ------------------------------------------------------------------------------
# plot_single_metric  -- one metric, all datasets, error bars
# ------------------------------------------------------------------------------


def plot_single_metric(
    averaged_csv: str,
    metric: str,
    save_dir: str = None,
    filter_resolutions: list = None,
    filter_methods: list = None,
    filter_datasets: list = None,
):
    """
    One figure per datatype. All datasets on a single axes.
    Visual encoding: Colour -> dataset | Marker+linestyle -> method
    Error bars: +/-1 std deviation.
    Legends placed outside the axes to the right.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {VALID_METRICS}")

    std_col = f"{metric}_std"

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mean_val = row.get(metric, "")
            std_val = row.get(std_col, "")
            rows.append(
                {
                    "subset": row["subset"],
                    "dataset": row["dataset"],
                    "datatype": row["datatype"],
                    "method": row["method"],
                    "resolution": int(row["resolution"]),
                    "num_samples": row.get("num_samples", "?"),
                    "mean": float(mean_val) if mean_val else float("nan"),
                    "std": float(std_val) if std_val else float("nan"),
                }
            )

    datatypes = list(dict.fromkeys(r["datatype"] for r in rows))
    methods = [m for m in dict.fromkeys(r["method"] for r in rows) if filter_methods is None or m in filter_methods]
    resolutions = sorted(r for r in set(r["resolution"] for r in rows) if filter_resolutions is None or r in filter_resolutions)

    axis_label = METRIC_LABELS.get(metric, metric)

    all_datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    if filter_datasets is not None:
        all_datasets = [d for d in all_datasets if d in filter_datasets]

    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=False)
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.75)

        for dataset in all_datasets:
            color = DATASET_COLOR_MAP.get(dataset, "#888888")
            ds_rows = [r for r in dt_rows if r["dataset"] == dataset]
            if not ds_rows:
                continue

            for method in methods:
                ls = METHOD_LINESTYLES.get(method, "-")
                marker = METHOD_MARKERS.get(method, "o")

                means, stds = [], []
                for res in resolutions:
                    match = [r for r in ds_rows if r["method"] == method and r["resolution"] == res]
                    if match and not math.isnan(match[0]["mean"]):
                        means.append(match[0]["mean"])
                        stds.append(match[0]["std"] if not math.isnan(match[0]["std"]) else 0.0)
                    else:
                        means.append(float("nan"))
                        stds.append(float("nan"))

                plot_res = [r for r, m in zip(resolutions, means) if not math.isnan(m)]
                plot_means = [m for m in means if not math.isnan(m)]
                plot_stds = [s for s, m in zip(stds, means) if not math.isnan(m)]

                if not plot_res:
                    continue

                ax.errorbar(
                    plot_res,
                    plot_means,
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

        colour_handles = [
            plt.Line2D([0], [0], color=DATASET_COLOR_MAP.get(ds, "#888"), linewidth=3, label=ds)
            for ds in all_datasets
            if any(r["dataset"] == ds for r in dt_rows)
        ]
        method_handles = [
            plt.Line2D(
                [0],
                [0],
                color="#444444",
                linestyle=METHOD_LINESTYLES.get(m, "-"),
                marker=METHOD_MARKERS.get(m, "o"),
                linewidth=1.8,
                markersize=6,
                label=m,
            )
            for m in methods
        ]
        legend_ds = fig.legend(
            handles=colour_handles,
            title="Dataset",
            title_fontsize=9,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0.77, 0.88),
            framealpha=0.9,
            edgecolor="#aaaaaa",
        )
        fig.legend(
            handles=method_handles,
            title="Method",
            title_fontsize=9,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0.77, 0.52),
            framealpha=0.9,
            edgecolor="#aaaaaa",
        )
        fig.add_artist(legend_ds)

        fig.suptitle(f"{axis_label}  |  Datatype: {datatype}", fontsize=12, fontweight="bold", y=0.97)

        if save_dir is not None:
            out = save_dir / f"metric_{metric}_{datatype}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

        plt.show()
        plt.close(fig)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Spatial resolution study for LithoBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--evaluate", action="store_true", help="Run the parallel image-level evaluation.")
    p.add_argument("--plot", action="store_true", help="Generate 4-metric overview plots.")
    p.add_argument(
        "--plot-metric",
        type=str,
        default=None,
        metavar="METRIC",
        help=f"Plot a single metric with error bars. Choose from: {VALID_METRICS}.",
    )
    p.add_argument("--tables", action="store_true", help="Print per-subset metric tables to terminal.")
    p.add_argument("--aggregate", action="store_true", help="Re-aggregate results_per_image.csv -> results_average.csv.")
    p.add_argument("--workers", type=int, default=None, help="Worker processes (default: cpu_count - 1).")
    p.add_argument("--timeout", type=int, default=120, help="Per-image timeout in seconds (default: 120).")
    p.add_argument("--samples", type=int, default=None, help="Cap images per subset. Omit or pass None for all images.")
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        metavar="M",
        help="Methods for --evaluate and/or plotting. Accepts: pw, avg, fft (or PointWise, Average, Fourier). Default: all three.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        metavar="METRIC",
        help="Metrics to compute during --evaluate. Accepts: psnr, ssim, mse, hf (or hf_ratio). Default: all four. Skipping hf greatly speeds up evaluation.",
    )
    p.add_argument("--resolutions", type=int, nargs="+", default=None, metavar="N", help="Resolutions to include in plots. Default: all.")
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        metavar="D",
        help="Datasets to include in plots. Accepts: metalset, viaset, stdmetal, stdcontact. Default: all.",
    )
    p.add_argument("--force", action="store_true", help="Delete existing CSVs and start fresh.")
    p.add_argument("--save-plots", action="store_true", help="Save plot PNGs to the output directory.")
    p.add_argument("--csv", type=str, default=None, help="Override path to results_average.csv.")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not any([args.evaluate, args.plot, args.tables, args.aggregate, args.plot_metric]):
        print("No action specified. Use --evaluate, --plot, --plot-metric, --aggregate, or --tables.")
        print("Run with --help for full usage.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)
    logger = get_logger()
    averaged_csv = Path(args.csv) if args.csv else OUTPUT_DIR / "results_average.csv"

    # ── Resolve method shorthands (shared by evaluate + plot) ─────────────────
    resolved_methods = None
    if args.methods:
        resolved_methods = []
        for m in args.methods:
            canonical = METHOD_SHORTHAND.get(m.lower(), m)
            if canonical not in ALL_METHODS:
                print("Unknown method '{}'. Valid options: pw, avg, fft (or PointWise, Average, Fourier)".format(m))
                sys.exit(1)
            resolved_methods.append(canonical)

    # ── Resolve dataset shorthands (plot only) ────────────────────────────────
    filter_datasets = None
    if args.datasets:
        filter_datasets = []
        for d in args.datasets:
            canonical = DATASET_SHORTHAND.get(d.lower(), d)
            if canonical not in DATASET_SHORTHAND.values():
                print("Unknown dataset '{}'. Valid options: metalset, viaset, stdmetal, stdcontact".format(d))
                sys.exit(1)
            filter_datasets.append(canonical)

    filter_resolutions = args.resolutions
    filter_methods = resolved_methods  # used for both evaluate and plot

    # ── Resolve metric shorthands (evaluate only) ─────────────────────────────
    resolved_metrics = None
    if args.metrics:
        resolved_metrics = []
        for m in args.metrics:
            canonical = METRIC_SHORTHAND.get(m.lower(), m.lower())
            if canonical not in ALL_METRICS:
                print("Unknown metric '{}'. Valid options: psnr, ssim, mse, hf (or hf_ratio)".format(m))
                sys.exit(1)
            if canonical not in resolved_metrics:
                resolved_metrics.append(canonical)

    logger.info(
        f"Session start | evaluate={args.evaluate} | plot={args.plot} | "
        f"tables={args.tables} | workers={args.workers} | "
        f"timeout={args.timeout} | force={args.force} | "
        f"samples={args.samples} | methods={filter_methods} | "
        f"averaged_csv={averaged_csv}"
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate:
        averaged_csv = run_evaluation(
            data_dict=DATA_DICT,
            output_dir=OUTPUT_DIR,
            num_workers=args.workers,
            num_samples=args.samples,
            eval_methods=filter_methods,  # None = all three
            eval_metrics=resolved_metrics,  # None = all four
            force=args.force,
            timeout=args.timeout,
        )

    # ── Standalone aggregation ────────────────────────────────────────────────
    if args.aggregate:
        per_image_csv = OUTPUT_DIR / "results_per_image.csv"
        if not per_image_csv.exists():
            print("results_per_image.csv not found. Run --evaluate first.")
        else:
            print("Aggregating results_per_image.csv ...")
            aggregate_to_averaged_csv(per_image_csv, averaged_csv)
            print(f"Done. results_average.csv written to: {averaged_csv}")
            logger.info(f"Standalone aggregation complete: {averaged_csv}")

    # ── Terminal tables ───────────────────────────────────────────────────────
    if args.tables:
        if not averaged_csv.exists():
            print("results_average.csv not found. Run --evaluate first.")
        else:
            print_detailed_tables(averaged_csv)

    # ── Plotting ──────────────────────────────────────────────────────────────
    if args.plot:
        if not averaged_csv.exists():
            print("results_average.csv not found. Run --evaluate first.")
        else:
            print("\nGenerating plots ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_metrics_from_csv(
                str(averaged_csv),
                save_dir=save_dir,
                filter_resolutions=filter_resolutions,
                filter_methods=filter_methods,
                filter_datasets=filter_datasets,
            )

    # ── Single-metric plot ────────────────────────────────────────────────────
    if args.plot_metric:
        metric = args.plot_metric.lower().strip()
        if metric not in VALID_METRICS:
            print(f"Unknown metric '{metric}'. Choose from: {VALID_METRICS}")
        elif not averaged_csv.exists():
            print("results_average.csv not found. Run --evaluate or --aggregate first.")
        else:
            print(f"\nGenerating single-metric plot for: {metric} ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_single_metric(
                str(averaged_csv),
                metric=metric,
                save_dir=save_dir,
                filter_resolutions=filter_resolutions,
                filter_methods=filter_methods,
                filter_datasets=filter_datasets,
            )

    logger.info("Session end")
