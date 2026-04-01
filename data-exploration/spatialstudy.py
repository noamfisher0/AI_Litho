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
  --evaluate              run the image-level parallel evaluation
  --plot                  generate figures from results_average.csv
  --plot-metric METRIC    plot a single metric with error bars
  --plot-bar              grouped bar chart per datatype for a given metric,
                          resolution and dataset (requires --bar-metric,
                          --bar-resolution, --bar-dataset)
  --bar-metric METRIC     metric for --plot-bar (psnr/ssim/mse/hf)
  --bar-resolution N      resolution for --plot-bar (e.g. 512)
  --bar-dataset D         dataset for --plot-bar (e.g. metalset)
  --tables                print per-subset metric tables to the terminal
  --aggregate             re-aggregate per-image CSV without re-evaluating
  --workers N             number of worker processes (default: cpu_count-1)
  --samples N             cap images per subset; omit for all images
  --methods M [M ...]     evaluation methods: pw/avg/fft (default: all three)
  --resolutions N [N ..]  resolutions to plot (default: all)
  --datasets D [D ...]    datasets to plot (default: all)
  --force                 delete existing CSVs and start fresh
  --save-plots            save PNG figures to the output directory
  --timeout N             per-image timeout in seconds (default: 120)

Visual comparison flags
-----------------------
  --plot-visual           generate visual comparison figures
  --visual-dataset DS     dataset to use (e.g. metalset) [required]
  --visual-datatype DT    single datatype (e.g. PixelILT); omit to generate
                          one figure per datatype, all using the same tile
  --visual-resolution N   resolution for method-comparison mode (e.g. 512);
                          omit to compare resolutions instead of methods
  --visual-method M       method for resolution-comparison mode (default: avg)
  --visual-seed N         random seed for reproducible tile selection
  --visual-reconstruction show NN-upsampled reconstruction (default: off,
                          shows the downsampled image at native resolution)

  Auto-selected modes:
    method comparison    : --visual-datatype + --visual-resolution both set
                           panels = Original | Method1 | Method2 | ...
    resolution comparison: --visual-datatype set, --visual-resolution omitted
                           panels = Original | 128px | 256px | 512px | 1024px
    all-datatype sweep   : --visual-datatype omitted (+ --visual-resolution set
                           or omitted) — one figure per datatype, same tile

Usage examples
--------------
  # All datatypes, method comparison at 512px (one figure per datatype)
  python spatialstudy.py --plot-visual --visual-dataset metalset --visual-resolution 512

  # Single datatype, compare resolutions using Average pooling
  python spatialstudy.py --plot-visual --visual-dataset metalset --visual-datatype PixelILT

  # Single datatype, compare resolutions using FFT
  python spatialstudy.py --plot-visual --visual-dataset metalset --visual-datatype PixelILT --visual-method fft

  # Single datatype, compare specific methods at 256px
  python spatialstudy.py --plot-visual --visual-dataset metalset --visual-datatype Litho --visual-resolution 256 --methods avg pw

  # Save all figures
  python spatialstudy.py --plot-visual --visual-dataset metalset --visual-resolution 512 --save-plots

  # Resolution comparison with reconstruction view
  python spatialstudy.py --plot-visual --visual-dataset viaset --visual-datatype Target --visual-reconstruction
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

NUM_SAMPLES = None

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

# Canonical datatype display order
DATATYPE_ORDER = ["LevelILT", "Litho", "PixelILT", "Printed", "Resist", "Target"]

# -- CSV schemas ---------------------------------------------------------------

PER_IMAGE_FIELDS = [
    "subset", "dataset", "datatype", "filename",
    "resolution", "method", "psnr", "ssim", "mse", "hf_ratio",
]

AVERAGED_FIELDS = [
    "subset", "dataset", "datatype", "num_samples",
    "resolution", "method",
    "psnr", "psnr_std", "ssim", "ssim_std",
    "mse", "mse_std", "hf_ratio", "hf_ratio_std",
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
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3,
        encoding="utf-8", mode="a",
    )

    def _rotate_name(default_name: str) -> str:
        path = Path(default_name)
        if ".log." in path.name:
            base, idx = path.name.rsplit(".log.", 1)
            if idx.isdigit():
                return str(path.with_name(f"{base}_log_{idx}.log"))
        return default_name

    fh.namer = _rotate_name
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
    h, w = image.shape[:2]
    sh, sw = h // target_size, w // target_size
    return image[::sh, ::sw][:target_size, :target_size]


def downsample_average(image, target_size: int):
    h, w = image.shape[:2]
    fh, fw = h // target_size, w // target_size
    factors = (fh, fw, 1) if image.ndim == 3 else (fh, fw)
    return downscale_local_mean(image, factors).astype(image.dtype)


def downsample_fourier(image, target_size: int):
    def _channel(ch):
        F = np.fft.fftshift(np.fft.fft2(ch.astype(np.float64)))
        cy, cx = F.shape[0] // 2, F.shape[1] // 2
        half = target_size // 2
        Fc = F[cy - half: cy + half, cx - half: cx + half]
        scale = (target_size ** 2) / (F.shape[0] * F.shape[1])
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


ALL_METHODS = {
    "PointWise": downsample_pointwise,
    "Average":   downsample_average,
    "Fourier":   downsample_fourier,
}


# ------------------------------------------------------------------------------
# Quality metrics
# ------------------------------------------------------------------------------

def upsample_to_original(image, original_size: int):
    return resize(
        image, (original_size, original_size),
        order=0, preserve_range=True, anti_aliasing=False,
    ).astype("float32")


def compute_mse(orig, recon) -> float:
    return float(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2))


def compute_psnr_from_mse(mse: float, data_range: float) -> float:
    if mse == 0 or data_range == 0:
        return float("nan")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def compute_psnr(orig, recon) -> float:
    mse = compute_mse(orig, recon)
    return compute_psnr_from_mse(mse, float(orig.max() - orig.min()))


def compute_ssim(orig, recon) -> float:
    o, r = orig.astype(np.float64), recon.astype(np.float64)
    L = float(o.max() - o.min()) or 1.0
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_o, mu_r = o.mean(), r.mean()
    sig_o = np.sqrt(np.mean((o - mu_o) ** 2))
    sig_r = np.sqrt(np.mean((r - mu_r) ** 2))
    sig_or = np.mean((o - mu_o) * (r - mu_r))
    num = (2 * mu_o * mu_r + C1) * (2 * sig_or + C2)
    den = (mu_o ** 2 + mu_r ** 2 + C1) * (sig_o ** 2 + sig_r ** 2 + C2)
    return float(num / den)


def compute_hf_ratio(orig, recon, fft_size: int = 512) -> float:
    def hf_energy(img):
        gray = to_grayscale(img)
        if gray.shape[0] != fft_size:
            from skimage.transform import resize as sk_resize
            gray = sk_resize(
                gray, (fft_size, fft_size),
                order=1, preserve_range=True, anti_aliasing=True,
            ).astype(np.float64)
        else:
            gray = gray.astype(np.float64)
        F = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        h, w = F.shape
        mask = np.ones((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = False
        return float(np.sum(F[mask] ** 2))

    orig_hf = hf_energy(orig)
    return float(hf_energy(recon) / orig_hf) if orig_hf > 0 else 1.0


# ------------------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------------------

BATCH_SIZE = 64
ROW_BUFFER_LIMIT = 2000


def _process_batch(task: tuple):
    batch_items, resolutions, eval_methods, eval_metrics = task
    logger = get_logger()
    active_methods = {m: ALL_METHODS[m] for m in eval_methods if m in ALL_METHODS}
    eval_metric_set = set(eval_metrics)
    do_psnr  = "psnr"     in eval_metric_set
    do_ssim  = "ssim"     in eval_metric_set
    do_mse   = "mse"      in eval_metric_set
    do_hf    = "hf_ratio" in eval_metric_set
    need_mse = do_psnr or do_mse

    all_rows = []
    for subset, dataset, datatype, file_path_str in batch_items:
        file_path = Path(file_path_str)
        img = load_image(file_path)
        if img is None:
            logger.warning(f"Failed to load {file_path} -- skipping")
            continue
        native_h   = img.shape[0]
        orig_gray  = to_grayscale(img)
        data_range = float(orig_gray.max() - orig_gray.min())

        for method_name, method_fn in active_methods.items():
            for res in resolutions:
                if res >= native_h:
                    continue
                try:
                    ds      = method_fn(img, res)
                    up      = upsample_to_original(ds, native_h)
                    up_gray = to_grayscale(up)
                    mse_val = compute_mse(orig_gray, up_gray) if need_mse else float("nan")
                    psnr    = compute_psnr_from_mse(mse_val, data_range) if do_psnr else float("nan")
                    ssim    = compute_ssim(orig_gray, up_gray)            if do_ssim else float("nan")
                    mse     = mse_val                                     if do_mse  else float("nan")
                    hf_r    = compute_hf_ratio(img, up)                   if do_hf   else float("nan")
                    all_rows.append({
                        "subset": subset, "dataset": dataset, "datatype": datatype,
                        "filename": file_path.name, "resolution": res, "method": method_name,
                        "psnr":     "" if math.isnan(psnr)  else round(psnr,  6),
                        "ssim":     "" if math.isnan(ssim)  else round(ssim,  6),
                        "mse":      "" if math.isnan(mse)   else round(mse,   6),
                        "hf_ratio": "" if math.isnan(hf_r)  else round(hf_r,  6),
                    })
                except Exception as exc:
                    logger.warning(
                        f"ERROR {subset} | {method_name} | res={res} | {file_path.name}: {exc}"
                    )

    logger.info(f"Batch done: {len(batch_items)} images, {len(all_rows)} rows")
    return all_rows


# ------------------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------------------

def load_completed_images(csv_path: Path) -> set:
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
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def aggregate_to_averaged_csv(per_image_csv: Path, averaged_csv: Path):
    METRICS = ("psnr", "ssim", "mse", "hf_ratio")
    values, meta = {}, {}
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
                if not (math.isnan(v) or math.isinf(v)):
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

            w.writerow({
                "subset": subset, "dataset": dataset, "datatype": datatype,
                "num_samples": n, "resolution": res, "method": method,
                "psnr": avg("psnr"), "psnr_std": std("psnr"),
                "ssim": avg("ssim"), "ssim_std": std("ssim"),
                "mse":  avg("mse"),  "mse_std":  std("mse"),
                "hf_ratio": avg("hf_ratio"), "hf_ratio_std": std("hf_ratio"),
            })
    return averaged_csv


# ------------------------------------------------------------------------------
# Evaluation orchestrator
# ------------------------------------------------------------------------------

ALL_METRICS = ("psnr", "ssim", "mse", "hf_ratio")

METRIC_SHORTHAND = {
    "psnr": "psnr", "ssim": "ssim", "mse": "mse",
    "hf": "hf_ratio", "hf_ratio": "hf_ratio",
}


def run_evaluation(
    data_dict:    dict,
    output_dir:   Path,
    num_workers:  int  = None,
    num_samples:  int  = None,
    eval_methods: list = None,
    eval_metrics: list = None,
    force:        bool = False,
    timeout:      int  = 120,
):
    import tqdm
    if eval_methods is None:
        eval_methods = list(ALL_METHODS.keys())
    if eval_metrics is None:
        eval_metrics = list(ALL_METRICS)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "results_per_image.csv"
    averaged_csv  = output_dir / "results_average.csv"
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
        f"eval_methods={eval_methods} | num_samples={num_samples}"
    )

    print("Scanning dataset directories ...")
    image_items = []
    for subset_key, image_dir in data_dict.items():
        parts    = subset_key.split("-", 1)
        dataset  = parts[0]
        datatype = parts[1] if len(parts) > 1 else ""
        path     = Path(image_dir)
        if not path.exists():
            logger.warning(f"Directory not found, skipping: {path}")
            print(f"  WARNING: {path} not found -- skipping {subset_key}")
            continue
        files = sorted(f for f in path.iterdir() if f.is_file())
        if num_samples is not None:
            already_done_names = {fname for (sub, fname) in completed if sub == subset_key}
            already_done_files = [f for f in files if f.name in already_done_names]
            remaining_files    = [f for f in files if f.name not in already_done_names]
            still_needed       = max(0, num_samples - len(already_done_files))
            new_sample         = random.sample(remaining_files, min(still_needed, len(remaining_files)))
            files              = already_done_files + new_sample
        for f in files:
            if (subset_key, f.name) in completed:
                continue
            image_items.append((subset_key, dataset, datatype, str(f)))

    tasks = [
        (image_items[i: i + BATCH_SIZE], TARGET_RESOLUTIONS, eval_methods, eval_metrics)
        for i in range(0, len(image_items), BATCH_SIZE)
    ]
    pending_images = len(image_items)
    pending        = len(tasks)

    if completed:
        print(f"\nResuming -- {len(completed)} images already done, "
              f"{pending_images} remaining ({pending} batches of {BATCH_SIZE}).\n")
    else:
        print(f"\nStarting fresh -- {pending_images} images to process "
              f"({pending} batches of {BATCH_SIZE}).\n")

    print(f"Methods:  {eval_methods}")
    print(f"Metrics:  {eval_metrics}")
    print(f"Samples per subset: {'all' if num_samples is None else num_samples}\n")

    if not tasks:
        print("Nothing to do -- all images already processed.")
        aggregate_to_averaged_csv(per_image_csv, averaged_csv)
        print(f"Averaged CSV written to: {averaged_csv}")
        return averaged_csv

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, pending)
    logger.info(
        f"Starting evaluation | images={pending_images} | batches={pending} | "
        f"workers={num_workers} | timeout={timeout}s"
    )
    print(f"Workers: {num_workers}  |  Batches: {pending}  |  Images: {pending_images}\n")

    failed_images, timed_out = 0, 0

    def _flush_buffer(buf, path):
        if buf:
            append_image_rows(path, buf)
            buf.clear()

    with cf.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_initializer,
        initargs=(str(LOG_FILE),),
    ) as executor:
        future_to_task = {executor.submit(_process_batch, task): task for task in tasks}
        row_buffer = []

        with tqdm.tqdm(total=pending_images, desc="Processing images",
                       unit="img", dynamic_ncols=True, file=sys.stdout) as pbar:
            for future in cf.as_completed(future_to_task, timeout=None):
                batch_task = future_to_task[future]
                n_in_batch = len(batch_task[0])
                try:
                    all_rows = future.result(timeout=timeout * BATCH_SIZE)
                    if all_rows:
                        row_buffer.extend(all_rows)
                        if len(row_buffer) >= ROW_BUFFER_LIMIT:
                            _flush_buffer(row_buffer, per_image_csv)
                    else:
                        failed_images += n_in_batch
                except cf.TimeoutError:
                    timed_out += n_in_batch
                    logger.error(f"TIMEOUT batch of {n_in_batch} images -- skipped")
                    pbar.set_postfix_str("batch timeout", refresh=True)
                    _flush_buffer(row_buffer, per_image_csv)
                except Exception as exc:
                    failed_images += n_in_batch
                    logger.error(f"ERROR in batch: {exc}")
                pbar.update(n_in_batch)

        _flush_buffer(row_buffer, per_image_csv)

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
    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    subsets     = list(dict.fromkeys(r["subset"] for r in rows))
    methods     = list(dict.fromkeys(r["method"] for r in rows))
    resolutions = sorted(set(int(r["resolution"]) for r in rows))

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


# ------------------------------------------------------------------------------
# Plotting constants
# ------------------------------------------------------------------------------

METHOD_MARKERS    = {"PointWise": "o", "Average": "s", "Fourier": "^"}
METHOD_LINESTYLES = {"PointWise": "-", "Average": "--", "Fourier": "-."}
METHOD_COLORS     = {"Average": "#2E86AB", "PointWise": "#E07B39", "Fourier": "#9B5DE5"}

DATASET_COLORS = ["#2E86AB", "#E07B39", "#6A994E", "#9B5DE5"]

METHOD_SHORTHAND = {
    "pw": "PointWise", "pointwise": "PointWise",
    "avg": "Average",  "average":   "Average",
    "fft": "Fourier",  "fourier":   "Fourier",
}

DATASET_SHORTHAND = {
    "metalset":   "MetalSet",
    "viaset":     "ViaSet",
    "stdmetal":   "StdMetal",
    "stdcontact": "StdContact",
}

METRIC_LABELS = {
    "psnr":     "PSNR (dB)",
    "ssim":     "SSIM",
    "mse":      "MSE",
    "hf_ratio": "HF Energy Retention",
}

DATASET_GROUPS = {
    "MetalSet_ViaSet":     ["MetalSet",   "ViaSet"],
    "StdContact_StdMetal": ["StdContact", "StdMetal"],
}

DATASET_COLOR_MAP = {
    "MetalSet":   DATASET_COLORS[0],
    "ViaSet":     DATASET_COLORS[1],
    "StdContact": DATASET_COLORS[2],
    "StdMetal":   DATASET_COLORS[3],
}

VALID_METRICS = ("psnr", "ssim", "mse", "hf_ratio")


# ------------------------------------------------------------------------------
# plot_metrics_from_csv  -- 4-metric overview
# ------------------------------------------------------------------------------

def plot_metrics_from_csv(
    averaged_csv:       str,
    save_dir:           str  = None,
    filter_resolutions: list = None,
    filter_methods:     list = None,
    filter_datasets:    list = None,
):
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
                **{m: (float(row[m]) if row[m] else float("nan"))
                   for m in ("psnr", "ssim", "mse", "hf_ratio")},
            })

    datatypes   = list(dict.fromkeys(r["datatype"] for r in rows))
    methods     = [m for m in dict.fromkeys(r["method"] for r in rows)
                   if filter_methods is None or m in filter_methods]
    resolutions = sorted(r for r in set(r["resolution"] for r in rows)
                         if filter_resolutions is None or r in filter_resolutions)
    metrics     = list(METRIC_LABELS.keys())
    num_samples = rows[0]["num_samples"] if rows else "?"

    all_datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    if filter_datasets is not None:
        all_datasets = [d for d in all_datasets if d in filter_datasets]
    dataset_color = {ds: DATASET_COLORS[i % len(DATASET_COLORS)]
                     for i, ds in enumerate(all_datasets)}

    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        for group_name, group_datasets in DATASET_GROUPS.items():
            group_datasets = [d for d in group_datasets
                              if filter_datasets is None or d in filter_datasets]
            if not any(r["dataset"] in group_datasets for r in dt_rows):
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=False)
            fig.subplots_adjust(top=0.91, bottom=0.18, hspace=0.42, wspace=0.30)
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
                                if r["dataset"] == dataset
                                and r["method"] == method
                                and r["resolution"] == res
                                and not math.isnan(r[metric])
                            ]
                            values.append(float(np.mean(matching)) if matching else float("nan"))
                        ax.plot(resolutions, values, color=color, linestyle=ls,
                                marker=marker, linewidth=2, markersize=7, label="_nolegend_")

                ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold", pad=6)
                ax.set_xlabel("Resolution", fontsize=12)
                ax.set_xticks(resolutions)
                ax.set_xticklabels([str(r) for r in resolutions], fontsize=11)
                ax.tick_params(axis="y", labelsize=11)
                ax.grid(True, alpha=0.25, linestyle="--")

            dataset_handles = [
                plt.Line2D([0], [0], color=dataset_color[ds], linewidth=3, label=ds)
                for ds in group_datasets
            ]
            legend_ds = fig.legend(
                handles=dataset_handles, title="Dataset",
                title_fontsize=11, fontsize=10,
                loc="upper left", bbox_to_anchor=(0.04, 0.13),
                ncol=len(group_datasets), framealpha=0.9, edgecolor="#aaaaaa",
            )
            method_handles = [
                plt.Line2D([0], [0], color="#444444",
                           linestyle=METHOD_LINESTYLES.get(m, "-"),
                           marker=METHOD_MARKERS.get(m, "o"),
                           linewidth=2, markersize=8, label=m)
                for m in methods
            ]
            fig.legend(
                handles=method_handles, title="Downsampling Method",
                title_fontsize=11, fontsize=10,
                loc="upper right", bbox_to_anchor=(0.96, 0.13),
                ncol=len(methods), framealpha=0.9, edgecolor="#aaaaaa",
            )
            fig.add_artist(legend_ds)

            pretty_group = " & ".join(group_datasets)
            fig.suptitle(
                f"Downsampling Study  |  Datatype: {datatype}  |  "
                f"Datasets: {pretty_group}  |  n = {num_samples} per subset",
                fontsize=13, fontweight="bold", y=0.97,
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
    averaged_csv:       str,
    metric:             str,
    save_dir:           str  = None,
    filter_resolutions: list = None,
    filter_methods:     list = None,
    filter_datasets:    list = None,
):
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
                         if filter_resolutions is None or r in filter_resolutions)
    axis_label  = METRIC_LABELS.get(metric, metric)

    all_datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    if filter_datasets is not None:
        all_datasets = [d for d in all_datasets if d in filter_datasets]

    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=False)
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.75)

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
                    match = [r for r in ds_rows if r["method"] == method and r["resolution"] == res]
                    if match and not math.isnan(match[0]["mean"]):
                        means.append(match[0]["mean"])
                        stds.append(match[0]["std"] if not math.isnan(match[0]["std"]) else 0.0)
                    else:
                        means.append(float("nan"))
                        stds.append(float("nan"))

                plot_res   = [r for r, m in zip(resolutions, means) if not math.isnan(m)]
                plot_means = [m for m in means if not math.isnan(m)]
                plot_stds  = [s for s, m in zip(stds, means) if not math.isnan(m)]
                if not plot_res:
                    continue
                ax.errorbar(plot_res, plot_means, yerr=plot_stds,
                            fmt=f"{marker}{ls}", color=color,
                            markersize=8, linewidth=2, capsize=5,
                            capthick=1.4, elinewidth=1.2, label="_nolegend_")

        ax.set_xlabel("Resolution", fontsize=13)
        ax.set_ylabel(axis_label, fontsize=13)
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(r) for r in resolutions], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.25, linestyle="--")

        colour_handles = [
            plt.Line2D([0], [0], color=DATASET_COLOR_MAP.get(ds, "#888"), linewidth=3, label=ds)
            for ds in all_datasets if any(r["dataset"] == ds for r in dt_rows)
        ]
        method_handles = [
            plt.Line2D([0], [0], color="#444444",
                       linestyle=METHOD_LINESTYLES.get(m, "-"),
                       marker=METHOD_MARKERS.get(m, "o"),
                       linewidth=2, markersize=7, label=m)
            for m in methods
        ]
        legend_ds = fig.legend(
            handles=colour_handles, title="Dataset",
            title_fontsize=11, fontsize=10,
            loc="upper left", bbox_to_anchor=(0.77, 0.88),
            framealpha=0.9, edgecolor="#aaaaaa",
        )
        fig.legend(
            handles=method_handles, title="Method",
            title_fontsize=11, fontsize=10,
            loc="upper left", bbox_to_anchor=(0.77, 0.52),
            framealpha=0.9, edgecolor="#aaaaaa",
        )
        fig.add_artist(legend_ds)
        fig.suptitle(f"{axis_label}  |  Datatype: {datatype}",
                     fontsize=14, fontweight="bold", y=0.97)

        if save_dir is not None:
            out = save_dir / f"metric_{metric}_{datatype}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

        plt.show()
        plt.close(fig)


# ------------------------------------------------------------------------------
# plot_metric_bar  -- grouped bar chart: datatypes on x-axis, bars per method
# ------------------------------------------------------------------------------

def plot_metric_bar(
    averaged_csv:   str,
    metric:         str,
    resolution:     int,
    dataset:        str,
    save_dir:       str  = None,
    filter_methods: list = None,
):
    if metric not in VALID_METRICS:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {VALID_METRICS}")

    std_col = f"{metric}_std"
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset:
                continue
            if int(row["resolution"]) != resolution:
                continue
            mean_val = row.get(metric, "")
            std_val  = row.get(std_col, "")
            rows.append({
                "datatype": row["datatype"],
                "method":   row["method"],
                "mean":     float(mean_val) if mean_val else float("nan"),
                "std":      float(std_val)  if std_val  else 0.0,
                "n":        row.get("num_samples", "?"),
            })

    if not rows:
        print(f"No data found for dataset='{dataset}', resolution={resolution}px.")
        return

    all_methods = list(dict.fromkeys(r["method"] for r in rows))
    methods     = [m for m in all_methods if filter_methods is None or m in filter_methods]
    if not methods:
        print("No methods remaining after filtering.")
        return

    present_dts = set(r["datatype"] for r in rows)
    datatypes   = [dt for dt in DATATYPE_ORDER if dt in present_dts]
    datatypes  += sorted(dt for dt in present_dts if dt not in DATATYPE_ORDER)

    n_dt  = len(datatypes)
    n_met = len(methods)
    width = 0.75 / n_met
    x     = np.arange(n_dt)

    fig, ax = plt.subplots(figsize=(max(9, n_dt * 1.4), 6))

    for i, method in enumerate(methods):
        offsets = x - 0.375 + (i + 0.5) * width
        means, errs = [], []
        for dt in datatypes:
            match = [r for r in rows if r["datatype"] == dt and r["method"] == method]
            if match and not math.isnan(match[0]["mean"]):
                means.append(match[0]["mean"])
                errs.append(match[0]["std"] if not math.isnan(match[0]["std"]) else 0.0)
            else:
                means.append(float("nan"))
                errs.append(0.0)

        color = METHOD_COLORS.get(method, "#888888")
        ax.bar(
            offsets, means, width=width * 0.88,
            label=method, color=color, alpha=0.88, edgecolor="white",
            yerr=errs, capsize=4,
            error_kw={"elinewidth": 1.4, "capthick": 1.4, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datatypes, rotation=30, ha="right", fontsize=13)
    ax.set_xlabel("Datatype", fontsize=14)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)}  |  Dataset: {dataset}  |  Resolution: {resolution}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="Method", title_fontsize=12, fontsize=11, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()

    if save_dir is not None:
        methods_tag = "_".join(methods)
        out = save_dir / f"bar_{metric}_{dataset}_{resolution}px_{methods_tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.show()
    plt.close(fig)


# ------------------------------------------------------------------------------
# Visual comparison — shared helpers
# ------------------------------------------------------------------------------

def _list_files(path: Path) -> list:
    """Return sorted list of files in a directory."""
    return sorted(f for f in path.iterdir() if f.is_file())


def _find_common_tile(dataset: str, seed: int = None) -> str:
    """
    Return a single tile filename that exists in every datatype directory for
    this dataset, ensuring all per-datatype figures show the same physical tile.
    Falls back to any file from the first available directory if no overlap.
    """
    keys  = [k for k in DATA_DICT if k.startswith(f"{dataset}-")]
    common = None
    for key in keys:
        p = Path(DATA_DICT[key])
        if not p.exists():
            continue
        names  = {f.name for f in p.iterdir() if f.is_file()}
        common = names if common is None else common & names

    rng = random.Random(seed)
    if common:
        return rng.choice(sorted(common))

    # Fallback: pick from the first available directory
    for key in keys:
        p = Path(DATA_DICT[key])
        if p.exists():
            files = _list_files(p)
            if files:
                return rng.choice(files).name
    raise RuntimeError(f"No image files found for dataset '{dataset}'.")


def _render_comparison_figure(
    panels:    list,
    suptitle:  str,
    save_path: Path = None,
):
    """
    Render panels = [(title, grayscale_array, mse_or_None, psnr_or_None, ssim_or_None)]
    into one figure.  save_path=None → show only.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 10), 5.5))
    if n == 1:
        axes = [axes]

    for ax, (label, img, mse, psnr, ssim) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")
        if mse is not None:
            psnr_str    = f"{psnr:.2f} dB" if not math.isnan(psnr) else "∞"
            metric_text = f"MSE  = {mse:.5f}\nPSNR = {psnr_str}\nSSIM = {ssim:.4f}"
            ax.text(
                0.5, -0.04, metric_text,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontfamily="monospace", color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                          edgecolor="#cccccc", linewidth=0.8),
            )

    fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=0.999)
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.02, right=0.98, wspace=0.06)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close(fig)


def _make_ds_panel(orig_gray, img, method_fn, res, native_h,
                   show_reconstruction, panel_label):
    """Downsample → optionally reconstruct → compute metrics → return panel tuple."""
    ds      = method_fn(img, res)
    up      = upsample_to_original(ds, native_h)
    up_gray = to_grayscale(up)
    dr      = float(orig_gray.max() - orig_gray.min()) or 1.0
    mse     = compute_mse(orig_gray, up_gray)
    psnr    = compute_psnr_from_mse(mse, dr)
    ssim    = compute_ssim(orig_gray, up_gray)
    display = up_gray if show_reconstruction else to_grayscale(ds)
    return panel_label, display, mse, psnr, ssim


# ------------------------------------------------------------------------------
# plot_visual_comparison  — three modes
# ------------------------------------------------------------------------------

def plot_visual_comparison(
    dataset:             str,
    datatype:            str  = None,
    resolution:          int  = None,
    methods:             list = None,
    show_reconstruction: bool = False,
    save_dir:            str  = None,
    seed:                int  = None,
    all_resolutions:     list = None,
):
    """
    Generate visual comparison figures with three operating modes.

    Mode A — method comparison
        Triggered when: resolution is set.
        One figure per datatype (or just the specified datatype).
        Panels: Original | Method1 | Method2 | ...
        All datatypes use the SAME tile filename for direct comparison.

    Mode B — resolution comparison
        Triggered when: resolution is None.
        One figure per datatype (or just the specified datatype).
        Panels: Original | 128px | 256px | 512px | 1024px
        Uses the first entry in `methods` as the single downsampling method.

    In both modes, when datatype=None every datatype is iterated and all
    figures share the same tile, so they are directly comparable across
    the different layout representations (Target, Litho, PixelILT, etc.).

    Parameters
    ----------
    dataset             : canonical dataset name (e.g. "MetalSet")
    datatype            : specific datatype, or None to iterate all
    resolution          : resolution for Mode A; None selects Mode B
    methods             : method list (Mode A: all used; Mode B: first used)
    show_reconstruction : show NN-upsampled image rather than downsampled image
    save_dir            : directory for PNG output; None = show only
    seed                : random seed for tile selection
    all_resolutions     : resolutions list for Mode B (default: TARGET_RESOLUTIONS)
    """
    if methods is None:
        methods = list(ALL_METHODS.keys())
    if all_resolutions is None:
        all_resolutions = sorted(TARGET_RESOLUTIONS)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "recon" if show_reconstruction else "ds"
    view_label = "Reconstruction (NN upsampled)" if show_reconstruction else "Downsampled"

    # ── Datatypes to iterate ──────────────────────────────────────────────────
    if datatype is not None:
        datatypes_to_plot = [datatype]
    else:
        present = [k.split("-", 1)[1] for k in DATA_DICT if k.startswith(f"{dataset}-")]
        datatypes_to_plot = [dt for dt in DATATYPE_ORDER if dt in present]
        datatypes_to_plot += sorted(dt for dt in present if dt not in DATATYPE_ORDER)

    # ── Shared tile: find a filename present in all datatype directories ───────
    tile_name = _find_common_tile(dataset, seed=seed)

    # ── Determine mode ────────────────────────────────────────────────────────
    resolution_mode = resolution is None   # True → Mode B

    for dt in datatypes_to_plot:
        subset_key = f"{dataset}-{dt}"
        img_dir    = DATA_DICT.get(subset_key)
        if img_dir is None:
            print(f"  Skipping unknown subset '{subset_key}'.")
            continue

        img_path = Path(img_dir) / tile_name
        if not img_path.exists():
            print(f"  Tile '{tile_name}' not found in {subset_key} — skipping.")
            continue

        img = load_image(img_path)
        if img is None:
            print(f"  Failed to load {img_path} — skipping.")
            continue

        native_h  = img.shape[0]
        orig_gray = to_grayscale(img)
        orig_panel = (f"Original\n({native_h} × {native_h})", orig_gray, None, None, None)

        # ── Mode B: resolution comparison, single method ───────────────────
        if resolution_mode:
            method_name = methods[0]
            fn          = ALL_METHODS.get(method_name)
            if fn is None:
                raise ValueError(f"Unknown method '{method_name}'.")

            panels = [orig_panel]
            for res in all_resolutions:
                if res >= native_h:
                    continue
                if show_reconstruction:
                    size_str = f"{res} → {native_h} px"
                else:
                    size_str = f"{res} × {res} px"
                panel_label = f"{res} px\n({size_str})"
                panels.append(_make_ds_panel(orig_gray, img, fn, res, native_h,
                                             show_reconstruction, panel_label))

            res_str  = "_".join(str(r) for r in all_resolutions)
            suptitle = (
                f"Down-Sampling Method: {method_name}  |  "
                f"Dataset: {dataset}  |  Datatype: {dt}  |  Tile: {tile_name}\n"
                f"Showing: {view_label}"
            )
            fname = (
                f"visual_rescomp"
                f"__{dataset}__{dt}"
                f"__method-{method_name}"
                f"__{mode_tag}"
                f"__tile-{Path(tile_name).stem}.png"
            )

        # ── Mode A: method comparison, single resolution ───────────────────
        else:
            if resolution >= native_h:
                print(f"  Resolution {resolution}px >= native {native_h}px "
                      f"for {dt} — skipping.")
                continue

            panels = [orig_panel]
            for method_name in methods:
                fn = ALL_METHODS.get(method_name)
                if fn is None:
                    raise ValueError(f"Unknown method '{method_name}'.")
                if show_reconstruction:
                    size_str = f"{resolution} → {native_h} px"
                else:
                    size_str = f"{resolution} × {resolution} px"
                panel_label = f"{method_name}\n({size_str})"
                panels.append(_make_ds_panel(orig_gray, img, fn, resolution, native_h,
                                             show_reconstruction, panel_label))

            methods_str = "-".join(methods)
            suptitle = (
                f"Resolution: {resolution} px  |  "
                f"Dataset: {dataset}  |  Datatype: {dt}  |  Tile: {tile_name}\n"
                # f"Showing: {view_label}"
            )
            fname = (
                f"visual_methcomp"
                f"__{dataset}__{dt}"
                f"__res-{resolution}px"
                f"__methods-{methods_str}"
                f"__{mode_tag}"
                f"__tile-{Path(tile_name).stem}.png"
            )

        save_path = (save_dir / fname) if save_dir is not None else None
        _render_comparison_figure(panels, suptitle, save_path)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Spatial resolution study for LithoBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--evaluate",    action="store_true")
    p.add_argument("--plot",        action="store_true")
    p.add_argument("--plot-metric", type=str, default=None, metavar="METRIC")
    p.add_argument("--plot-bar",    action="store_true")
    p.add_argument("--bar-metric",     type=str, default=None, metavar="METRIC")
    p.add_argument("--bar-resolution", type=int, default=None, metavar="N")
    p.add_argument("--bar-dataset",    type=str, default=None, metavar="D")
    p.add_argument("--tables",      action="store_true")
    p.add_argument("--aggregate",   action="store_true")
    p.add_argument("--workers",     type=int, default=None)
    p.add_argument("--timeout",     type=int, default=120)
    p.add_argument("--samples",     type=int, default=None)
    p.add_argument("--methods",     type=str, nargs="+", default=None, metavar="M",
                   help="Methods: pw, avg, fft. Default: all three.")
    p.add_argument("--metrics",     type=str, nargs="+", default=None, metavar="METRIC")
    p.add_argument("--resolutions", type=int, nargs="+", default=None, metavar="N")
    p.add_argument("--datasets",    type=str, nargs="+", default=None, metavar="D")
    p.add_argument("--force",       action="store_true")
    p.add_argument("--save-plots",  action="store_true")
    p.add_argument("--csv",         type=str, default=None)

    # ── Visual comparison ─────────────────────────────────────────────────────
    p.add_argument("--plot-visual", action="store_true",
                   help="Generate visual comparison figures.")
    p.add_argument("--visual-dataset", type=str, default=None, metavar="DS",
                   help="Dataset (e.g. metalset). Required for --plot-visual.")
    p.add_argument("--visual-datatype", type=str, default=None, metavar="DT",
                   help="Single datatype (e.g. PixelILT). "
                        "Omit to generate one figure per datatype using the same tile.")
    p.add_argument("--visual-resolution", type=int, default=None, metavar="N",
                   help="Resolution for method-comparison mode (e.g. 512). "
                        "Omit to compare resolutions instead.")
    p.add_argument("--visual-method", type=str, default="avg", metavar="M",
                   help="Method for resolution-comparison mode (default: avg). "
                        "Ignored when --methods is set or when --visual-resolution is given.")
    p.add_argument("--visual-seed", type=int, default=None, metavar="N",
                   help="Random seed for reproducible tile selection.")
    p.add_argument("--visual-reconstruction", action="store_true", default=False,
                   help="Show NN-upsampled reconstruction instead of the downsampled "
                        "image at native size (default: off).")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not any([args.evaluate, args.plot, args.tables, args.aggregate,
                args.plot_metric, args.plot_visual, args.plot_bar]):
        print("No action specified. Use --evaluate, --plot, --plot-metric, --plot-bar, "
              "--aggregate, --plot-visual, or --tables.")
        print("Run with --help for full usage.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)
    logger = get_logger()
    averaged_csv = Path(args.csv) if args.csv else OUTPUT_DIR / "results_average.csv"

    # ── Resolve method shorthands ─────────────────────────────────────────────
    resolved_methods = None
    if args.methods:
        resolved_methods = []
        for m in args.methods:
            canonical = METHOD_SHORTHAND.get(m.lower(), m)
            if canonical not in ALL_METHODS:
                print(f"Unknown method '{m}'. Valid: pw, avg, fft")
                sys.exit(1)
            resolved_methods.append(canonical)

    # ── Resolve dataset shorthands (for --datasets filter) ────────────────────
    filter_datasets = None
    if args.datasets:
        filter_datasets = []
        for d in args.datasets:
            canonical = DATASET_SHORTHAND.get(d.lower(), d)
            if canonical not in DATASET_SHORTHAND.values():
                print(f"Unknown dataset '{d}'. Valid: metalset, viaset, stdmetal, stdcontact")
                sys.exit(1)
            filter_datasets.append(canonical)

    filter_resolutions = args.resolutions
    filter_methods     = resolved_methods

    # ── Resolve metric shorthands ─────────────────────────────────────────────
    resolved_metrics = None
    if args.metrics:
        resolved_metrics = []
        for m in args.metrics:
            canonical = METRIC_SHORTHAND.get(m.lower(), m.lower())
            if canonical not in ALL_METRICS:
                print(f"Unknown metric '{m}'. Valid: psnr, ssim, mse, hf")
                sys.exit(1)
            if canonical not in resolved_metrics:
                resolved_metrics.append(canonical)

    logger.info(
        f"Session start | evaluate={args.evaluate} | plot={args.plot} | "
        f"tables={args.tables} | plot_bar={args.plot_bar} | plot_visual={args.plot_visual} | "
        f"workers={args.workers} | force={args.force} | samples={args.samples} | "
        f"methods={filter_methods} | averaged_csv={averaged_csv}"
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate:
        averaged_csv = run_evaluation(
            data_dict=DATA_DICT, output_dir=OUTPUT_DIR,
            num_workers=args.workers, num_samples=args.samples,
            eval_methods=filter_methods, eval_metrics=resolved_metrics,
            force=args.force, timeout=args.timeout,
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

    # ── Terminal tables ───────────────────────────────────────────────────────
    if args.tables:
        if not averaged_csv.exists():
            print("results_average.csv not found. Run --evaluate first.")
        else:
            print_detailed_tables(averaged_csv)

    # ── 4-metric overview plots ───────────────────────────────────────────────
    if args.plot:
        if not averaged_csv.exists():
            print("results_average.csv not found. Run --evaluate first.")
        else:
            print("\nGenerating plots ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_metrics_from_csv(
                str(averaged_csv), save_dir=save_dir,
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
                str(averaged_csv), metric=metric, save_dir=save_dir,
                filter_resolutions=filter_resolutions,
                filter_methods=filter_methods,
                filter_datasets=filter_datasets,
            )

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    if args.plot_bar:
        errors = []
        if not args.bar_metric:
            errors.append("--bar-metric is required (e.g. --bar-metric ssim)")
        if not args.bar_resolution:
            errors.append("--bar-resolution is required (e.g. --bar-resolution 512)")
        if not args.bar_dataset:
            errors.append("--bar-dataset is required (e.g. --bar-dataset metalset)")
        if errors:
            for e in errors:
                print(f"Error: {e}")
            sys.exit(1)

        bar_metric  = METRIC_SHORTHAND.get(args.bar_metric.lower(), args.bar_metric.lower())
        bar_dataset = DATASET_SHORTHAND.get(args.bar_dataset.lower(), args.bar_dataset)
        if bar_metric not in VALID_METRICS:
            print(f"Unknown metric '{args.bar_metric}'. Choose from: {VALID_METRICS}")
            sys.exit(1)

        if not averaged_csv.exists():
            print("results_average.csv not found. Run --evaluate or --aggregate first.")
        else:
            print(f"\nGenerating bar chart: metric={bar_metric}, "
                  f"resolution={args.bar_resolution}px, dataset={bar_dataset} ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_metric_bar(
                str(averaged_csv), metric=bar_metric,
                resolution=args.bar_resolution, dataset=bar_dataset,
                save_dir=save_dir, filter_methods=filter_methods,
            )

    # ── Visual comparison ─────────────────────────────────────────────────────
    if args.plot_visual:
        if args.visual_dataset is None:
            print("Error: --plot-visual requires --visual-dataset.")
            sys.exit(1)

        vis_dataset = DATASET_SHORTHAND.get(args.visual_dataset.lower(), args.visual_dataset)

        # ── Resolve methods for visual comparison ─────────────────────────────
        # Priority: --methods > --visual-method (for resolution mode)
        if filter_methods is not None:
            vis_methods = filter_methods
        elif args.visual_resolution is None:
            # Resolution mode: single method via --visual-method
            vm = METHOD_SHORTHAND.get(args.visual_method.lower(), args.visual_method)
            if vm not in ALL_METHODS:
                print(f"Unknown method '{args.visual_method}'. Valid: pw, avg, fft")
                sys.exit(1)
            vis_methods = [vm]
        else:
            # Method mode: use all three by default
            vis_methods = list(ALL_METHODS.keys())

        # Resolutions for Mode B
        vis_resolutions = filter_resolutions if filter_resolutions else sorted(TARGET_RESOLUTIONS, reverse=True)


        if args.visual_resolution is None:
            mode_desc = f"resolution comparison (method={vis_methods[0]})"
        else:
            mode_desc = f"method comparison (resolution={args.visual_resolution}px)"
        dt_desc = args.visual_datatype if args.visual_datatype else "all datatypes"

        print(f"\nGenerating visual comparisons: {mode_desc}, "
              f"dataset={vis_dataset}, datatypes={dt_desc}, "
              f"reconstruction={args.visual_reconstruction} ...")

        save_dir = str(OUTPUT_DIR) if args.save_plots else None
        plot_visual_comparison(
            dataset=vis_dataset,
            datatype=args.visual_datatype,
            resolution=args.visual_resolution,
            methods=vis_methods,
            show_reconstruction=args.visual_reconstruction,
            save_dir=save_dir,
            seed=args.visual_seed,
            all_resolutions=vis_resolutions,
        )

    logger.info("Session end")
