"""
metricstudy.py
==============
L2² and EPE metric study for the LithoBench dataset.

Computes the LithoBench / DiffOPC standard evaluation metrics between
ground-truth Printed / Resist tiles and the corresponding Target tiles.

---------------------------------------------------------------------------
Datatype distinctions
---------------------------------------------------------------------------
  Printed  — hard binary {0, 1} image obtained by thresholding the aerial
             image at I_th (CTR model, LithoBench §3.2).
  Resist   — continuous (0, 1) sigmoid image Z = σ(α(I − I_th)) stored as
             float in [0, 1].  This is the soft resist model used during ILT
             gradient optimisation.

Both are stored on disk as PNG; the script loads them as float32 in [0,1].
For EPE computation the candidate is binarised at --epe-bin-threshold
(default 0.5, which corresponds to the physical I_th after sigmoid
normalisation).  For L2² no binarisation is applied — the metric is
computed in the continuous domain for both datatypes, as per the DiffOPC
definition (eq. 3).  As a result, L2² values are *not* directly comparable
between Printed (integer-valued differences) and Resist (fractional
differences); this is expected and physically meaningful.

---------------------------------------------------------------------------
Metrics
---------------------------------------------------------------------------
  L2²  — ‖Z − T‖²₂  (sum of squared pixel differences, images in [0,1])
          Same as DiffOPC eq. 3 / LithoBench eq. 5.

  EPE  — LithoBench counting EPE (NOT the DiffOPC differentiable EPE loss).
          Probe points are sampled equidistantly (--epe-spacing px) along
          horizontal and vertical edges of the *target* binary pattern.
          Each probe has a well-defined edge normal (±row or ±col direction).
          A violation is recorded when the signed distance measured
          *along the normal* to the nearest boundary of the binarised
          candidate exceeds --epe-threshold pixels in magnitude.
          The EPE score is the integer count of violations.

          Note: DiffOPC's EPE is a differentiable training loss (windowed
          squared-pixel sums + sigmoid, eq. 7–8) — a smooth surrogate for
          this count, not an evaluation metric.  The counting EPE here
          matches the LithoBench benchmark definition.

Architecture
------------
Uses a flat image-pair work queue fed to a ProcessPoolExecutor — one task per
(target, candidate) image pair.  Results are streamed to a per-image CSV
immediately.  Aggregation (mean + std per subset) is deferred to the
aggregation step.

Two output CSVs
---------------
  metrics_per_image.csv   — one row per image pair with its L2² and EPE scores
  metrics_averaged.csv    — one row per subset with mean and std deviation

Resume behaviour
----------------
On startup the completed (subset, filename) pairs are read from
metrics_per_image.csv and excluded from the work queue.  Stop at any time
with Ctrl+C and resume by re-running the same command.
Use --force to discard existing results and start fresh.

Flags
-----
  --evaluate              run the parallel metric evaluation
  --aggregate             re-aggregate metrics_per_image.csv → metrics_averaged.csv
  --plot                  generate histograms from metrics_per_image.csv
  --plot-mean-std         generate mean±std comparison plots
  --tables                print per-subset mean/std tables to terminal
  --workers N             number of worker processes (default: cpu_count - 1)
  --batch-size N          image pairs per worker task batch (default: 16)
  --samples N             cap image pairs per subset (default: None = all)
  --force                 delete existing CSVs and start fresh
  --save-plots            save plot PNGs to the output directory
  --bins N                number of histogram bins (default: 40)
  --candidates A B ...    which candidate datatypes to evaluate against Target
                            choices: Printed, Resist (default: both)
                            e.g. --candidates Printed
                                 --candidates Resist
                                 --candidates Printed Resist
  --datatypes A B ...     restrict plots to these datatypes  e.g. --datatypes Printed Resist
  --datasets  A B ...     restrict plots to these datasets   e.g. --datasets MetalSet ViaSet
  --epe-spacing N         probe-point spacing along target edges in pixels (default: 2)
  --epe-threshold F       EPE violation distance threshold in pixels (default: 1.0)
  --epe-bin-threshold F   intensity threshold to binarise the candidate before EPE
                            (default: 0.5 = physical I_th post-sigmoid; no effect on
                            the already-binary Printed datatype)

Usage examples
--------------
  python metricstudy.py --evaluate
  python metricstudy.py --evaluate --workers 8 --samples 500
  python metricstudy.py --aggregate
  python metricstudy.py --plot --save-plots
  python metricstudy.py --plot-mean-std --save-plots
  python metricstudy.py --plot --datatypes Printed --datasets MetalSet ViaSet
  python metricstudy.py --tables
  python metricstudy.py --evaluate --aggregate --plot --save-plots
  python metricstudy.py --evaluate --candidates Printed
  python metricstudy.py --evaluate --candidates Resist
  python metricstudy.py --evaluate --force --workers 8
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
# DATA_ROOT and OUTPUT_DIR are resolved after argument parsing so that
# --data-root and --output-dir can override them.  These are placeholders.
DATA_ROOT  = None
OUTPUT_DIR = None
LOG_FILE   = None

# Default EPE parameters (matching LithoBench / DiffOPC conventions)
DEFAULT_EPE_SPACING       = 2      # probe point spacing in pixels
DEFAULT_EPE_THRESHOLD     = 1.0    # violation threshold in pixels
DEFAULT_EPE_BIN_THRESHOLD = 0.5    # candidate binarisation threshold for EPE
                                   # 0.5 = sigmoid midpoint = physical I_th

def _build_data_dict(data_root: Path) -> dict:
    """Build the flat DATA_DICT from a resolved data root path."""
    return {
        # ── MetalSet ──────────────────────────────────────────────────────────
        "MetalSet-Printed":   str(data_root / "MetalSet"       / "printed"),
        "MetalSet-Resist":    str(data_root / "MetalSet"       / "resist"),
        "MetalSet-Target":    str(data_root / "MetalSet"       / "target"),
        # ── ViaSet ────────────────────────────────────────────────────────────
        "ViaSet-Printed":     str(data_root / "ViaSet"         / "printed"),
        "ViaSet-Resist":      str(data_root / "ViaSet"         / "resist"),
        "ViaSet-Target":      str(data_root / "ViaSet"         / "target"),
        # ── StdMetal ──────────────────────────────────────────────────────────
        "StdMetal-Printed":   str(data_root / "StdMetal"       / "printed"),
        "StdMetal-Resist":    str(data_root / "StdMetal"       / "resist"),
        "StdMetal-Target":    str(data_root / "StdMetal"       / "target"),
        # ── StdContact ────────────────────────────────────────────────────────
        "StdContact-Printed": str(data_root / "StdContactFull" / "printed"),
        "StdContact-Resist":  str(data_root / "StdContactFull" / "resist"),
        "StdContact-Target":  str(data_root / "StdContactFull" / "target"),
    }

# ── CSV schemas ───────────────────────────────────────────────────────────────

PER_IMAGE_FIELDS = [
    "subset", "dataset", "datatype", "filename",
    "l2_sq", "epe_violations",
]

AVERAGED_FIELDS = [
    "subset", "dataset", "datatype",
    "num_samples",
    "mean_l2_sq",  "std_l2_sq",
    "mean_epe",    "std_epe",
]


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("metricstudy")
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
    return logging.getLogger("metricstudy")


# ──────────────────────────────────────────────────────────────────────────────
# Resume helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_completed_pairs(csv_path: Path) -> set:
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
# Image loading
# ──────────────────────────────────────────────────────────────────────────────

def load_image_gray(path: Path) -> np.ndarray:
    """
    Load image as float32 grayscale in [0, 1].
    Returns None on failure.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = img.astype("float32")
    if img.max() > 1.0:
        img /= 255.0
    return img


# ──────────────────────────────────────────────────────────────────────────────
# L2² metric
# ──────────────────────────────────────────────────────────────────────────────

def compute_l2_sq(candidate: np.ndarray, target: np.ndarray) -> float:
    """
    Squared L2 error between candidate image Z and target T:
        L2²(Z, T) = ‖Z − T‖²₂  (sum of squared pixel differences)

    Both inputs should be float32 arrays in [0, 1] with identical shape.
    This is the metric used in LithoBench / DiffOPC (eq. 3/5 in each paper).
    """
    diff = candidate.astype("float64") - target.astype("float64")
    return float(np.sum(diff * diff))


# ──────────────────────────────────────────────────────────────────────────────
# EPE metric
# ──────────────────────────────────────────────────────────────────────────────

def _binary(img: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Binarise a [0,1] float image to uint8 {0, 255}.

    For Printed tiles (already hard binary) the default thresh=0.5 has no
    practical effect.  For Resist tiles (sigmoid output, continuous in (0,1))
    thresh=0.5 corresponds to the physical intensity threshold I_th, i.e. the
    contour at which the sigmoid equals 0.5 — the standard binarisation point.
    The threshold is exposed as --epe-bin-threshold so it can be adjusted if
    needed.
    """
    return (img >= thresh).astype("uint8") * 255


def _sample_edge_probes_with_normals(
    binary_target: np.ndarray,
    spacing: int,
) -> tuple:
    """
    Detect horizontal-edge (top/bottom boundary) and vertical-edge
    (left/right boundary) pixels of the binary target pattern, sample them
    equidistantly, and return their coordinates together with the outward
    edge-normal direction for each probe.

    Returns
    -------
    probes : (N, 2) int32 array of (row, col) probe coordinates
    normals: (N, 2) int8  array of (drow, dcol) unit normal vectors
             For horizontal edges: normal is (±1, 0)  — perpendicular to edge
             For vertical   edges: normal is (0, ±1)  — perpendicular to edge

    Background
    ----------
    LithoBench EPE samples probes on horizontal and vertical edges of the
    target and checks distortion in the *normal direction* (perpendicular to
    the edge).  Using a directional normal rather than Euclidean distance is
    important at corners and for distinguishing inner vs. outer violations.
    """
    # Horizontal-edge kernel: erode only along the row direction.
    # Pixels that survive erosion are interior; XOR reveals top/bottom boundary.
    kernel_h = np.array([[0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0]], dtype="uint8")
    # Vertical-edge kernel: erode only along the column direction.
    kernel_v = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]], dtype="uint8")

    eroded_h = cv2.erode(binary_target, kernel_h)
    eroded_v = cv2.erode(binary_target, kernel_v)

    edge_h = cv2.bitwise_xor(binary_target, eroded_h)   # top/bottom boundary
    edge_v = cv2.bitwise_xor(binary_target, eroded_v)   # left/right boundary

    H, W = binary_target.shape
    mask_bool = binary_target > 0

    all_probes  = []
    all_normals = []

    # ── Horizontal-edge probes: normal is ±row direction ──────────────────────
    rows_h, cols_h = np.where(edge_h > 0)
    if len(rows_h) > 0:
        # Determine outward normal sign: if the pixel above is outside the mask
        # (or at border) → probe is on the top face → normal points upward (-row)
        dr = np.where(
            (rows_h == 0) | ~mask_bool[np.clip(rows_h - 1, 0, H-1), cols_h],
            -1, 1
        ).astype("int8")
        dc = np.zeros_like(dr)
        order  = np.lexsort((cols_h, rows_h))
        rows_s = rows_h[order][::spacing]
        cols_s = cols_h[order][::spacing]
        dr_s   = dr[order][::spacing]
        dc_s   = dc[order][::spacing]
        all_probes.append(np.stack([rows_s, cols_s], axis=1))
        all_normals.append(np.stack([dr_s, dc_s], axis=1))

    # ── Vertical-edge probes: normal is ±col direction ────────────────────────
    rows_v, cols_v = np.where(edge_v > 0)
    if len(rows_v) > 0:
        # Outward normal: if pixel to the left is outside → left face → normal (-col)
        dc = np.where(
            (cols_v == 0) | ~mask_bool[rows_v, np.clip(cols_v - 1, 0, W-1)],
            -1, 1
        ).astype("int8")
        dr = np.zeros_like(dc)
        order  = np.lexsort((cols_v, rows_v))
        rows_s = rows_v[order][::spacing]
        cols_s = cols_v[order][::spacing]
        dr_s   = dr[order][::spacing]
        dc_s   = dc[order][::spacing]
        all_probes.append(np.stack([rows_s, cols_s], axis=1))
        all_normals.append(np.stack([dr_s, dc_s], axis=1))

    if not all_probes:
        return (np.empty((0, 2), dtype="int32"),
                np.empty((0, 2), dtype="int8"))

    probes  = np.concatenate(all_probes,  axis=0).astype("int32")
    normals = np.concatenate(all_normals, axis=0).astype("int8")
    return probes, normals


def _normal_distance(
    bin_candidate: np.ndarray,
    probes:        np.ndarray,
    normals:       np.ndarray,
) -> np.ndarray:
    """
    For each probe, measure the signed displacement along the outward edge
    normal to the nearest boundary of bin_candidate.

    Strategy: march from each probe along its normal until we find a
    transition, or until we exceed a maximum search radius.  This is exact
    for Manhattan edges (the normal is always ±row or ±col) and avoids the
    diagonal ambiguity of full Euclidean distance transforms.

    Returns an array of signed distances (float32), one per probe.
    Positive  → probe is outside the candidate (erosion / inner violation).
    Negative  → probe is inside  the candidate (dilation / outer violation).
    """
    H, W        = bin_candidate.shape
    inside_mask = bin_candidate > 0
    N           = len(probes)
    # Maximum search radius: larger than any realistic EPE (tile size / 2)
    max_radius  = max(H, W) // 2

    signed_dists = np.empty(N, dtype="float32")

    for i in range(N):
        r, c   = int(probes[i, 0]),   int(probes[i, 1])
        dr, dc = int(normals[i, 0]),  int(normals[i, 1])

        probe_inside = inside_mask[r, c]

        # March outward along the normal to find the first boundary crossing
        dist = 0
        found = False
        for step in range(1, max_radius + 1):
            nr = r + dr * step
            nc = c + dc * step
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                dist = step   # hit image edge
                found = True
                break
            if inside_mask[nr, nc] != probe_inside:
                dist = step
                found = True
                break
        if not found:
            dist = max_radius

        # Sign: positive if probe is outside candidate, negative if inside
        if probe_inside:
            signed_dists[i] = -float(dist)
        else:
            signed_dists[i] = float(dist)

    return signed_dists


def compute_epe(
    candidate:     np.ndarray,
    target:        np.ndarray,
    spacing:       int   = DEFAULT_EPE_SPACING,
    threshold:     float = DEFAULT_EPE_THRESHOLD,
    bin_threshold: float = DEFAULT_EPE_BIN_THRESHOLD,
) -> int:
    """
    LithoBench counting EPE between candidate Z and target T.

    Probe points are placed equidistantly (spacing) along the horizontal and
    vertical edges of the binarised target.  For each probe the signed
    displacement to the nearest boundary of the binarised candidate is
    measured *along the outward edge normal* (±row for horizontal edges,
    ±col for vertical edges) — matching the LithoBench directional EPE
    definition.  A violation is recorded when |normal_displacement| > threshold.

    Note: this is the *counting* EPE used for evaluation in LithoBench (and
    reported in Tables 3/5 of DiffOPC).  It differs from the DiffOPC
    differentiable EPE loss (a windowed squared-pixel sum passed through a
    sigmoid, eq. 7–8 in the paper) which is a smooth training surrogate, not
    a violation count.

    Parameters
    ----------
    candidate      : float32 [0,1] candidate image (Printed or Resist)
    target         : float32 [0,1] binary target image
    spacing        : probe point sampling interval in pixels along each edge
    threshold      : violation threshold in pixels (typically 1 px = 1 nm at
                     the 2048×2048 / 1px=1nm LithoBench resolution)
    bin_threshold  : intensity threshold used to binarise the candidate
                     (default 0.5 = sigmoid midpoint / physical I_th)
    """
    bin_target    = _binary(target,    thresh=0.5)          # target is always binary
    bin_candidate = _binary(candidate, thresh=bin_threshold)

    probes, normals = _sample_edge_probes_with_normals(bin_target, spacing)
    if len(probes) == 0:
        return 0

    signed_dists = _normal_distance(bin_candidate, probes, normals)
    violations   = int(np.sum(np.abs(signed_dists) > threshold))
    return violations


# ──────────────────────────────────────────────────────────────────────────────
# Worker function
# ──────────────────────────────────────────────────────────────────────────────

def _process_pair(task: tuple):
    """
    Compute L2² and EPE for a single (candidate, target) image pair.

    Parameters
    ----------
    task : (subset, dataset, datatype, filename,
            candidate_path_str, target_path_str,
            epe_spacing, epe_threshold, epe_bin_threshold, log_file)

    Returns
    -------
    dict row on success, None on failure.
    """
    import matplotlib
    matplotlib.use("Agg")

    (subset, dataset, datatype, filename,
     cand_path_str, tgt_path_str,
     epe_spacing, epe_threshold, epe_bin_threshold, log_file) = task

    logger = setup_logging(Path(log_file))
    logger.debug(f"Processing {subset}/{filename}")

    cand = load_image_gray(Path(cand_path_str))
    tgt  = load_image_gray(Path(tgt_path_str))

    if cand is None or tgt is None:
        logger.warning(f"Failed to load images for {subset}/{filename} — skipping")
        return None

    # Resize candidate to match target if needed (should not happen in practice)
    if cand.shape != tgt.shape:
        cand = cv2.resize(cand, (tgt.shape[1], tgt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    try:
        l2_sq = compute_l2_sq(cand, tgt)
        epe   = compute_epe(cand, tgt, spacing=epe_spacing,
                            threshold=epe_threshold,
                            bin_threshold=epe_bin_threshold)

        logger.debug(
            f"DONE {subset}/{filename}  L2²={l2_sq:.2f}  EPE={epe}"
        )
        return {
            "subset":         subset,
            "dataset":        dataset,
            "datatype":       datatype,
            "filename":       filename,
            "l2_sq":          round(l2_sq, 4),
            "epe_violations": epe,
        }
    except Exception as exc:
        logger.warning(f"ERROR {subset}/{filename}: {exc}")
        return None


def _process_pair_batch(batch_tasks: list):
    """Process a batch of pairs in one worker call."""
    rows = []
    failed = 0
    for task in batch_tasks:
        row = _process_pair(task)
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
    Read metrics_per_image.csv and compute mean + std per subset,
    writing results to metrics_averaged.csv.
    """
    records: dict = {}  # subset → {"l2": [...], "epe": [...], "dataset": ..., "datatype": ...}

    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s  = row["subset"]
            l2 = row.get("l2_sq", "")
            ep = row.get("epe_violations", "")
            if l2 == "" or ep == "":
                continue
            try:
                fl2 = float(l2)
                fep = float(ep)
            except ValueError:
                continue
            if math.isnan(fl2) or math.isinf(fl2):
                continue
            if s not in records:
                records[s] = {
                    "dataset":  row["dataset"],
                    "datatype": row["datatype"],
                    "l2":  [],
                    "epe": [],
                }
            records[s]["l2"].append(fl2)
            records[s]["epe"].append(fep)

    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AVERAGED_FIELDS)
        w.writeheader()
        for subset in sorted(records.keys()):
            r  = records[subset]
            l2 = np.array(r["l2"])
            ep = np.array(r["epe"])
            w.writerow({
                "subset":      subset,
                "dataset":     r["dataset"],
                "datatype":    r["datatype"],
                "num_samples": len(l2),
                "mean_l2_sq":  round(float(np.mean(l2)), 4),
                "std_l2_sq":   round(float(np.std(l2)),  4),
                "mean_epe":    round(float(np.mean(ep)), 4),
                "std_epe":     round(float(np.std(ep)),  4),
            })

    return averaged_csv


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    data_dict:         dict,
    output_dir:        Path,
    num_workers:       int   = None,
    num_samples:       int   = None,
    batch_size:        int   = 16,
    force:             bool  = False,
    timeout:           int   = 120,
    epe_spacing:       int   = DEFAULT_EPE_SPACING,
    epe_threshold:     float = DEFAULT_EPE_THRESHOLD,
    epe_bin_threshold: float = DEFAULT_EPE_BIN_THRESHOLD,
    candidates:        list  = None,   # e.g. ["printed"], ["resist"], or both
):
    """
    Build a flat image-pair work queue for all (dataset, datatype) combinations,
    dispatch to a ProcessPoolExecutor, stream results to metrics_per_image.csv,
    then aggregate to metrics_averaged.csv.
    """
    import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "metrics_per_image.csv"
    averaged_csv  = output_dir / "metrics_averaged.csv"
    logger        = get_logger()

    if force:
        for p in (per_image_csv, averaged_csv):
            if p.exists():
                p.unlink()
        logger.info("--force: cleared existing CSVs")
        print("--force: cleared existing results, starting fresh.\n")

    completed = load_completed_pairs(per_image_csv)
    logger.info(f"Already completed: {len(completed)} pairs")

    print("Scanning dataset directories ...")
    tasks = []

    # Build nested lookup: dataset -> {datatype -> dir_path}
    # from the flat DATA_DICT keys ("Dataset-Datatype")
    nested: dict = {}
    for key, path in data_dict.items():
        ds, dt = key.rsplit("-", 1)
        nested.setdefault(ds, {})[dt.lower()] = Path(path)

    for dataset, dt_map in nested.items():
        tgt_dir = dt_map.get("target")
        if tgt_dir is None or not tgt_dir.exists():
            print(f"  WARNING: target dir not found — skipping {dataset}")
            logger.warning(f"Target dir not found, skipping dataset: {dataset}")
            continue

        tgt_files = {f.name: f for f in sorted(tgt_dir.iterdir()) if f.is_file()}

        active_candidates = [c.lower() for c in candidates] if candidates else ["printed", "resist"]
        for datatype in active_candidates:
            cand_dir = dt_map.get(datatype)
            if cand_dir is None or not cand_dir.exists():
                logger.info(f"No {datatype} dir for {dataset}, skipping.")
                continue

            cand_files = {f.name: f for f in sorted(cand_dir.iterdir()) if f.is_file()}
            common     = sorted(set(tgt_files) & set(cand_files))

            if num_samples is not None:
                common = random.sample(common, min(num_samples, len(common)))

            subset = f"{dataset}-{datatype.capitalize()}"

            for fname in common:
                if (subset, fname) in completed:
                    continue
                tasks.append((
                    subset, dataset, datatype.capitalize(),
                    fname,
                    str(cand_files[fname]),
                    str(tgt_files[fname]),
                    epe_spacing, epe_threshold, epe_bin_threshold,
                    str(LOG_FILE),
                ))

    pending = len(tasks)

    if completed:
        print(f"\nResuming — {len(completed)} pairs already done, "
              f"{pending} remaining.\n")
    else:
        print(f"\nStarting fresh — {pending} image pairs to process.\n")

    if not pending:
        if completed:
            print("Nothing to do — all pairs already processed.")
            logger.info("All pairs complete, aggregating.")
            aggregate_to_averaged_csv(per_image_csv, averaged_csv)
            print(f"metrics_averaged.csv written to: {averaged_csv}")
        else:
            print("No image pairs found. Check that DATA_ROOT points to your "
                  "lithobench-main directory and that printed/resist/target "
                  "subdirectories exist.")
            logger.warning("No pairs found and no prior results — nothing to do.")
        return per_image_csv, averaged_csv

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, pending)
    batch_size  = max(1, int(batch_size))

    batched_tasks = [
        tasks[i:i + batch_size]
        for i in range(0, len(tasks), batch_size)
    ]

    logger.info(
        f"Starting evaluation | pairs={pending} | workers={num_workers} | "
        f"batch_size={batch_size} | candidates={active_candidates} | "
        f"epe_spacing={epe_spacing} | epe_threshold={epe_threshold} | "
        f"epe_bin_threshold={epe_bin_threshold}"
    )
    print(
        f"Workers: {num_workers}  |  Pairs: {pending}  |  "
        f"Batch size: {batch_size} ({len(batched_tasks)} batches)\n"
    )

    failed   = 0
    timedout = 0

    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(_process_pair_batch, batch): batch
            for batch in batched_tasks
        }

        with tqdm.tqdm(
            total=pending,
            desc="Computing L2² and EPE",
            unit="pair",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for future in cf.as_completed(future_to_batch):
                batch        = future_to_batch[future]
                first_subset = batch[0][0]
                first_file   = batch[0][3]
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
                        f"{first_subset}/{first_file} — skipped {len(batch)} pairs"
                    )
                    pbar.set_postfix_str(
                        f"timeout: {first_file[:20]}",
                        refresh=True,
                    )
                except Exception as exc:
                    failed += len(batch)
                    logger.error(
                        f"ERROR batch starting {first_subset}/{first_file}: {exc}"
                    )

                pbar.update(len(batch))

    issues = failed + timedout
    if issues:
        print(f"\n  {failed} failed, {timedout} timed out. "
              f"See {LOG_FILE} for details.")
    else:
        print("\n  All pairs processed successfully.")

    print("\nAggregating results ...")
    aggregate_to_averaged_csv(per_image_csv, averaged_csv)
    logger.info(f"Aggregation complete: {averaged_csv}")
    print(f"  metrics_per_image.csv : {per_image_csv}")
    print(f"  metrics_averaged.csv  : {averaged_csv}")

    return per_image_csv, averaged_csv


# ──────────────────────────────────────────────────────────────────────────────
# Terminal table
# ──────────────────────────────────────────────────────────────────────────────

def print_tables(averaged_csv: Path):
    """Print mean and std for L2² and EPE per subset, grouped by dataset."""
    rows = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)

    cw = {"dt": 10, "n": 7, "ml2": 14, "sl2": 14, "mepe": 10, "sepe": 10}
    header = (
        f"{'Datatype':<{cw['dt']}}"
        f"  {'N':>{cw['n']}}"
        f"  {'Mean L2²':>{cw['ml2']}}"
        f"  {'Std L2²':>{cw['sl2']}}"
        f"  {'Mean EPE':>{cw['mepe']}}"
        f"  {'Std EPE':>{cw['sepe']}}"
    )
    width = len(header)

    for dataset in [d for d in ["MetalSet", "ViaSet", "StdMetal", "StdContact"]
                    if d in groups] + \
                   [d for d in sorted(groups) if d not in
                    ["MetalSet", "ViaSet", "StdMetal", "StdContact"]]:
        dataset_rows = groups[dataset]
        print(f"\n{'=' * width}")
        print(f"  {dataset}")
        print(f"{'=' * width}")
        print(header)
        print("-" * width)
        for row in sorted(dataset_rows, key=lambda r: r["datatype"]):
            print(
                f"{row['datatype']:<{cw['dt']}}"
                f"  {row['num_samples']:>{cw['n']}}"
                f"  {float(row['mean_l2_sq']):>{cw['ml2']}.2f}"
                f"  {float(row['std_l2_sq']):>{cw['sl2']}.2f}"
                f"  {float(row['mean_epe']):>{cw['mepe']}.4f}"
                f"  {float(row['std_epe']):>{cw['sepe']}.4f}"
            )
        print("-" * width)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers — filter & ordering
# ──────────────────────────────────────────────────────────────────────────────

DATASET_COLORS = {
    "MetalSet":   "#2E86AB",
    "ViaSet":     "#E07B39",
    "StdContact": "#6A994E",
    "StdMetal":   "#9B5DE5",
}
DEFAULT_COLOR = "#888888"

DATASET_ORDER = ["MetalSet", "ViaSet", "StdMetal", "StdContact"]


def _resolve_filters(datatypes_arg, datasets_arg):
    dt_filter = {d.lower() for d in datatypes_arg} if datatypes_arg else None
    ds_filter = {d.lower() for d in datasets_arg}  if datasets_arg  else None
    return dt_filter, ds_filter


def _apply_dataset_filter_and_order(datasets: list, ds_filter) -> list:
    if ds_filter is not None:
        datasets = [ds for ds in datasets if ds.lower() in ds_filter]
    known   = [ds for ds in DATASET_ORDER if ds in datasets]
    unknown = sorted(ds for ds in datasets if ds not in DATASET_ORDER)
    return known + unknown


def _apply_datatype_filter(datatypes: list, dt_filter) -> list:
    if dt_filter is None:
        return datatypes
    return [dt for dt in datatypes if dt.lower() in dt_filter]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (histograms)
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_histograms(
    per_image_csv: str,
    bins:          int  = 40,
    save_dir:      str  = None,
    dt_filter:     set  = None,
    ds_filter:     set  = None,
):
    """
    Generate one figure per (metric, datatype) combination.
    Each figure contains one subplot per dataset.
    Mean and ±1σ are marked as vertical lines.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    records_l2: dict  = {}   # (dataset, datatype) → [float]
    records_epe: dict = {}

    with open(per_image_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l2  = row.get("l2_sq", "")
            epe = row.get("epe_violations", "")
            if l2 == "" or epe == "":
                continue
            try:
                fl2  = float(l2)
                fepe = float(epe)
            except ValueError:
                continue
            if math.isnan(fl2) or math.isinf(fl2):
                continue
            key = (row["dataset"], row["datatype"])
            records_l2.setdefault(key,  []).append(fl2)
            records_epe.setdefault(key, []).append(fepe)

    datatypes = _apply_datatype_filter(
        sorted({dt for _, dt in records_l2.keys()}), dt_filter)
    datasets  = _apply_dataset_filter_and_order(
        list({ds for ds, _ in records_l2.keys()}), ds_filter)

    for metric_label, metric_xlabel, records in [
        ("L2²",   "L2² Score",         records_l2),
        ("EPE",   "EPE Violations",     records_epe),
    ]:
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

                ax.axvline(mean,       color="#222222", linewidth=1.8,
                           linestyle="-",  label=f"Mean {mean:.2f}")
                ax.axvline(mean - std, color="#222222", linewidth=1.2,
                           linestyle="--", label=f"±1σ  {std:.2f}")
                ax.axvline(mean + std, color="#222222", linewidth=1.2,
                           linestyle="--")

                ax.set_title(f"{dataset}\nn={n:,}", fontsize=13, fontweight="bold")
                ax.set_xlabel(metric_xlabel, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.tick_params(axis="both", labelsize=11)
                ax.legend(fontsize=10, framealpha=0.85)
                ax.grid(True, alpha=0.2, linestyle="--")

            fig.suptitle(
                f"{metric_label} — Datatype: {datatype}",
                fontsize=15, fontweight="bold", y=0.97,
            )

            if save_dir is not None:
                slug = metric_label.replace("²", "_sq").replace(" ", "_")
                out  = save_dir / f"hist_{slug}_{datatype}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"  Saved: {out}")

            plt.show()
            plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (mean ± std bar / errorbar)
# ──────────────────────────────────────────────────────────────────────────────

def plot_mean_std(
    averaged_csv: str,
    save_dir:     str = None,
    dt_filter:    set = None,
    ds_filter:    set = None,
):
    """
    Two side-by-side subplots per datatype: L2² mean±std  and  EPE mean±std.
    Each point is one dataset.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(averaged_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "subset":    row["subset"],
                "dataset":   row["dataset"],
                "datatype":  row["datatype"],
                "mean_l2":   float(row["mean_l2_sq"]),
                "std_l2":    float(row["std_l2_sq"]),
                "mean_epe":  float(row["mean_epe"]),
                "std_epe":   float(row["std_epe"]),
                "n":         int(row["num_samples"]),
            })

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

        fig, (ax_l2, ax_epe) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Datatype: {datatype}", fontsize=14, fontweight="bold")
        x = np.arange(len(rows))

        for i, row in enumerate(rows):
            color = DATASET_COLORS.get(row["dataset"], DEFAULT_COLOR)
            for ax, mean_key, std_key in [
                (ax_l2,  "mean_l2",  "std_l2"),
                (ax_epe, "mean_epe", "std_epe"),
            ]:
                ax.errorbar(
                    x[i],
                    row[mean_key],
                    yerr=row[std_key],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=6,
                    markersize=8,
                    label=row["dataset"],
                )

        for ax, ylabel, title in [
            (ax_l2,  "Mean L2² Score",       "L2² (Squared L2 Error)"),
            (ax_epe, "Mean EPE Violations",   "EPE (Edge Placement Error)"),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels([r["dataset"] for r in rows], rotation=25,
                               ha="right", fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_xlabel("Dataset", fontsize=13)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")
            ax.margins(x=0.12)

        plt.tight_layout()

        if save_dir is not None:
            out = Path(save_dir) / f"mean_std_{datatype}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

        plt.show()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="L2² and EPE metric study for the LithoBench dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--evaluate",      action="store_true",
                   help="Run the parallel metric evaluation.")
    p.add_argument("--aggregate",     action="store_true",
                   help="Re-aggregate metrics_per_image.csv → metrics_averaged.csv.")
    p.add_argument("--plot",          action="store_true",
                   help="Generate histograms from metrics_per_image.csv.")
    p.add_argument("--plot-mean-std", action="store_true",
                   help="Generate mean±std comparison plots.")
    p.add_argument("--tables",        action="store_true",
                   help="Print per-subset mean/std tables to terminal.")
    p.add_argument("--workers",       type=int, default=None,
                   help="Worker processes (default: cpu_count - 1).")
    p.add_argument("--batch-size",    type=int, default=16,
                   help="Image pairs per worker task batch (default: 16).")
    p.add_argument("--samples",       type=int, default=None,
                   help="Cap image pairs per subset — None means all.")
    p.add_argument("--timeout",       type=int, default=120,
                   help="Per-batch timeout in seconds (default: 120).")
    p.add_argument("--force",         action="store_true",
                   help="Delete existing CSVs and start fresh.")
    p.add_argument("--save-plots",    action="store_true",
                   help="Save plot PNGs to the output directory.")
    p.add_argument("--bins",          type=int, default=40,
                   help="Number of histogram bins (default: 40).")
    p.add_argument("--data-root",      type=str, default=None,
                   help="Path to the lithobench-main directory. "
                        "Defaults to ../lithobench-main relative to this script.")
    p.add_argument("--output-dir",     type=str, default=None,
                   help="Directory for output CSVs and plots. "
                        "Defaults to metric_study_output/ next to this script.")
    p.add_argument("--candidates",     nargs="+", default=None, metavar="CANDIDATE",
                   help="Which candidate datatypes to evaluate against Target. "
                        "Choices: Printed, Resist (default: both). "
                        "E.g. --candidates Printed  or  --candidates Printed Resist")
    p.add_argument("--datatypes",     nargs="+", default=None, metavar="DATATYPE",
                   help="Restrict all plots to these datatypes (case-insensitive). "
                        "E.g. --datatypes Printed Resist")
    p.add_argument("--datasets",      nargs="+", default=None, metavar="DATASET",
                   help="Restrict all plots to these datasets (case-insensitive). "
                        "E.g. --datasets MetalSet ViaSet")
    p.add_argument("--epe-spacing",   type=int, default=DEFAULT_EPE_SPACING,
                   help=f"Probe-point spacing along target edges in pixels "
                        f"(default: {DEFAULT_EPE_SPACING}).")
    p.add_argument("--epe-threshold",     type=float, default=DEFAULT_EPE_THRESHOLD,
                   help=f"EPE violation distance threshold in pixels "
                        f"(default: {DEFAULT_EPE_THRESHOLD}).")
    p.add_argument("--epe-bin-threshold", type=float, default=DEFAULT_EPE_BIN_THRESHOLD,
                   help=f"Intensity threshold to binarise the candidate image before "
                        f"EPE computation (default: {DEFAULT_EPE_BIN_THRESHOLD}). "
                        f"Corresponds to sigmoid midpoint / physical I_th. "
                        f"Has no effect on the already-binary Printed datatype.")
    args = p.parse_args()
    if args.candidates:
        valid = {"printed", "resist"}
        bad = [c for c in args.candidates if c.lower() not in valid]
        if bad:
            p.error(f"--candidates: invalid choice(s): {bad}. Choose from Printed, Resist.")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not any([args.evaluate, args.aggregate, args.plot,
                args.plot_mean_std, args.tables]):
        print("No action specified. Use --evaluate, --aggregate, --plot, "
              "--plot-mean-std, or --tables.")
        print("Run with --help for full usage.")
        sys.exit(0)

    # Resolve paths now that args are available
    DATA_ROOT  = Path(args.data_root).resolve()  if args.data_root  else PROJECT_ROOT.parent / "lithobench-main"
    OUTPUT_DIR = Path(args.output_dir).resolve() if args.output_dir else PROJECT_ROOT / "metric_study_output"
    LOG_FILE   = OUTPUT_DIR / "metric_study.log"
    DATA_DICT  = _build_data_dict(DATA_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_FILE)
    logger = get_logger()

    per_image_csv = OUTPUT_DIR / "metrics_per_image.csv"
    averaged_csv  = OUTPUT_DIR / "metrics_averaged.csv"

    dt_filter, ds_filter = _resolve_filters(args.datatypes, args.datasets)

    logger.info(
        f"Session start | evaluate={args.evaluate} | aggregate={args.aggregate} | "
        f"plot={args.plot} | plot_mean_std={args.plot_mean_std} | "
        f"tables={args.tables} | workers={args.workers} | "
        f"batch_size={args.batch_size} | samples={args.samples} | "
        f"force={args.force} | candidates={args.candidates} | "
        f"datatypes={args.datatypes} | datasets={args.datasets} | "
        f"epe_spacing={args.epe_spacing} | epe_threshold={args.epe_threshold} | "
        f"epe_bin_threshold={args.epe_bin_threshold}"
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate:
        num_samples = args.samples
        per_image_csv, averaged_csv = run_evaluation(
            data_dict         = DATA_DICT,
            output_dir        = OUTPUT_DIR,
            num_workers       = args.workers,
            num_samples       = num_samples,
            batch_size        = args.batch_size,
            force             = args.force,
            timeout           = args.timeout,
            epe_spacing       = args.epe_spacing,
            epe_threshold     = args.epe_threshold,
            epe_bin_threshold = args.epe_bin_threshold,
            candidates        = args.candidates,
        )

    # ── Standalone aggregation ────────────────────────────────────────────────
    if args.aggregate:
        if not per_image_csv.exists():
            print("metrics_per_image.csv not found. Run --evaluate first.")
        else:
            print("Aggregating metrics_per_image.csv ...")
            aggregate_to_averaged_csv(per_image_csv, averaged_csv)
            print(f"Done. metrics_averaged.csv written to: {averaged_csv}")
            logger.info(f"Standalone aggregation complete: {averaged_csv}")

    # ── Terminal tables ───────────────────────────────────────────────────────
    if args.tables:
        if not averaged_csv.exists():
            print("metrics_averaged.csv not found. "
                  "Run --evaluate or --aggregate first.")
        else:
            print_tables(averaged_csv)

    # ── Histograms ────────────────────────────────────────────────────────────
    if args.plot:
        if not per_image_csv.exists():
            print("metrics_per_image.csv not found. Run --evaluate first.")
        else:
            print("\nGenerating metric histograms ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_metric_histograms(
                str(per_image_csv),
                bins      = args.bins,
                save_dir  = save_dir,
                dt_filter = dt_filter,
                ds_filter = ds_filter,
            )

    # ── Mean / std plots ──────────────────────────────────────────────────────
    if args.plot_mean_std:
        if not averaged_csv.exists():
            print("metrics_averaged.csv not found. "
                  "Run --evaluate or --aggregate first.")
        else:
            print("\nGenerating mean±std plots ...")
            save_dir = str(OUTPUT_DIR) if args.save_plots else None
            plot_mean_std(
                str(averaged_csv),
                save_dir  = save_dir,
                dt_filter = dt_filter,
                ds_filter = ds_filter,
            )

    logger.info("Session end")
