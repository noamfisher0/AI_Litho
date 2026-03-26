import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tqdm
from skimage.transform import resize, downscale_local_mean


# ──────────────────────────────────────────────────────────────────────────────
# Paths & Dataset
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "lithobench-main"

DATA_DICT = {
    "MetalSet-Printed":     str(DATA_ROOT / "MetalSet"    / "printed"),
    "MetalSet-Resist":      str(DATA_ROOT / "MetalSet"    / "resist"),
    "MetalSet-Target":      str(DATA_ROOT / "MetalSet"    / "target"),
    "MetalSet-LevelILT":    str(DATA_ROOT / "MetalSet"    / "levelsetILT"),
    "MetalSet-Litho":       str(DATA_ROOT / "MetalSet"    / "litho"),
    "MetalSet-PixelILT":    str(DATA_ROOT / "MetalSet"    / "pixelILT"),
    "ViaSet-Printed":       str(DATA_ROOT / "ViaSet"      / "printed"),
    "ViaSet-Resist":        str(DATA_ROOT / "ViaSet"      / "resist"),
    "ViaSet-Target":        str(DATA_ROOT / "ViaSet"      / "target"),
    "ViaSet-LevelILT":      str(DATA_ROOT / "ViaSet"      / "levelsetILT"),
    "ViaSet-Litho":         str(DATA_ROOT / "ViaSet"      / "litho"),
    "ViaSet-PixelILT":      str(DATA_ROOT / "ViaSet"      / "pixelILT"),
    "StdContact-Printed":   str(DATA_ROOT / "StdContact"  / "printed"),
    "StdContact-Resist":    str(DATA_ROOT / "StdContact"  / "resist"),
    "StdContact-Target":    str(DATA_ROOT / "StdContact"  / "target"),
    "StdContact-Litho":     str(DATA_ROOT / "StdContact"  / "litho"),
    "StdContact-PixelILT":  str(DATA_ROOT / "StdContact"  / "pixelILT"),
    "StdMetal-Printed":     str(DATA_ROOT / "StdMetal"    / "printed"),
    "StdMetal-Resist":      str(DATA_ROOT / "StdMetal"    / "resist"),
    "StdMetal-Target":      str(DATA_ROOT / "StdMetal"    / "target"),
    "StdMetal-Litho":       str(DATA_ROOT / "StdMetal"    / "litho"),
    "StdMetal-PixelILT":    str(DATA_ROOT / "StdMetal"    / "pixelILT"),
}

# Target resolutions to evaluate (must be integer divisors of 2048 for Average
# method; Point-Wise and Fourier work with any value)
TARGET_RESOLUTIONS = [1024, 512, 256, 128]

# How many images to sample per subset for the study
NUM_SAMPLES = 5


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_dict, selected_keys=None, number_of_samples=None):
    if selected_keys is None:
        selected_keys = list(data_dict.keys())

    data = {}
    for key in selected_keys:
        path = Path(data_dict[key])
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")

        files = [p.name for p in path.iterdir() if p.is_file()]
        selected_files = (
            random.sample(files, min(number_of_samples, len(files)))
            if number_of_samples is not None else files
        )

        data[key] = []
        for file in tqdm.tqdm(selected_files, desc=f"Loading {key}"):
            img = plt.imread(path / file)
            data[key].append(img)

        data[key] = np.array(data[key])

    return data


def to_grayscale(image):
    """Convert image to grayscale if it is RGB/RGBA."""
    if image.ndim == 3:
        return np.mean(image[..., :3], axis=-1)
    return image


# ──────────────────────────────────────────────────────────────────────────────
# Downsampling methods
# ──────────────────────────────────────────────────────────────────────────────

def downsample_pointwise(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Point-Wise (Nearest-Neighbour) Downsampling
    -------------------------------------------
    Selects every N-th pixel along each axis (stride = original / target).
    No smoothing is applied — the output pixel value is simply copied from
    the nearest source pixel.  Very fast, zero blurring, but can introduce
    aliasing artefacts (jagged edges, moire patterns) when the scale factor
    is large, because high-frequency content is not removed before subsampling.
    """
    h, w = image.shape[:2]
    stride_h = h // target_size
    stride_w = w // target_size
    return image[::stride_h, ::stride_w][: target_size, : target_size]


def downsample_average(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Average (Block-Mean) Downsampling
    ----------------------------------
    Partitions the image into non-overlapping blocks of size
    (original / target) x (original / target) and replaces each block with
    its mean value.  This acts as a low-pass filter before subsampling,
    which suppresses aliasing.  The result is smoother than Point-Wise but
    may appear slightly blurred because fine detail is averaged out.
    Requires the original size to be exactly divisible by the target size.
    Uses skimage's `downscale_local_mean` under the hood.
    """
    h, w = image.shape[:2]
    factor_h = h // target_size
    factor_w = w // target_size
    if image.ndim == 3:
        factors = (factor_h, factor_w, 1)
    else:
        factors = (factor_h, factor_w)
    return downscale_local_mean(image, factors).astype(image.dtype)


def downsample_fourier(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Fourier (Frequency-Domain) Downsampling
    ----------------------------------------
    Transforms the image to the frequency domain via a 2-D FFT, retains only
    the central low-frequency coefficients (a crop of size target x target in
    frequency space), then inverse-transforms back to the spatial domain.
    This is the theoretically optimal anti-aliasing approach: it enforces the
    Nyquist limit exactly and preserves all frequencies that are representable
    at the target resolution while discarding those that would alias.
    Works on real-valued images; the imaginary residual after iFFT is
    discarded (it is numerically negligible for real inputs).
    """
    if image.ndim == 3:
        channels = [_fourier_channel(image[..., c], target_size)
                    for c in range(image.shape[2])]
        result = np.stack(channels, axis=-1)
    else:
        result = _fourier_channel(image, target_size)

    # Rescale to the original value range to avoid clipping artefacts
    orig_min, orig_max = image.min(), image.max()
    result = np.clip(result, result.min(), result.max())
    if result.max() - result.min() > 1e-8:
        result = (result - result.min()) / (result.max() - result.min())
        result = result * (orig_max - orig_min) + orig_min
    return result.astype(np.float32)


def _fourier_channel(channel: np.ndarray, target_size: int) -> np.ndarray:
    """Helper: Fourier downsampling for a single 2-D channel."""
    F = np.fft.fftshift(np.fft.fft2(channel.astype(np.float64)))
    h, w = F.shape
    ch, cw = h // 2, w // 2
    half = target_size // 2
    F_crop = F[ch - half: ch + half, cw - half: cw + half]
    # Scale factor preserves mean pixel energy
    scale = (target_size ** 2) / (h * w)
    reconstructed = np.fft.ifft2(np.fft.ifftshift(F_crop)).real * scale
    return reconstructed.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def upsample_to_original(image: np.ndarray, original_size: int) -> np.ndarray:
    """Upscale a downsampled image back to original_size x original_size
    (nearest-neighbour, so no extra smoothing is introduced)."""
    return resize(image, (original_size, original_size),
                  order=0, preserve_range=True, anti_aliasing=False
                  ).astype(np.float32)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error — lower is better."""
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    return float(np.mean((o - r) ** 2))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray,
                 data_range: float = None) -> float:
    """
    Peak Signal-to-Noise Ratio (dB) — higher is better.
    data_range defaults to the original image's value range.
    """
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(original.max() - original.min())
        if data_range == 0:
            return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) — ranges [-1, 1]; 1 = identical.
    Uses the standard constants C1=(0.01*L)^2, C2=(0.03*L)^2 with L=data_range.
    """
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    L = float(o.max() - o.min()) or 1.0
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2

    mu_o, mu_r = o.mean(), r.mean()
    sigma_o = np.sqrt(np.mean((o - mu_o) ** 2))
    sigma_r = np.sqrt(np.mean((r - mu_r) ** 2))
    sigma_or = np.mean((o - mu_o) * (r - mu_r))

    num = (2 * mu_o * mu_r + C1) * (2 * sigma_or + C2)
    den = (mu_o ** 2 + mu_r ** 2 + C1) * (sigma_o ** 2 + sigma_r ** 2 + C2)
    return float(num / den)


def compute_hf_ratio(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    High-Frequency Energy Retention Ratio.
    Computes the fraction of high-frequency energy (outside the central 50%
    of the Fourier spectrum) retained after downsampling+upsampling.
    A ratio close to 1 means fine detail is well preserved; close to 0 means
    high-frequency content has been lost.
    """
    def hf_energy(img):
        F = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float64))))
        h, w = F.shape
        mask = np.ones((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = False
        return float(np.sum(F[mask] ** 2))

    o_gray = to_grayscale(original)
    r_gray = to_grayscale(reconstructed)
    orig_hf = hf_energy(o_gray)
    if orig_hf == 0:
        return 1.0
    return float(hf_energy(r_gray) / orig_hf)


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

METHODS = {
    "PointWise": downsample_pointwise,
    "Average":   downsample_average,
    "Fourier":   downsample_fourier,
}


def evaluate_dataset(data: dict,
                     resolutions: list = TARGET_RESOLUTIONS) -> dict:
    """
    For every (subset, method, resolution) triplet, compute the four quality
    metrics averaged across all sampled images.

    Returns a nested dict:
        results[subset][method][resolution] = {
            "mse": float, "psnr": float, "ssim": float, "hf_ratio": float
        }
    """
    results = {}
    for subset, images in data.items():
        results[subset] = {m: {r: [] for r in resolutions} for m in METHODS}
        for img in tqdm.tqdm(images, desc=f"Evaluating {subset}"):
            img_f = img.astype(np.float32)
            orig_gray = to_grayscale(img_f)
            orig_size = img_f.shape[0]          # assumed square

            for method_name, method_fn in METHODS.items():
                for res in resolutions:
                    ds = method_fn(img_f, res)
                    up = upsample_to_original(ds, orig_size)

                    # Use grayscale for scalar metrics
                    up_gray = to_grayscale(up)

                    mse      = compute_mse(orig_gray, up_gray)
                    psnr     = compute_psnr(orig_gray, up_gray)
                    ssim     = compute_ssim(orig_gray, up_gray)
                    hf_ratio = compute_hf_ratio(img_f, up)

                    results[subset][method_name][res].append(
                        {"mse": mse, "psnr": psnr,
                         "ssim": ssim, "hf_ratio": hf_ratio}
                    )

    # Average over sampled images
    for subset in results:
        for method in results[subset]:
            for res in results[subset][method]:
                records = results[subset][method][res]
                results[subset][method][res] = {
                    k: float(np.mean([r[k] for r in records]))
                    for k in ("mse", "psnr", "ssim", "hf_ratio")
                }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

METHOD_MARKERS = {"PointWise": "o", "Average": "s", "Fourier": "^"}
DATASET_COLORS = ["#2E86AB", "#E07B39", "#6A994E", "#9B5DE5"]  # up to 4 datasets
DATASET_LINESTYLES = ["-", "--", "-.", ":"]

METRIC_LABELS = {
    "psnr":     "PSNR (dB) ↑",
    "ssim":     "SSIM ↑",
    "mse":      "MSE ↓",
    "hf_ratio": "HF Energy Retention ↑",
}


def plot_metrics_from_csv(csv_path: str, save_dir: str = None):
    """
    Read the CSV produced by save_results_csv() and generate one figure per
    (datatype x dataset-group) combination.

    Visual encoding:
      Colour      → dataset        (primary comparison axis)
      Marker shape → method        (secondary comparison axis)
      Line style  → method         (redundant with marker, aids print/greyscale)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file written by save_results_csv().
    save_dir : str or None
        Directory to save PNG files. If None, figures are only shown and not saved.
    """
    import csv

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset":     row["dataset"],
                "datatype":    row["datatype"],
                "resolution":  int(row["resolution"]),
                "method":      row["method"],
                "psnr":        float(row["psnr"]),
                "ssim":        float(row["ssim"]),
                "mse":         float(row["mse"]),
                "hf_ratio":    float(row["hf_ratio"]),
                "num_samples": int(row["num_samples"]),
            })

    datatypes   = list(dict.fromkeys(r["datatype"]  for r in rows))
    methods     = list(dict.fromkeys(r["method"]    for r in rows))
    resolutions = sorted(set(r["resolution"] for r in rows))
    metrics     = list(METRIC_LABELS.keys())

    num_samples = rows[0]["num_samples"] if rows else "?"

    # Fixed dataset groupings — two figures per datatype
    DATASET_GROUPS = {
        "MetalSet_ViaSet":     ["MetalSet", "ViaSet"],
        "StdContact_StdMetal": ["StdContact", "StdMetal"],
    }

    # Colour per dataset within each group (same colours reused across groups
    # since the two groups never appear in the same figure)
    GROUP_COLORS = [DATASET_COLORS[0], DATASET_COLORS[1]]

    # Line style + marker are both keyed on method for redundant encoding
    method_ls = {m: DATASET_LINESTYLES[i] for i, m in enumerate(methods)}

    # ── One pair of figures per datatype ─────────────────────────────────────
    for datatype in datatypes:
        dt_rows = [r for r in rows if r["datatype"] == datatype]

        for group_name, group_datasets in DATASET_GROUPS.items():
            if not any(r["dataset"] in group_datasets for r in dt_rows):
                continue

            dataset_color = {ds: GROUP_COLORS[i]
                             for i, ds in enumerate(group_datasets)}

            fig, axes = plt.subplots(
                2, 2, figsize=(14, 11),
                constrained_layout=False,
            )
            fig.subplots_adjust(top=0.91, bottom=0.18, hspace=0.38, wspace=0.28)
            axes = axes.flatten()

            for ax, metric in zip(axes, metrics):
                for dataset in group_datasets:
                    color = dataset_color[dataset]
                    for method in methods:
                        marker = METHOD_MARKERS.get(method, "o")
                        ls     = method_ls[method]

                        values = []
                        for res in resolutions:
                            matching = [
                                r[metric] for r in dt_rows
                                if r["dataset"]     == dataset
                                and r["method"]     == method
                                and r["resolution"] == res
                            ]
                            values.append(np.mean(matching) if matching else np.nan)

                        ax.plot(
                            resolutions, values,
                            color=color, linestyle=ls, marker=marker,
                            linewidth=2, markersize=6,
                            label="_nolegend_",
                        )

                ax.set_title(METRIC_LABELS[metric], fontsize=11,
                             fontweight="bold", pad=6)
                ax.set_xlabel("Resolution (px)", fontsize=9)
                ax.set_xticks(resolutions)
                ax.set_xticklabels([str(r) for r in resolutions], fontsize=8)
                ax.grid(True, alpha=0.25, linestyle="--")

            # ── Legend 1: Dataset (colour) ─────────────────────────────────────
            dataset_handles = [
                plt.Line2D([0], [0],
                           color=dataset_color[ds], linewidth=3, label=ds)
                for ds in group_datasets
            ]
            legend_dataset = fig.legend(
                handles=dataset_handles,
                title="Dataset",
                title_fontsize=10,
                fontsize=9,
                loc="lower left",
                bbox_to_anchor=(0.04, 0.01),
                ncol=len(group_datasets),
                framealpha=0.9,
                edgecolor="#aaaaaa",
            )

            # ── Legend 2: Method (marker + line style, neutral colour) ──────────
            method_handles = [
                plt.Line2D([0], [0],
                           color="#444444",
                           linestyle=method_ls[m],
                           marker=METHOD_MARKERS.get(m, "o"),
                           linewidth=2, markersize=7, label=m)
                for m in methods
            ]
            fig.legend(
                handles=method_handles,
                title="Downsampling Method",
                title_fontsize=10,
                fontsize=9,
                loc="lower right",
                bbox_to_anchor=(0.96, 0.01),
                ncol=len(methods),
                framealpha=0.9,
                edgecolor="#aaaaaa",
            )
            fig.add_artist(legend_dataset)

            # ── Title ──────────────────────────────────────────────────────────
            pretty_group = " & ".join(group_datasets)
            fig.suptitle(
                f"Downsampling Study  |  Datatype: {datatype}  |  "
                f"Datasets: {pretty_group}  |  n = {num_samples} per subset",
                fontsize=12, fontweight="bold", y=0.97,
            )

            if save_dir is not None:
                out_path = save_dir / f"metrics_{datatype}_{group_name}.png"
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {out_path}")

            plt.show()

METRIC_DISPLAY = {
    "psnr":     ("PSNR (dB)", "higher"),
    "ssim":     ("SSIM",      "higher"),
    "mse":      ("MSE",       "lower"),
    "hf_ratio": ("HF Retain", "higher"),
}
METRICS = list(METRIC_DISPLAY.keys())


def _best_marker(values, higher_is_better: bool):
    """Return '*' for the best value in a list, '' for the rest."""
    best = max(values) if higher_is_better else min(values)
    return ["*" if v == best else " " for v in values]


def print_detailed_tables(results: dict):
    """
    Print one table per dataset-datatype (subset).

    Layout per table
    ----------------
    Columns : Resolution | Method | PSNR | SSIM | MSE | HF Retain
    Rows    : one per (resolution x method) combination
    A '*' marks the best method for each metric at each resolution.
    """
    resolutions = sorted(next(iter(
        next(iter(results.values())).values()
    )).keys())

    # Column widths
    cw = {"res": 6, "method": 10, "psnr": 10, "ssim": 8, "mse": 12, "hf": 10}
    sep = "-"

    def _header_line():
        return (f"{'Res':>{cw['res']}}  {'Method':<{cw['method']}}"
                f"  {'PSNR(dB)':>{cw['psnr']}}"
                f"  {'SSIM':>{cw['ssim']}}"
                f"  {'MSE':>{cw['mse']}}"
                f"  {'HF Retain':>{cw['hf']}}")

    header = _header_line()
    width = len(header)

    for subset in sorted(results.keys()):
        print(f"\n{'=' * width}")
        print(f"  {subset}  (n={NUM_SAMPLES})")
        print(f"{'=' * width}")
        print(header)
        print(sep * width)

        for res in resolutions:
            method_names = list(METHODS.keys())
            psnr_vals = [results[subset][m][res]["psnr"]     for m in method_names]
            ssim_vals = [results[subset][m][res]["ssim"]     for m in method_names]
            mse_vals  = [results[subset][m][res]["mse"]      for m in method_names]
            hf_vals   = [results[subset][m][res]["hf_ratio"] for m in method_names]

            psnr_mark = _best_marker(psnr_vals, higher_is_better=True)
            ssim_mark = _best_marker(ssim_vals, higher_is_better=True)
            mse_mark  = _best_marker(mse_vals,  higher_is_better=False)
            hf_mark   = _best_marker(hf_vals,   higher_is_better=True)

            for i, method in enumerate(method_names):
                res_col = str(res) if i == 0 else ""
                print(
                    f"{res_col:>{cw['res']}}  {method:<{cw['method']}}"
                    f"  {psnr_vals[i]:>{cw['psnr'] - 1}.4f}{psnr_mark[i]}"
                    f"  {ssim_vals[i]:>{cw['ssim'] - 1}.4f}{ssim_mark[i]}"
                    f"  {mse_vals[i]:>{cw['mse'] - 1}.6f}{mse_mark[i]}"
                    f"  {hf_vals[i]:>{cw['hf'] - 1}.4f}{hf_mark[i]}"
                )
            print(sep * width)

    print(f"\n* = best method for that metric at that resolution")


def save_results_csv(results: dict, save_dir: str = "resolution_study_output",
                     num_samples: int = NUM_SAMPLES):
    """
    Save all evaluation results to a single CSV file.

    Columns: subset, dataset, datatype, num_samples, resolution,
             method, psnr, ssim, mse, hf_ratio
    One row per (subset x resolution x method) combination.
    """
    import csv

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "resolution_study_results.csv"

    resolutions = sorted(next(iter(
        next(iter(results.values())).values()
    )).keys())

    fieldnames = [
        "subset", "dataset", "datatype", "num_samples",
        "resolution", "method",
        "psnr", "ssim", "mse", "hf_ratio",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for subset in sorted(results.keys()):
            parts = subset.split("-", 1)
            dataset  = parts[0]
            datatype = parts[1] if len(parts) > 1 else ""

            for res in resolutions:
                for method in METHODS:
                    m = results[subset][method][res]
                    writer.writerow({
                        "subset":      subset,
                        "dataset":     dataset,
                        "datatype":    datatype,
                        "num_samples": num_samples,
                        "resolution":  res,
                        "method":      method,
                        "psnr":        round(m["psnr"],     6),
                        "ssim":        round(m["ssim"],     6),
                        "mse":         round(m["mse"],      6),
                        "hf_ratio":    round(m["hf_ratio"], 6),
                    })

    print(f"\nCSV saved to: {csv_path}")
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # print("Loading dataset samples ...")
    # data = load_data(DATA_DICT, number_of_samples=NUM_SAMPLES)

    # print("\nRunning downsampling evaluation ...")
    # results = evaluate_dataset(data, resolutions=TARGET_RESOLUTIONS)

    save_dir = Path("resolution_study_output")

    # print_detailed_tables(results)

    # save_results_csv(results, save_dir=str(save_dir),
    #                  num_samples=NUM_SAMPLES)

    print("\nGenerating metric plots ...")
    csv_path = save_dir / "resolution_study_results.csv"
    plot_metrics_from_csv(str(csv_path), save_dir=str(save_dir))
