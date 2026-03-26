#!/usr/bin/env python

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from skimage.transform import resize


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "lithobench-main"

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
    if image.ndim == 3:
        return np.mean(image[..., :3], axis=-1)
    return image


def compute_pixel_density(images, num_samples=None):
    if len(images) == 0:
        return []

    if num_samples is None:
        sampled_images = list(images)
    else:
        sampled_images = random.sample(list(images), min(num_samples, len(images)))

    densities = []
    for image in tqdm.tqdm(sampled_images, desc="Computing pixel densities"):
        active_pixels = image > 0
        if image.ndim == 3:
            active_pixels = np.any(active_pixels, axis=-1)
        densities.append(float(np.mean(active_pixels)))

    return densities


def compute_density_by_directory(data, num_samples=None):
    density_map = {}
    for key, images in data.items():
        density_map[key] = compute_pixel_density(images, num_samples=num_samples)
    return density_map


def plot_density_histograms(density_map, bins=20):
    n_dirs = len(density_map)
    if n_dirs == 0:
        return

    n_cols = min(3, n_dirs)
    n_rows = (n_dirs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (label, densities) in enumerate(density_map.items()):
        ax = axes[idx]
        ax.hist(densities, bins=bins, alpha=0.8)
        ax.set_title(f"{label} Pixel Density")
        ax.set_xlabel("Pixel Density")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    for idx in range(n_dirs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def compute_average_tile(images, num_samples=None, downsample_size=None):
    if len(images) == 0:
        return None

    if num_samples is None:
        sampled_images = list(images)
    else:
        sampled_images = random.sample(list(images), min(num_samples, len(images)))

    accumulator = None

    for image in tqdm.tqdm(sampled_images, desc="Computing average tile"):
        gray = to_grayscale(image)

        if downsample_size is not None:
            gray = resize(
                gray,
                (downsample_size, downsample_size),
                anti_aliasing=True,
                preserve_range=True,
            )

        gray = gray.astype(np.float32)

        if accumulator is None:
            accumulator = np.zeros_like(gray, dtype=np.float64)

        accumulator += gray

    return accumulator / len(sampled_images)


def compute_heatmap_by_directory(data, num_samples=None, downsample_size=None):
    heatmap_map = {}
    for key, images in data.items():
        heatmap_map[key] = compute_average_tile(
            images,
            num_samples=num_samples,
            downsample_size=downsample_size,
        )
    return heatmap_map


def plot_heatmaps(heatmap_map, cmap="viridis"):
    n_dirs = len(heatmap_map)
    if n_dirs == 0:
        return

    n_cols = min(3, n_dirs)
    n_rows = (n_dirs + n_cols - 1) // n_cols

    valid_maps = [hm for hm in heatmap_map.values() if hm is not None]
    global_vmin = min(np.min(hm) for hm in valid_maps)
    global_vmax = max(np.max(hm) for hm in valid_maps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (label, avg_tile) in enumerate(heatmap_map.items()):
        ax = axes[idx]
        if avg_tile is None:
            ax.axis("off")
            continue

        im = ax.imshow(avg_tile, cmap=cmap, vmin=global_vmin, vmax=global_vmax)
        ax.set_title(f"{label} Average Tile")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_dirs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def compute_average_fft(images, num_samples=None, downsample_size=256):
    if len(images) == 0:
        return None

    if num_samples is None:
        sampled_images = list(images)
    else:
        sampled_images = random.sample(list(images), min(num_samples, len(images)))

    fft_accumulator = None

    for image in tqdm.tqdm(sampled_images, desc="Computing FFTs"):
        gray = to_grayscale(image)
        small_tile = resize(
            gray,
            (downsample_size, downsample_size),
            anti_aliasing=True,
            preserve_range=True,
        )

        fft = np.fft.fft2(small_tile)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted).astype(np.float64)

        if fft_accumulator is None:
            fft_accumulator = np.zeros_like(magnitude, dtype=np.float64)

        fft_accumulator += magnitude

    return fft_accumulator / len(sampled_images)


def compute_fft_by_directory(data, num_samples=None, downsample_size=256):
    fft_map = {}
    for key, images in data.items():
        fft_map[key] = compute_average_fft(
            images,
            num_samples=num_samples,
            downsample_size=downsample_size,
        )
    return fft_map


def plot_fft_spectra(fft_map, log_scale=True, cmap="inferno"):
    n_dirs = len(fft_map)
    if n_dirs == 0:
        return

    n_cols = min(3, n_dirs)
    n_rows = (n_dirs + n_cols - 1) // n_cols

    processed_maps = {}
    for key, fft_spectrum in fft_map.items():
        if fft_spectrum is None:
            processed_maps[key] = None
        else:
            processed_maps[key] = np.log(fft_spectrum + 1e-8) if log_scale else fft_spectrum

    valid_maps = [m for m in processed_maps.values() if m is not None]
    global_vmin = min(np.min(m) for m in valid_maps)
    global_vmax = max(np.max(m) for m in valid_maps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (label, spectrum) in enumerate(processed_maps.items()):
        ax = axes[idx]
        if spectrum is None:
            ax.axis("off")
            continue

        im = ax.imshow(spectrum, cmap=cmap, vmin=global_vmin, vmax=global_vmax)
        ax.set_title(f"{label} Average FFT")
        ax.axis("off")
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Log Magnitude" if log_scale else "Magnitude",
        )

    for idx in range(n_dirs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def radial_profile(fft_spectrum, num_bins=100):
    center = np.array(fft_spectrum.shape) // 2
    y, x = np.indices(fft_spectrum.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    radial_sum = np.bincount(r.ravel(), fft_spectrum.ravel())
    radial_count = np.bincount(r.ravel())
    profile = radial_sum / (radial_count + 1e-8)

    return profile[:num_bins]


def compute_radial_profiles_by_directory(fft_map, num_bins=100):
    radial_map = {}
    for key, fft_spectrum in fft_map.items():
        if fft_spectrum is None:
            radial_map[key] = None
        else:
            radial_map[key] = radial_profile(fft_spectrum, num_bins=num_bins)
    return radial_map


def plot_radial_profiles(radial_map):
    plt.figure(figsize=(9, 6))

    for label, profile in radial_map.items():
        if profile is not None:
            plt.plot(profile, label=label)

    plt.title("Radial Profiles of Average FFT Spectra")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Average Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



def load_matched_printed_target_pairs(data_dict, dataset_prefix, number_of_samples=None):
    printed_key = f"{dataset_prefix}-Printed"
    target_key = f"{dataset_prefix}-Target"

    printed_path = Path(data_dict[printed_key])
    target_path = Path(data_dict[target_key])

    printed_files = {p.name for p in printed_path.iterdir() if p.is_file()}
    target_files = {p.name for p in target_path.iterdir() if p.is_file()}

    common_files = sorted(printed_files & target_files)
    if not common_files:
        raise ValueError(f"No matching files found for {printed_key} and {target_key}")

    if number_of_samples is not None:
        common_files = random.sample(common_files, min(number_of_samples, len(common_files)))

    printed_images = []
    target_images = []

    for fname in tqdm.tqdm(common_files, desc=f"Loading matched pairs for {dataset_prefix}"):
        p = plt.imread(printed_path / fname)
        t = plt.imread(target_path / fname)

        printed_images.append(to_grayscale(p).astype(np.float32))
        target_images.append(to_grayscale(t).astype(np.float32))

    return np.stack(printed_images), np.stack(target_images), common_files


def calc_printed_target_diff(data_dict, dataset_prefix, num_samples=None):
    printed_images, target_images, _ = load_matched_printed_target_pairs(
        data_dict, dataset_prefix, number_of_samples=num_samples
    )

    diffs = np.abs(printed_images - target_images)
    avg_diff = np.mean(diffs)
    print(f"Average pixel difference for {dataset_prefix}: {avg_diff:.6f}")
    return avg_diff


def plot_printed_target_diff_curve(data_dict, dataset_prefix, num_samples=None):
    printed_images, target_images, _ = load_matched_printed_target_pairs(
        data_dict, dataset_prefix, number_of_samples=num_samples
    )

    diffs = np.abs(printed_images - target_images)
    avg_diff_per_image = np.mean(diffs, axis=(1, 2))

    plt.figure(figsize=(9, 6))
    plt.plot(avg_diff_per_image)
    plt.title(f"{dataset_prefix}: Mean Absolute Difference per Tile (Printed vs Target)")
    plt.xlabel("Sample Index")
    plt.ylabel("Mean Absolute Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_printed_target_error_heatmap(data_dict, dataset_prefix, num_samples=None):
    printed_images, target_images, _ = load_matched_printed_target_pairs(
        data_dict, dataset_prefix, number_of_samples=num_samples
    )

    diffs = np.abs(printed_images - target_images)
    avg_error_map = np.mean(diffs, axis=0)

    plt.figure(figsize=(7, 7))
    im = plt.imshow(avg_error_map, cmap="hot")
    plt.title(f"{dataset_prefix}: Average |Printed - Target|")
    plt.axis("off")
    plt.colorbar(im, label="Mean Absolute Error")
    plt.tight_layout()
    plt.show()


def main() -> None:
    selected_keys = [
        "StdContact-Printed",
        "StdContact-Resist",
        "StdContact-Target",
        "StdContact-Litho",
        "StdContact-PixelILT",
    ]

    # data = load_data(DATA_DICT, selected_keys=selected_keys, number_of_samples=2000)

    # density_map = compute_density_by_directory(data, num_samples=2000)
    # plot_density_histograms(density_map)

    # heatmap_map = compute_heatmap_by_directory(
    #     data,
    #     num_samples=2000,
    #     downsample_size=256,
    # )
    # plot_heatmaps(heatmap_map)

    # fft_map = compute_fft_by_directory(
    #     data,
    #     num_samples=1000,
    #     downsample_size=256,
    # )
    # plot_fft_spectra(fft_map)

    # radial_map = compute_radial_profiles_by_directory(fft_map, num_bins=100)
    # plot_radial_profiles(radial_map)

    dataset_prefix = "StdContact"
    plot_printed_target_diff_curve(DATA_DICT, dataset_prefix, num_samples=2000)
    plot_printed_target_error_heatmap(DATA_DICT, dataset_prefix, num_samples=2000)


if __name__ == "__main__":
    main()
