#!/usr/bin/env python

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "lithobench-main"

DATA_DICT = {
    "MetalSet-Printed": str(DATA_ROOT / "MetalSet" / "printed"),
    "MetalSet-Resist": str(DATA_ROOT / "MetalSet" / "resist"),
    "MetalSet-Target": str(DATA_ROOT / "MetalSet" / "target"),
    "ViaSet-Printed": str(DATA_ROOT / "ViaSet" / "printed"),
    "ViaSet-Resist": str(DATA_ROOT / "ViaSet" / "resist"),
    "ViaSet-Target": str(DATA_ROOT / "ViaSet" / "target"),
    "StdContact-Printed": str(DATA_ROOT / "StdContact" / "printed"),
    "StdContact-Resist": str(DATA_ROOT / "StdContact" / "resist"),
    "StdContact-Target": str(DATA_ROOT / "StdContact" / "target"),
    "StdMetal-Printed": str(DATA_ROOT / "StdMetal" / "printed"),
    "StdMetal-Resist": str(DATA_ROOT / "StdMetal" / "resist"),
    "StdMetal-Target": str(DATA_ROOT / "StdMetal" / "target"),
}


def load_data(data_dict, selected_keys=None, number_of_samples=1000):
    if selected_keys is None:
        selected_keys = list(data_dict.keys())

    data = {}
    for key in selected_keys:
        path = Path(data_dict[key])
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")

        files = [p.name for p in path.iterdir() if p.is_file()]
        selected_files = random.sample(files, min(number_of_samples, len(files)))
        data[key] = []
        for file in tqdm.tqdm(selected_files, desc=f"Loading {key}"):
            img = plt.imread(path / file)
            data[key].append(img)
        data[key] = np.array(data[key])
    return data


def compute_pixel_density(images, num_samples=1000):
    if len(images) == 0:
        return []

    sampled_images = random.sample(list(images), min(num_samples, len(images)))
    densities = []

    for image in tqdm.tqdm(sampled_images, desc="Computing pixel densities"):
        active_pixels = image > 0
        if image.ndim == 3:
            active_pixels = np.any(active_pixels, axis=-1)
        densities.append(float(np.mean(active_pixels)))

    return densities


def compute_density_by_directory(data, num_samples=1000):
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



def main() -> None:
    selected_keys = ["MetalSet-Printed", "ViaSet-Printed"]
    data = load_data(DATA_DICT, selected_keys=selected_keys, number_of_samples=1000)
    density_map = compute_density_by_directory(data, num_samples=1000)
    plot_density_histograms(density_map)


if __name__ == "__main__":
    main()
