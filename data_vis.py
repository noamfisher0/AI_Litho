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

def load_data(data_dict, data_types, number_of_samples=1000):
    data = {}
    for key, path in data_dict.items():
        for data_type in data_types:
            if data_type not in key:
                continue
        files = [p.name for p in Path(path).iterdir() if p.is_file()]
        selected_files = random.sample(files, min(number_of_samples, len(files)))
        data[key] = []
        for file in tqdm.tqdm(selected_files, desc=f"Loading {key}"):
            img = plt.imread(Path(path) / file)
            data[key].append(img)
        data[key] = np.array(data[key])
    return data


def compute_pixel_density(tiles, num_samples=1000):
    densities = {}
    samples_tiles = random.sample(list(tiles.items()), min(num_samples, len(tiles)))

    for key, image in tqdm.tqdm(samples_tiles, desc="Computing pixel densities"):
        densities[key] = np.mean(image > 0)
    return densities


def plot_density_histogram(densities, label=None):
    plt.figure(figsize=(10, 6))
    plt.hist(list(densities.values()), bins=20, alpha=0.7, label=label)
    plt.title("Pixel Density Distribution")
    plt.xlabel("Pixel Density")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()



def main() -> None:
    data = load_data(DATA_DICT, ["MetalSet-Printed"], number_of_samples=1000)

    densities = compute_pixel_density(data["MetalSet-Printed"], num_samples=1000)
    plot_density_histogram(densities, label="MetalSet-Printed")


if __name__ == "__main__":
    main()
