import numpy as np
import pandas as pd

all_data = pd.read_csv(
    "/Users/noamfisher/Desktop/NF-Thesis/density_study_output/density_per_image.csv"
)

# ── Separate the two datatypes ────────────────────────────────────────────────
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

# ── Join on (dataset, filename) to guarantee correct tile pairing ─────────────
df_paired = pd.merge(df_pixelilt, df_target, on=["dataset", "filename"], how="inner")

print(f"Paired tiles: {len(df_paired)}")
print(df_paired.head())

# ── Filter near-zero Target density to avoid inf/unstable ratios ──────────────
# Threshold: 5th percentile of non-zero Target densities per dataset
MIN_TARGET_DENSITY = 1e-4   # hard floor — tiles below this are effectively empty
df_valid = df_paired[df_paired["density_target"] > MIN_TARGET_DENSITY].copy()

n_dropped = len(df_paired) - len(df_valid)
print(f"Dropped {n_dropped} tiles with near-zero Target density "
      f"({100 * n_dropped / len(df_paired):.1f}%)")

# ── Per-tile expansion coefficient ───────────────────────────────────────────
df_valid["exp_coeff"] = df_valid["density_pixelilt"] / df_valid["density_target"]

print(df_valid[["dataset", "filename", "density_target",
                "density_pixelilt", "exp_coeff"]].head(10))

# ── Aggregate: mean, std, median per dataset ──────────────────────────────────
df_avg_dataset = (
    df_valid.groupby("dataset")["exp_coeff"]
    .agg(
        n="count",
        mean_exp="mean",
        std_exp="std",
        median_exp="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )
    .reset_index()
)

print("\nPer-dataset expansion coefficient summary:")
print(df_avg_dataset.to_string(index=False))


import matplotlib.pyplot as plt
import numpy as np

DATASET_ORDER  = ["MetalSet", "ViaSet", "StdMetal", "StdContact"]
DATASET_COLORS = {
    "MetalSet":   "#2E86AB",
    "ViaSet":     "#E07B39",
    "StdMetal":   "#9B5DE5",
    "StdContact": "#6A994E",
}

def plot_expansion_coefficient(df_avg: pd.DataFrame, save_path: str = None):
    """
    Plot mean ± std of the per-tile expansion coefficient (PixelILT / Target)
    for each dataset, ordered by DATASET_ORDER.

    Parameters
    ----------
    df_avg    : DataFrame with columns dataset, n, mean_exp, std_exp, median_exp
    save_path : optional filepath string to save the PNG; None = show only
    """
    # ── Order rows by canonical dataset order ─────────────────────────────────
    order  = [ds for ds in DATASET_ORDER if ds in df_avg["dataset"].values]
    df_plt = (
        df_avg.set_index("dataset")
        .loc[order]
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_plt))

    for i, row in df_plt.iterrows():
        color = DATASET_COLORS.get(row["dataset"], "#888888")
        ax.errorbar(
            x[i], row["mean_exp"],
            yerr=row["std_exp"],
            fmt="o",
            color=color, ecolor=color,
            elinewidth=2, capsize=6, markersize=9,
            label=row["dataset"],
        )
        # Annotate mean value above the error bar
        ax.text(
            x[i],
            row["mean_exp"] + row["std_exp"] + 0.05,
            f"μ={row['mean_exp']:.2f}\nσ={row['std_exp']:.2f}\nn={int(row['n']):,}",
            ha="center", va="bottom", fontsize=8, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plt["dataset"], fontsize=11)
    ax.set_ylabel("Expansion Coefficient  (PixelILT density / Target density)", fontsize=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_title("Per-Tile ILT Expansion Coefficient  |  Mean ± Std by Dataset", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.margins(x=0.15)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)


# ── Call site ─────────────────────────────────────────────────────────────────
plot_expansion_coefficient(
    df_avg_dataset,
    save_path="/Users/noamfisher/Desktop/NF-Thesis/density_study_output/expansion_coefficient.png",
)
