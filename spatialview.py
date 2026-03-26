import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np

from spatialstudy import METHODS



path = "/Users/noamfisher/Desktop/NF-Thesis/resolution_study_output/resolution_study_results.csv"
df = pd.read_csv(path)


print(df.columns)
METHOD_COLORS = {"PointWise": "#E74C3C", "Average": "#2ECC71", "Fourier": "#3498DB"}
DATASET_SHAPE = {"MetalSet": "o", "ViaSet": "s", "StdMetal": "d", "StdContact:": "^"}
METRIC_LABELS = {
    "psnr":     "PSNR (dB)",
    "ssim":     "SSIM",
    "mse":      "MSE",
    "hf_ratio": "HF Energy Retention",
}


def plot_metric_by_res(results_path: str,
                       metric: str,
                       save_path: str = None):
    df = pd.read_csv(results_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    datatypes = df["datatype"].unique()
    resolutions = sorted(df["resolution"].unique())
    for dt in datatypes:
        subset = df[df["datatype"] == dt]
        for method in METHODS:
            method_subset = subset[subset["method"] == method]
            for i, row in method_subset.iterrows():

                ax.plot(row["resolution"], row[metric],
                        marker=DATASET_SHAPE.get(row["dataset"], "o"),
                        label=f"{method} ({row['dataset']})",
                    color=METHOD_COLORS.get(method, "#000000"), linewidth=2)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Resolution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Resolution (px)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xticks(resolutions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


plot_metric_by_res(path, metric="psnr",)



# def plot_metrics_by_resolution(results: dict,
#                                results_df: pd.DataFrame,
#                                metric: str = "psnr",
#                                save_path: str = None):
#     """
#     One plot per parent dataset; x-axis = resolution, one line per method.
#     Averages the metric across subsets of the same parent dataset.
#     """
#     groups: dict[str, list[str]] = {}
#     for key in results:
#         dataset = key.split("-")[0]
#         datatype
#         groups.setdefault(dataset, []).append(key)

#     n = len(groups)
#     fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
#     if n == 1:
#         axes = [axes]

#     resolutions = sorted(next(iter(
#         next(iter(results.values())).values()
#     )).keys())

#     for ax, (parent, subsets) in zip(axes, groups.items()):
#         for method in METHODS:
#             values = []
#             for res in resolutions:
#                 v = np.mean([results[s][method][res][metric]
#                              for s in subsets])
#                 values.append(v)
#             ax.plot(resolutions, values,
#                     marker="o", label=method,
#                     color=METHOD_COLORS[method], linewidth=2)

#         ax.set_title(parent, fontsize=12, fontweight="bold")
#         ax.set_xlabel("Target Resolution (px)")
#         ax.set_ylabel(METRIC_LABELS[metric])
#         ax.set_xticks(resolutions)
#         ax.legend()
#         ax.grid(True, alpha=0.3)

#     fig.suptitle(f"Downsampling Study — {METRIC_LABELS[metric]}",
#                  fontsize=14, fontweight="bold", y=1.02)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()


# def plot_all_metrics(results: dict, save_dir: str = "."):
#     """Convenience wrapper: generate one figure per metric."""
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     for metric in METRIC_LABELS:
#         plot_metrics_by_resolution(
#             results, metric=metric,
#             save_path=str(save_dir / f"study_{metric}.png")
#         )


# plot_metric_by_datatype(df, metric="mse")
