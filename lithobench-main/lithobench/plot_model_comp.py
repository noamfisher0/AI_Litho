"""Grouped bar charts comparing FNO-Conv and ViT on LithoBench forwards problem."""
import matplotlib.pyplot as plt
import numpy as np

# ── Results ───────────────────────────────────────────────────────────────────
median_results = {
    "FNO-Conv": {
        "MetalSet Aerial":   0.002547,
        "MetalSet Printed": 0.018694,
        "StdMetal Aerial":   0.003829,
        "StdMetal Printed": 0.020175,
    },
    "ViT-Small": {
        "MetalSet Aerial":   0.091979,
        "MetalSet Printed": 0.157075,
        "StdMetal Aerial":   0.091389,
        "StdMetal Printed": 0.168606,
    },
}

mean_results = {
    "FNO-Conv": {
        "MetalSet Aerial":   0.002611,
        "MetalSet Printed": 0.019482,
        "StdMetal Aerial":   0.005267,
        "StdMetal Printed": 0.025653,
    },
    "ViT-Small": {
        "MetalSet Aerial":   0.092307,
        "MetalSet Printed": 0.158936,
        "StdMetal Aerial":   0.093811,
        "StdMetal Printed": 0.175735,
    },
    "DOINN": {
        "MetalSet Aerial":   0.002915,  # sqrt(8.5e-6), LithoBench Table 2 Subtask 1
        "MetalSet Printed": 0.025690,  # sqrt(6.6e-4)
        "StdMetal Aerial":   0.004243,  # sqrt(1.8e-5), LithoBench Table 2 Subtask 3
        "StdMetal Printed": 0.034641,  # sqrt(1.2e-3)
    },
}

# ── Shared settings ───────────────────────────────────────────────────────────
groups = ["MetalSet Aerial", "MetalSet Printed", "StdMetal Aerial", "StdMetal Printed"]
x = np.arange(len(groups))
colors = {"FNO-Conv": "#4C72B0", "ViT-Small": "#DD8452", "DOINN": "#55A868"}


def plot_grouped_bar(ax, data, title):
    models = list(data.keys())
    n = len(models)
    width = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for model, offset in zip(models, offsets):
        values = [data[model][g] for g in groups]
        ax.bar(x + offset, values, width, label=model, color=colors[model])

    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


fig1, ax1 = plt.subplots(figsize=(10, 5))
plot_grouped_bar(ax1, median_results, title="FNO-Conv vs ViT-Small — Median RMSE (10 Epochs)")
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(10, 5))
plot_grouped_bar(ax2, mean_results, title="FNO-Conv vs ViT-Small vs DOINN — Mean RMSE (10 Epochs)")
fig2.tight_layout()

plt.show()
