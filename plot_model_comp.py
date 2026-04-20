"""Grouped bar charts comparing FNO-Conv and ViT on LithoBench forwards problem."""
import matplotlib.pyplot as plt
import numpy as np

# ── Results ───────────────────────────────────────────────────────────────────
median_results_10ep = {
    "FNO-Conv": {
        "MetalSet Aerial":   0.002547,
        "MetalSet Printed": 0.018694,
        "StdMetal Aerial":   0.003829,
        "StdMetal Printed": 0.020175,
    },
    "ViT": {
        "MetalSet Aerial":   0.091979,
        "MetalSet Printed": 0.157075,
        "StdMetal Aerial":   0.091389,
        "StdMetal Printed": 0.168606,
    },
}

mean_results_10ep  = {
    "FNO-Conv": {
        "MetalSet Aerial":   0.002611,
        "MetalSet Printed": 0.019482,
        "StdMetal Aerial":   0.005267,
        "StdMetal Printed": 0.025653,
    },
    "ViT": {
        "MetalSet Aerial":   0.092307,
        "MetalSet Printed": 0.158936,
        "StdMetal Aerial":   0.093811,
        "StdMetal Printed": 0.175735,
    },
}

mean_results_doinn = {
    "DOINN": {
        "MetalSet Aerial":   0.002915,  # sqrt(8.5e-6), LithoBench Table 2 Subtask 1
        "MetalSet Printed": 0.025690,  # sqrt(6.6e-4)
        "StdMetal Aerial":   0.004243,  # sqrt(1.8e-5), LithoBench Table 2 Subtask 3
        "StdMetal Printed": 0.034641,  # sqrt(1.2e-3)
    },
}


median_results_100ep = {
    "FNO-Conv": {
        "MetalSet Aerial":   0.00215,
        "MetalSet Printed": 0.01526,
        "StdMetal Aerial":   0.00383,
        "StdMetal Printed": 0.02394,
    },
    "ViT": {
        "MetalSet Aerial":   0.09198,
        "MetalSet Printed": 0.15708,
        "StdMetal Aerial":   0.09139,
        "StdMetal Printed": 0.16861,
    },
}

mean_results_100ep = {
    "FNO-Conv": {
        "MetalSet Aerial":   0.00219,
        "MetalSet Printed": 0.01607,
        "StdMetal Aerial":   0.00534,
        "StdMetal Printed": 0.02912,
    },
    "ViT": {
        "MetalSet Aerial":   0.09231,
        "MetalSet Printed": 0.15894,
        "StdMetal Aerial":   0.09381,
        "StdMetal Printed": 0.17573,
    },
}

mean_results = {**mean_results_100ep, **mean_results_doinn}
median_results = median_results_100ep

# ── Shared settings ───────────────────────────────────────────────────────────
groups = ["MetalSet Aerial", "MetalSet Printed", "StdMetal Aerial", "StdMetal Printed"]
x = np.arange(len(groups))
colors = {"FNO-Conv": "#4C72B0", "ViT": "#DD8452", "DOINN": "#55A868"}


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


def plot_epoch_comparison(data_10ep, data_100ep, metric_label="Mean RMSE"):
    """Compare 10-epoch and 100-epoch performance for each model."""
    models = list(data_100ep.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    width = 0.35
    epoch_colors = {"10 epochs": "#A1C9F4", "100 epochs": "#4C72B0"}
    ymax = max(
        max(max(data_10ep[model][g] for g in groups), max(data_100ep[model][g] for g in groups))
        for model in models
    )
    ymax *= 1.15

    for ax, model in zip(axes, models):
        vals_10 = [data_10ep[model][g] for g in groups]
        vals_100 = [data_100ep[model][g] for g in groups]

        bars_10 = ax.bar(x - width / 2, vals_10, width, label="10 epochs", color=epoch_colors["10 epochs"])
        bars_100 = ax.bar(x + width / 2, vals_100, width, label="100 epochs", color=epoch_colors["100 epochs"])

        ax.set_title(model, fontsize=12, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax)
        ax.bar_label(bars_10, fmt="%.3f", padding=3, fontsize=8)
        ax.bar_label(bars_100, fmt="%.3f", padding=3, fontsize=8)

    axes[0].set_ylabel("RMSE", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle(f"10 vs 100 Epoch Training Comparison ({metric_label})", fontsize=13, y=0.965)
    fig.tight_layout(rect=[0, 0.14, 1, 0.935], w_pad=1.0)
    return fig


fig1, ax1 = plt.subplots(figsize=(10, 5))
plot_grouped_bar(ax1, median_results, title="FNO-Conv vs ViT — Median RMSE")
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(10, 5))
plot_grouped_bar(ax2, mean_results, title="FNO-Conv vs ViT vs DOINN — Mean RMSE")
fig2.tight_layout()

fig3 = plot_epoch_comparison(
    mean_results_10ep,
    mean_results_100ep,
    metric_label="Mean RMSE",
)

fig4 = plot_epoch_comparison(
    median_results_10ep,
    median_results_100ep,
    metric_label="Median RMSE",
)

plt.show()
