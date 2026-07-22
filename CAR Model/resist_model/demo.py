"""Run a deterministic PyTorch line-space example and write diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .config import Grid, SimulationConfig
from .development import DevelopmentParameters, mack_rate, vertical_development
from .exposure import acid_from_dose
from .solver import simulate


def _line_space_dose(grid: Grid) -> torch.Tensor:
    x = (torch.arange(grid.nx, dtype=torch.float64) + 0.5) * grid.dx_nm
    y = (torch.arange(grid.ny, dtype=torch.float64) + 0.5)[:, None] * grid.dy_nm
    phase = torch.cos(2.0 * torch.pi * x / 32.0)
    aerial = 0.5 * (1.0 + torch.tanh(2.2 * phase))
    envelope = 0.96 + 0.04 * torch.cos(2.0 * torch.pi * y / grid.ly_nm)
    return 28.0 * aerial * envelope


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(__file__).resolve().parents[1] / "output"
    output.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig()
    grid = config.grid
    dose = _line_space_dose(grid)
    exposure = acid_from_dose(dose)
    q0 = torch.full_like(dose, 0.30)
    p0 = torch.ones_like(dose)
    acid, quencher, protection = simulate((exposure.acid, q0, p0), config)

    development = DevelopmentParameters()
    rate = mack_rate(protection, development)
    thickness, remaining_fraction = vertical_development(protection, development)

    fields = [
        (dose, "Dose", "mJ cm$^{-2}$"),
        (exposure.acid, "Initial acid", "normalized"),
        (acid, "Acid after PEB", "normalized"),
        (quencher, "Quencher after PEB", "normalized"),
        (protection, "Protected fraction", "normalized"),
        (remaining_fraction, "Remaining resist", "fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.7), constrained_layout=True)
    extent = (0.0, grid.lx_nm, 0.0, grid.ly_nm)
    for axis, (field, title, label) in zip(axes.ravel(), fields):
        image = axis.imshow(
            field.detach().cpu().numpy(),
            origin="lower",
            extent=extent,
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_xlabel("x (nm)")
        axis.set_ylabel("y (nm)")
        cbar = fig.colorbar(image, ax=axis, shrink=0.82)
        cbar.set_label(label)
    fig.suptitle("Depth-averaged CAR simulation (PyTorch)", fontsize=14)
    fig.savefig(output / "demo_fields.png", dpi=180)
    plt.close(fig)

    torch.save(
        {
            "dose": dose,
            "acid_initial": exposure.acid,
            "acid_final": acid,
            "quencher_final": quencher,
            "protection_final": protection,
            "dissolution_rate_nm_s": rate,
            "remaining_thickness_nm": thickness,
            "step_count": math.ceil(config.peb_time_s / config.max_step_s),
        },
        output / "demo_state.pt",
    )
    acid_minus_q_drift = torch.mean(
        (acid - quencher) - (exposure.acid - q0)
    ).item()
    summary = "\n".join(
        [
            f"backend=pytorch_{torch.__version__}",
            f"device={acid.device}",
            f"dtype={acid.dtype}",
            f"final_time_s={config.peb_time_s:.16g}",
            f"steps={math.ceil(config.peb_time_s / config.max_step_s)}",
            f"min_acid={acid.min().item():.8e}",
            f"max_acid={acid.max().item():.8e}",
            f"min_quencher={quencher.min().item():.8e}",
            f"min_protection={protection.min().item():.8e}",
            f"max_protection={protection.max().item():.8e}",
            f"mean_acid_minus_quencher_change_with_loss={acid_minus_q_drift:.8e}",
            f"remaining_fraction_min={remaining_fraction.min().item():.8e}",
            f"remaining_fraction_max={remaining_fraction.max().item():.8e}",
        ]
    )
    (output / "demo_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
