"""Dill-like PAG depletion adapted to chemically amplified resist exposure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ExposureResult:
    acid: torch.Tensor
    pag_remaining: torch.Tensor
    reacted_fraction: torch.Tensor


def acid_from_dose(
    dose_mj_cm2: torch.Tensor,
    *,
    pag_initial: float | torch.Tensor = 1.0,
    exposure_constant_cm2_mj: float = 0.08,
    acid_yield: float = 0.85,
) -> ExposureResult:
    """Map local energy dose to acid using PAG depletion.

    ``G_end = G0*exp(-C*E)`` and ``A0 = eta*(G0-G_end)``. Concentrations
    are normalized by a common reference concentration. ``acid_yield`` folds
    quantum yield and survival to the start of PEB into one bounded factor.
    """

    if not isinstance(dose_mj_cm2, torch.Tensor) or not dose_mj_cm2.is_floating_point():
        raise TypeError("dose must be a floating-point torch.Tensor")
    dose = dose_mj_cm2
    pag = torch.as_tensor(pag_initial, dtype=dose.dtype, device=dose.device)
    if dose.ndim != 2:
        raise ValueError(f"dose must be a 2-D field, got shape {dose.shape}")
    values = dose.detach()
    if not torch.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("dose must contain finite, non-negative values")
    pag_values = pag.detach()
    if not torch.isfinite(pag_values).all() or (pag_values < 0.0).any():
        raise ValueError("pag_initial must contain finite, non-negative values")
    if not math.isfinite(exposure_constant_cm2_mj) or exposure_constant_cm2_mj < 0.0:
        raise ValueError("exposure_constant_cm2_mj must be finite and non-negative")
    if not math.isfinite(acid_yield) or not 0.0 <= acid_yield <= 1.0:
        raise ValueError("acid_yield must lie in [0, 1]")

    try:
        pag_b = torch.broadcast_to(pag, dose.shape)
    except RuntimeError as exc:
        raise ValueError("pag_initial is not broadcastable to the dose field") from exc

    depletion = exposure_constant_cm2_mj * dose
    reacted = -torch.expm1(-depletion)
    remaining = pag_b * torch.exp(-depletion)
    acid = acid_yield * pag_b * reacted
    return ExposureResult(acid=acid, pag_remaining=remaining, reacted_fraction=reacted)
