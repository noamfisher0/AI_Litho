"""PyTorch-native positive reaction-diffusion numerics for CAR PEB."""

from __future__ import annotations

import math

import torch

from .config import Grid, PEBParameters
from .dct import dct2, idct2


def neumann_eigenvalues(
    grid: Grid,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Eigenvalues of the cell-centered finite-volume Neumann Laplacian."""

    kx = torch.arange(grid.nx, dtype=dtype, device=device)
    ky = torch.arange(grid.ny, dtype=dtype, device=device)
    lam_x = 2.0 * (torch.cos(torch.pi * kx / grid.nx) - 1.0) / grid.dx_nm**2
    lam_y = 2.0 * (torch.cos(torch.pi * ky / grid.ny) - 1.0) / grid.dy_nm**2
    return lam_y[:, None] + lam_x[None, :]


def _clean_roundoff(field: torch.Tensor, factor: float) -> torch.Tensor:
    """Remove only sign errors consistent with transform roundoff."""

    minimum = float(field.detach().min().item())
    if minimum >= 0.0:
        return field
    scale = max(1.0, float(field.detach().abs().max().item()))
    tolerance = factor * torch.finfo(field.dtype).eps * scale
    if minimum < -tolerance:
        raise FloatingPointError(
            f"nonphysical negative concentration {minimum:.3e} exceeds "
            f"roundoff tolerance {tolerance:.3e}"
        )
    return field.clamp_min(0.0)


def diffuse_neumann(
    field: torch.Tensor,
    diffusivity_nm2_s: float,
    duration_s: float,
    eigenvalues: torch.Tensor,
    *,
    roundoff_factor: float = 256.0,
    enforce_nonnegative: bool = False,
) -> torch.Tensor:
    """Exact-in-time evolution of the chosen discrete diffusion operator."""

    if duration_s < 0.0 or diffusivity_nm2_s < 0.0:
        raise ValueError("duration and diffusivity must be non-negative")
    if duration_s == 0.0 or diffusivity_nm2_s == 0.0:
        return field.clone()
    if eigenvalues.shape != field.shape[-2:]:
        raise ValueError("eigenvalue shape does not match the field")
    lam = eigenvalues.to(dtype=field.dtype, device=field.device)
    decay = torch.exp(diffusivity_nm2_s * duration_s * lam)
    result = idct2(dct2(field) * decay)
    return _clean_roundoff(result, roundoff_factor) if enforce_nonnegative else result


def exact_neutralization(
    acid: torch.Tensor,
    quencher: torch.Tensor,
    duration_s: float,
    rate_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact positive flow for ``a'=q'=-rate*a*q``.

    All branches use differentiable PyTorch operations. One symmetric formula
    handles either excess species; exact equality uses the analytic limit.
    """

    if acid.shape != quencher.shape or acid.device != quencher.device:
        raise ValueError("acid and quencher must have matching shapes and devices")
    if acid.dtype != quencher.dtype or not acid.is_floating_point():
        raise ValueError("acid and quencher must have the same floating dtype")
    if duration_s < 0.0 or rate_s < 0.0:
        raise ValueError("duration and rate must be non-negative")
    if duration_s == 0.0 or rate_s == 0.0:
        return acid.clone(), quencher.clone()
    if (acid.detach() < 0.0).any() or (quencher.detach() < 0.0).any():
        raise ValueError("exact_neutralization requires non-negative inputs")

    delta = acid - quencher
    acid_rich = delta > 0.0
    excess = delta.abs()
    safe_excess = excess.clamp_min(torch.finfo(acid.dtype).tiny)
    lean = torch.minimum(acid, quencher)
    exponent = rate_s * safe_excess * duration_s
    lean_after = (
        safe_excess
        * lean
        * torch.exp(-exponent)
        / (safe_excess - lean * torch.expm1(-exponent))
    )
    rich_after = lean_after + safe_excess
    acid_unequal = torch.where(acid_rich, rich_after, lean_after)
    quencher_unequal = torch.where(acid_rich, lean_after, rich_after)

    common = 0.5 * (acid + quencher)
    equal_after = common / (1.0 + rate_s * common * duration_s)
    # The delta terms vanish in value at equality but supply the correct
    # limiting Jacobian to autograd.
    acid_equal = equal_after + 0.5 * delta
    quencher_equal = equal_after - 0.5 * delta
    equal = delta == 0.0
    return (
        torch.where(equal, acid_equal, acid_unequal),
        torch.where(equal, quencher_equal, quencher_unequal),
    )


def reaction_step_symmetric(
    acid: torch.Tensor,
    quencher: torch.Tensor,
    protection: torch.Tensor,
    duration_s: float,
    chemistry: PEBParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order palindromic composition of exact reaction subflows."""

    if duration_s < 0.0:
        raise ValueError("duration must be non-negative")
    half = 0.5 * duration_s
    loss = math.exp(-chemistry.k_loss_s * half)
    acid = acid * loss
    acid, quencher = exact_neutralization(acid, quencher, half, chemistry.k_quench_s)
    protection = protection * torch.exp(-chemistry.k_deprotection_s * acid * duration_s)
    acid, quencher = exact_neutralization(acid, quencher, half, chemistry.k_quench_s)
    acid = acid * loss
    return acid, quencher, protection


def strang_step(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    duration_s: float,
    chemistry: PEBParameters,
    eigenvalues: torch.Tensor,
    *,
    roundoff_factor: float = 256.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diffusion half-step, symmetric reaction, diffusion half-step."""

    acid, quencher, protection = state
    half = 0.5 * duration_s

    def half_diffusion(field: torch.Tensor, diffusivity: float) -> torch.Tensor:
        return diffuse_neumann(
            field,
            diffusivity,
            half,
            eigenvalues,
            roundoff_factor=roundoff_factor,
            enforce_nonnegative=True,
        )

    acid = half_diffusion(acid, chemistry.acid_diffusivity_nm2_s)
    quencher = half_diffusion(quencher, chemistry.quencher_diffusivity_nm2_s)
    acid, quencher, protection = reaction_step_symmetric(
        acid, quencher, protection, duration_s, chemistry
    )
    acid = half_diffusion(acid, chemistry.acid_diffusivity_nm2_s)
    quencher = half_diffusion(quencher, chemistry.quencher_diffusivity_nm2_s)
    return acid, quencher, protection
