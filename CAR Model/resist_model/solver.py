"""Run the PyTorch post-exposure bake model."""

from __future__ import annotations

import math

import torch

from .config import SimulationConfig
from .numerics import neumann_eigenvalues, strang_step


Fields = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_NAMES = ("acid", "quencher", "protection")


def _validate_fields(fields: Fields, config: SimulationConfig) -> None:
    if not isinstance(fields, tuple) or len(fields) != 3:
        raise ValueError("fields must be a tuple (acid, quencher, protection)")
    if any(
        not isinstance(field, torch.Tensor) or not field.is_floating_point()
        for field in fields
    ):
        raise TypeError("fields must be floating-point torch.Tensor objects")

    reference = fields[0]
    for name, field in zip(_NAMES, fields):
        if field.shape != config.grid.shape:
            raise ValueError(
                f"{name} shape {tuple(field.shape)} does not match "
                f"grid {config.grid.shape}"
            )
        if field.dtype != reference.dtype or field.device != reference.device:
            raise ValueError("fields must share one dtype and device")
        values = field.detach()
        if not torch.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} must contain finite, non-negative values")
    if (fields[2].detach() > 1.0).any():
        raise ValueError("normalized protection must lie in [0, 1]")


def _step_schedule(total_s: float, max_step_s: float) -> list[float]:
    """Return equal steps that land exactly on the requested final time."""

    if total_s == 0.0:
        return []
    count = math.ceil(total_s / max_step_s)
    return [total_s / count] * count


def simulate(
    fields: Fields,
    config: SimulationConfig = SimulationConfig(),
) -> Fields:
    """Run PEB and return acid, quencher, and protected fraction."""

    config.validate()
    _validate_fields(fields, config)

    state = fields
    steps = _step_schedule(config.peb_time_s, config.max_step_s)
    eigenvalues = neumann_eigenvalues(
        config.grid, device=state[0].device, dtype=state[0].dtype
    )

    for duration in steps:
        state = strang_step(
            state,
            duration,
            config.chemistry,
            eigenvalues,
            roundoff_factor=config.roundoff_factor,
        )
    for name, field in zip(_NAMES, state):
        values = field.detach()
        if not torch.isfinite(values).all() or (values < 0.0).any():
            raise FloatingPointError(f"{name} became non-finite or negative")
    if (state[2].detach() > 1.0 + 1e-13).any():
        raise FloatingPointError("protection exceeded its normalized upper bound")

    return state
