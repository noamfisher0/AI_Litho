"""PyTorch Mack dissolution law and depth-averaged development surrogate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DevelopmentParameters:
    rate_min_nm_s: float = 0.01
    rate_max_nm_s: float = 20.0
    mack_a: float = 0.05
    selectivity_n: float = 3.0
    development_time_s: float = 3.0
    film_thickness_nm: float = 40.0

    def validate(self) -> None:
        if any(not math.isfinite(value) for value in vars(self).values()):
            raise ValueError("development parameters must be finite")
        if self.rate_min_nm_s < 0.0 or self.rate_max_nm_s < self.rate_min_nm_s:
            raise ValueError("need 0 <= rate_min_nm_s <= rate_max_nm_s")
        if self.mack_a <= 0.0 or self.selectivity_n <= 0.0:
            raise ValueError("mack_a and selectivity_n must be positive")
        if self.development_time_s < 0.0 or self.film_thickness_nm <= 0.0:
            raise ValueError(
                "development time must be non-negative and thickness positive"
            )


def mack_rate(
    protection: torch.Tensor,
    parameters: DevelopmentParameters = DevelopmentParameters(),
) -> torch.Tensor:
    """Dissolution rate with exact endpoints ``r(1)=r_min``, ``r(0)=r_max``."""

    parameters.validate()
    if not isinstance(protection, torch.Tensor) or not protection.is_floating_point():
        raise TypeError("protection must be a floating-point torch.Tensor")
    values = protection.detach()
    if not torch.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("protection must be finite and lie in [0, 1]")
    deprotected_power = (1.0 - protection).pow(parameters.selectivity_n)
    shape = (
        (parameters.mack_a + 1.0)
        * deprotected_power
        / (parameters.mack_a + deprotected_power)
    )
    return (
        parameters.rate_min_nm_s
        + (parameters.rate_max_nm_s - parameters.rate_min_nm_s) * shape
    )


def vertical_development(
    protection: torch.Tensor,
    parameters: DevelopmentParameters = DevelopmentParameters(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return remaining thickness and fraction for independent columns."""

    rate = mack_rate(protection, parameters)
    remaining_nm = torch.clamp_min(
        parameters.film_thickness_nm - rate * parameters.development_time_s, 0.0
    )
    return remaining_nm, remaining_nm / parameters.film_thickness_nm
