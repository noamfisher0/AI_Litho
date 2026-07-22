"""Dimensionally explicit configuration for the resist model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


@dataclass(frozen=True)
class Grid:
    """Uniform cell-centered lateral grid.

    Coordinates are ``x_i=(i+1/2)*dx`` and ``y_j=(j+1/2)*dy``, with
    ``dx=lx_nm/nx`` and ``dy=ly_nm/ny``. This convention matches the DCT-II
    Neumann finite-volume Laplacian used by the solver.
    """

    nx: int = 96
    ny: int = 96
    lx_nm: float = 192.0
    ly_nm: float = 192.0

    @property
    def dx_nm(self) -> float:
        return self.lx_nm / self.nx

    @property
    def dy_nm(self) -> float:
        return self.ly_nm / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def validate(self) -> None:
        if isinstance(self.nx, bool) or not isinstance(self.nx, int) or self.nx < 2:
            raise ValueError("nx must be an integer >= 2")
        if isinstance(self.ny, bool) or not isinstance(self.ny, int) or self.ny < 2:
            raise ValueError("ny must be an integer >= 2")
        if not math.isfinite(self.lx_nm) or self.lx_nm <= 0.0:
            raise ValueError("lx_nm must be finite and positive")
        if not math.isfinite(self.ly_nm) or self.ly_nm <= 0.0:
            raise ValueError("ly_nm must be finite and positive")


@dataclass(frozen=True)
class PEBParameters:
    """Dimensionless-concentration PEB coefficients.

    Acid ``a`` and quencher ``q`` are normalized by a common reference
    concentration. Therefore ``k_quench_s`` and ``k_deprotection_s`` have units
    of 1/s in the normalized equations. Diffusivities are in nm^2/s.
    """

    acid_diffusivity_nm2_s: float = 0.5
    quencher_diffusivity_nm2_s: float = 0.1
    k_quench_s: float = 0.8
    k_loss_s: float = 0.02
    k_deprotection_s: float = 0.5

    def validate(self) -> None:
        for name, value in vars(self).items():
            _finite_nonnegative(name, value)


@dataclass(frozen=True)
class SimulationConfig:
    grid: Grid = field(default_factory=Grid)
    chemistry: PEBParameters = field(default_factory=PEBParameters)
    peb_time_s: float = 10.0
    max_step_s: float = 0.1
    roundoff_factor: float = 256.0

    def validate(self) -> None:
        self.grid.validate()
        self.chemistry.validate()
        _finite_nonnegative("peb_time_s", self.peb_time_s)
        if not math.isfinite(self.max_step_s) or self.max_step_s <= 0.0:
            raise ValueError("max_step_s must be finite and positive")
        if not math.isfinite(self.roundoff_factor) or self.roundoff_factor < 1.0:
            raise ValueError("roundoff_factor must be finite and >= 1")
