"""Autograd-native DCT-II implemented with PyTorch FFT operations."""

from __future__ import annotations

import torch


def _axis(tensor: torch.Tensor, dim: int, function: str) -> tuple[int, int]:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{function} expects a floating-point torch.Tensor")
    if tensor.ndim == 0:
        raise ValueError(f"{function} expects at least one dimension")
    dim %= tensor.ndim
    size = tensor.shape[dim]
    if size == 0:
        raise ValueError(f"{function} cannot transform an empty axis")
    return dim, size


def _dct_order(size: int, device: torch.device) -> torch.Tensor:
    even = torch.arange(0, size, 2, device=device)
    odd = torch.arange(1, size, 2, device=device).flip(0)
    return torch.cat((even, odd))


def _angles(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    shape = [1] * tensor.ndim
    shape[dim] = size
    k = torch.arange(size, dtype=tensor.dtype, device=tensor.device)
    return torch.pi * k.reshape(shape) / (2.0 * size)


def dct_ii(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return the unnormalized DCT-II along one axis."""

    dim, size = _axis(tensor, dim, "dct_ii")
    reordered = tensor.index_select(dim, _dct_order(size, tensor.device))
    spectrum = torch.fft.fft(reordered, dim=dim)
    theta = _angles(tensor, dim, size)
    return 2.0 * (spectrum.real * torch.cos(theta) + spectrum.imag * torch.sin(theta))


def idct_ii(coefficients: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Invert dct_ii along one axis."""

    dim, size = _axis(coefficients, dim, "idct_ii")
    half = coefficients / 2.0
    theta = _angles(coefficients, dim, size)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    flipped = half.flip([dim])
    zero = torch.zeros_like(half.narrow(dim, 0, 1))
    rotated_imaginary = torch.cat((zero, -flipped.narrow(dim, 0, size - 1)), dim=dim)

    # Reconstruct the phase-rotated complex FFT spectrum.
    spectrum = torch.complex(
        half * cos_theta - rotated_imaginary * sin_theta,
        half * sin_theta + rotated_imaginary * cos_theta,
    )
    reordered = torch.fft.ifft(spectrum, dim=dim).real

    order = _dct_order(size, coefficients.device)
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(size, device=coefficients.device)
    return reordered.index_select(dim, inverse_order)


def dct2(field: torch.Tensor) -> torch.Tensor:
    return dct_ii(dct_ii(field, dim=-1), dim=-2)


def idct2(coefficients: torch.Tensor) -> torch.Tensor:
    return idct_ii(idct_ii(coefficients, dim=-1), dim=-2)
