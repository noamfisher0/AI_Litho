#!/usr/bin/env python3
"""
Confirmed ICCAD-2013 parameters
--------------------------------
    lam    = 193    nm
    NA     = 1.35
    sigma_in  = 0.3
    sigma_out = 0.9
    pixel  = 1      nm
    N      = 2048       (simulation grid)
    Dz     = 25     nm  (defocus condition)
    n      = 1.44       (water immersion medium at 193 nm)
    KernelNum = 24

Usage
-----
    python scripts/tcc_reconstruct.py               # synthetic mask
    python scripts/tcc_reconstruct.py --nc  /path/to/LithoBenchData-256/LithoBench-MetalSet.nc
    python scripts/tcc_reconstruct.py --pt  /path/to/lso/data/00000.pt   # M_bin key
    python scripts/tcc_reconstruct.py --n-medium 1.0  # try air (should fail physics)

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Physical / numerical constants
# ---------------------------------------------------------------------------
LAM       = 193.0    # nm
NA        = 1.35
SIGMA_IN  = 0.3
SIGMA_OUT = 0.9
PIXEL     = 1.0      # nm
N_GRID    = 2048
DZ_FOCUS  = 0.0      # nm  (no defocus)
DZ_DEF    = 25.0     # nm
N_SOURCE  = 201      # source grid side (odd)
K_SIZE    = 35       # kernel spatial support (odd)
N_KEEP    = 24       # number of SOCS eigenpairs to keep

KERNEL_DIR = Path(__file__).resolve().parents[0] / "lithobench/lithosim/kernel"


# ===========================================================================
# 1.  Source discretisation
# ===========================================================================

def make_source_grid(sigma_in=SIGMA_IN, sigma_out=SIGMA_OUT, n_source=N_SOURCE):
    """Return (S, 2) array of physical source frequencies [cyc/nm] inside
    the annulus sigma_in <= rho <= sigma_out, plus area element df_s^2.

    Parameters
    ----------
    sigma_in  : inner partial coherence radius (normalised to NA/lam)
    sigma_out : outer partial coherence radius
    n_source  : grid side length (odd recommended)
    """
    # Normalised coords u in [-1, 1]^2 scaled to sigma_out
    u = np.linspace(-1.0, 1.0, n_source)   # (n_source,)
    du = u[1] - u[0]
    UX, UY = np.meshgrid(u, u, indexing='ij')   # (n_source, n_source)
    R_norm = np.sqrt(UX**2 + UY**2)

    annulus = (R_norm >= sigma_in) & (R_norm <= sigma_out)
    # Physical source freq in cyc/nm
    pupil_cutoff = NA / LAM
    fs_x = UX[annulus] * pupil_cutoff
    fs_y = UY[annulus] * pupil_cutoff
    fs   = np.stack([fs_x, fs_y], axis=1)   # (S, 2)

    # Area element in physical freq units
    df_s = du * pupil_cutoff
    area_element = df_s**2

    return fs, area_element


# ===========================================================================
# 2.  Kernel frequency grid
# ===========================================================================

def make_kernel_freq_grid():
    """Return (K, 2) array of kernel frequencies [cyc/nm], K = 35*35 = 1225.

    Convention (from _kernelMult in exact.py):
        kernel[r, c]  <->  freq  ((r-17)*df, (c-17)*df)
    where  df = 1 / (N_GRID * PIXEL).
    """
    df = 1.0 / (N_GRID * PIXEL)
    half = K_SIZE // 2                         # 17
    idx  = np.arange(K_SIZE) - half            # -17..+17
    FX, FY = np.meshgrid(idx * df, idx * df, indexing='ij')   # (35, 35) cyc/nm
    fk = np.stack([FX.ravel(), FY.ravel()], axis=1)           # (1225, 2)
    return fk, df


# ===========================================================================
# 3.  Pupil function
# ===========================================================================

def pupil_mask(f_xy):
    """Boolean pupil mask for an array of freqs f_xy (M, 2) in cyc/nm."""
    pupil_cutoff = NA / LAM
    return (f_xy[:, 0]**2 + f_xy[:, 1]**2) <= pupil_cutoff**2 + 1e-30


def pupil_phase_defocus(f_xy, Dz, n_medium):
    """Defocus phase phi(f) = 2*pi/lam * Dz * sqrt(n^2 - (lam*|f|)^2).

    Returns phase array (M,), zero outside the pupil (magnitude is 0 there
    anyway so the value doesn't matter).

    Note: the global constant term (value at f=0) is a common phase factor and
    cancels in the TCC — it is NOT subtracted here.  This keeps the formula
    identical to the classical exit-pupil wavefront.
    """
    lam_f_sq = LAM**2 * (f_xy[:, 0]**2 + f_xy[:, 1]**2)
    inside = n_medium**2 - lam_f_sq
    # Clamp to zero outside pupil to avoid sqrt of negative
    inside_safe = np.maximum(inside, 0.0)
    return (2 * np.pi / LAM) * Dz * np.sqrt(inside_safe)


# ===========================================================================
# 4.  TCC assembly
# ===========================================================================

def assemble_tcc(fs, area_element, fk, Dz=0.0, n_medium=1.44, batch_size=2000):
    """Assemble the TCC matrix (K×K) by accumulating rank-1 source updates.

    TCC[k, l] = sum_s  area_element * P(f_s + f_k) * P*(f_s + f_l)

    For the defocus pupil:
        P_def(f) = exp(i * phi(f))  inside the pupil,  0 outside

    Parameters
    ----------
    fs           : (S, 2) source points in cyc/nm
    area_element : scalar area element in (cyc/nm)^2
    fk           : (K, 2) kernel freq points in cyc/nm
    Dz           : defocus in nm (0 = focus)
    n_medium     : immersion medium index
    batch_size   : number of source points per batch (memory control)

    Returns
    -------
    TCC : (K, K) complex128 Hermitian matrix
    """
    K = fk.shape[0]   # 1225
    S = fs.shape[0]

    TCC = np.zeros((K, K), dtype=np.complex128)

    # Precompute defocus phase at all kernel freqs
    phi_fk = pupil_phase_defocus(fk, Dz, n_medium)   # (K,)
    inside_fk = pupil_mask(fk)                        # (K,) bool

    for b_start in range(0, S, batch_size):
        b_end  = min(b_start + batch_size, S)
        fs_b   = fs[b_start:b_end]                    # (B, 2)
        B      = fs_b.shape[0]

        # Shifted freqs for each (source, kernel) pair: (B, K, 2)
        f_shift = fs_b[:, None, :] + fk[None, :, :]  # (B, K, 2)

        # Flatten to (B*K, 2) for vectorised pupil check
        f_shift_flat = f_shift.reshape(-1, 2)
        inside_shift = pupil_mask(f_shift_flat).reshape(B, K)  # (B, K)

        if Dz == 0.0:
            # Focus: P(f) = real binary
            P_shifted = inside_shift.astype(np.complex128)    # (B, K)
        else:
            # Defocus: P(f) = exp(i * phi(f_s + f_k)) if inside pupil
            phi_shift = pupil_phase_defocus(f_shift_flat, Dz, n_medium)  # (B*K,)
            phi_shift = phi_shift.reshape(B, K)
            P_shifted = inside_shift * np.exp(1j * phi_shift)   # (B, K)

        # TCC += P_shifted^T @ P_shifted.conj()
        # (K, B) @ (B, K)  ->  TCC[k, l] = sum_b P[b,k] * conj(P[b,l])
        TCC += P_shifted.T @ P_shifted.conj()

    TCC *= area_element
    return TCC


# ===========================================================================
# 5.  Eigendecomposition → SOCS eigenpairs
# ===========================================================================

def tcc_to_socs(TCC, n_keep=N_KEEP):
    """Eigendecompose TCC (Hermitian) → top-n_keep (eigenvalue, eigenvector) pairs.

    Returns
    -------
    eigenvalues  : (n_keep,) real, sorted descending
    eigenvectors : (n_keep, K) complex  (each row is one eigenvector)
    """
    # eigh returns eigenvalues in ascending order
    vals, vecs = np.linalg.eigh(TCC)     # vecs: columns are eigenvectors

    # Sort descending
    idx      = np.argsort(vals)[::-1]
    vals     = vals[idx]
    vecs     = vecs[:, idx]

    # Keep only positive eigenvalues up to n_keep
    eigenvalues  = vals[:n_keep]
    eigenvectors = vecs[:, :n_keep].T     # (n_keep, K) — row = eigenvector

    return eigenvalues, eigenvectors


# ===========================================================================
# 6.  Normalisation to match shipped kernels
# ===========================================================================

def normalise_to_shipped(sigma_recon, h_recon, sigma_shipped, h_shipped):
    """Scale reconstructed eigenvalues so that sum(sigma_k) matches shipped.

    The aerial image contribution of each mode is:
        sigma_k * |h_k ⊛ mask|^2

    The eigenvectors from eigh are unit-normalised (||h_k||_2 = 1 over the
    1225-point grid) and must stay that way.  The sole free parameter is a
    global scalar alpha applied only to the eigenvalues:

        alpha = sum(sigma_shipped) / sum(sigma_recon)
        sigma_scaled = alpha * sigma_recon
        h_scaled     = h_recon          (unchanged)

    This scales the aerial image by exactly alpha.  Applying sqrt(alpha) to h
    instead would scale sigma * |h|^2 by alpha * alpha = alpha^2 — wrong.

    Returns
    -------
    sigma_scaled : (n_keep,) eigenvalues rescaled to match shipped total power
    h_recon      : (n_keep, 35, 35) eigenvectors, unchanged
    """
    total_recon   = sigma_recon.sum()
    total_shipped = sigma_shipped.sum()
    alpha         = total_shipped / total_recon if total_recon > 0 else 1.0

    return alpha * sigma_recon, h_recon


# ===========================================================================
# 7.  SOCS aerial image computation (numpy, mirrors exact.py logic)
# ===========================================================================

def socs_aerial(mask_2d, kernels_35, scales):
    """Compute aerial image from a 2D mask using SOCS eigenpairs.

    Mirrors the exact.py logic:
        1. cmask_fft = fft2(mask, norm="forward")          [N×N complex]
        2. For each k: tmp_k = ifft2(kernel_k_padded * cmask_fft)
        3. I = sum_k  sigma_k * |tmp_k|^2

    Parameters
    ----------
    mask_2d     : (H, W) float array, H = W = 2048
    kernels_35  : (n_keep, 35, 35) complex array — freq-domain eigenfunctions
    scales      : (n_keep,) real array — eigenvalues

    Returns
    -------
    aerial : (H, W) float array
    """
    H, W       = mask_2d.shape
    n_keep     = kernels_35.shape[0]
    half       = K_SIZE // 2   # 17

    cmask_fft  = np.fft.fft2(mask_2d.astype(np.complex128), norm="forward")  # (H, W)
    aerial     = np.zeros((H, W), dtype=np.float64)

    for k in range(n_keep):
        h_k    = kernels_35[k]   # (35, 35)
        # Place 35×35 kernel into corners of full H×W freq grid (mirrors _kernelMult)
        K_full = np.zeros((H, W), dtype=np.complex128)
        K_full[:half+1,  :half+1 ] = h_k[-(half+1):, -(half+1):]
        K_full[:half+1,  -half:  ] = h_k[-(half+1):,  :half    ]
        K_full[-half:,   :half+1 ] = h_k[ :half,     -(half+1):]
        K_full[-half:,   -half:  ] = h_k[ :half,      :half    ]

        tmp_k  = np.fft.ifft2(K_full * cmask_fft, norm="forward")   # (H, W)
        aerial += scales[k] * np.abs(tmp_k)**2

    return aerial.real


# ===========================================================================
# 8.  Mask loading
# ===========================================================================

def load_mask_nc(nc_path, sample_idx=0):
    """Load pixelILT mask and litho ground truth from a LithoBench NetCDF file.

    Returns
    -------
    mask  : (H, W) float32 — input mask (pixelILT)
    litho : (H, W) float32 or None — ground-truth aerial image, if present
    """
    try:
        import netCDF4 as nc4
    except ImportError:
        sys.exit("ERROR: netCDF4 not installed — pip install netCDF4")

    ds = nc4.Dataset(nc_path, "r")

    def _load_var(name):
        if name not in ds.variables:
            return None
        v = np.array(ds.variables[name][sample_idx])
        return v[0].astype(np.float32) if v.ndim == 3 else v.astype(np.float32)

    mask  = _load_var("pixelILT")
    litho = _load_var("litho")
    ds.close()

    if mask is None:
        sys.exit(f"ERROR: 'pixelILT' variable not found in {nc_path}")
    return mask, litho


def load_mask_pt(pt_path):
    """Load a binary mask from an LSO .pt file (key: 'M_bin')."""
    d    = torch.load(pt_path, map_location="cpu", weights_only=False)
    mask = d["M_bin"]
    if mask.ndim == 3:
        mask = mask[0]
    return mask.numpy().astype(np.float32)


def make_synthetic_mask(H=2048, W=2048, period=20):
    """Synthetic binary mask: horizontal lines of period `period` pixels."""
    mask = np.zeros((H, W), dtype=np.float32)
    for row in range(0, H, period):
        mask[row:row + period//2, :] = 1.0
    # Add a few vertical lines too
    for col in range(0, W, 47):
        mask[:, col:col + 10] = 1.0
    return mask


def save_aerial_comparison_figure(output_path, gt, recon, shipped=None):
    """Save a side-by-side aerial image comparison figure."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("    Matplotlib is not installed; skipping comparison figure.")
        return

    if shipped is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        panels = [
            (gt, "GT aerial image", "viridis"),
            (recon, "Reconstructed aerial image", "viridis"),
            (np.abs(gt - recon), "Absolute error", "magma"),
        ]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.ravel()
        panels = [
            (gt, "GT aerial image", "viridis"),
            (recon, "Reconstructed aerial image", "viridis"),
            (shipped, "Shipped aerial image", "viridis"),
            (np.abs(gt - recon), "|GT - reconstructed|", "magma"),
        ]

    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("ICCAD 2013 aerial image comparison", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved comparison figure to {output_path}")


# ===========================================================================
# 9.  Main
# ===========================================================================

def focus_eigenvalues(sigma_in, sigma_out, n_source=101, batch_size=2000):
    """Build focus TCC and return the top-24 normalised eigenvalues.

    Normalises so that sum(sigma_recon) = 1 (shape only — no shipped reference
    needed inside the sweep inner loop).

    Parameters
    ----------
    sigma_in, sigma_out : annular source radii (normalised to NA/lam)
    n_source            : source grid side (101 is fast enough for sweeps)
    batch_size          : source batching for _assemble_tcc

    Returns
    -------
    eigenvalues : (N_KEEP,) real array, sorted descending, sum = 1
    """
    fs, area_el = make_source_grid(sigma_in, sigma_out, n_source)
    fk, _       = make_kernel_freq_grid()
    TCC         = assemble_tcc(fs, area_el, fk, Dz=0.0, n_medium=1.44,
                               batch_size=batch_size)
    vals, _     = tcc_to_socs(TCC, N_KEEP)
    vals        = np.maximum(vals, 0.0)          # clip tiny negatives from numerics
    total       = vals.sum()
    return vals / total if total > 0 else vals


def spectrum_shape_error(sigma_recon_norm, sigma_shipped_norm):
    """Sum of squared log-ratios over all 24 modes.

        E = sum_k  (log(sigma_shipped[k] / sigma_recon[k]))^2

    Both inputs must be normalised so that sum = 1.
    Modes where either value is <= 0 are skipped.
    """
    safe = (sigma_recon_norm > 0) & (sigma_shipped_norm > 0)
    log_ratios = np.log(sigma_shipped_norm[safe] / sigma_recon_norm[safe])
    return float(np.sum(log_ratios**2))


def run_sweep(s_shipped_norm):
    """Two-stage sweep over (sigma_in, sigma_out).

    Stage 1: sweep sigma_in ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}, sigma_out=0.9 fixed.
    Stage 2: sweep sigma_out ∈ {0.7, 0.8, 0.9, 1.0}, sigma_in = best from stage 1.

    Parameters
    ----------
    s_shipped_norm : (24,) shipped eigenvalues normalised to sum=1, sorted descending.

    Prints a results table and returns the best (sigma_in, sigma_out) pair.
    """
    import time

    shipped_ratio_01 = s_shipped_norm[0] / s_shipped_norm[1]   # target σ₀/σ₁

    def _report_row(si, so, evals_norm, error, elapsed):
        ratio_01  = evals_norm[0] / evals_norm[1] if evals_norm[1] > 0 else np.nan
        top5_rat  = [f"{s_shipped_norm[k]/evals_norm[k]:.3f}" if evals_norm[k] > 0
                     else "inf" for k in range(5)]
        print(f"  si={si:.1f}  so={so:.2f}  err={error:7.4f}  "
              f"σ₀/σ₁={ratio_01:.3f}  "
              f"top-5 ratios=[{', '.join(top5_rat)}]  "
              f"({elapsed:.1f}s)")

    # ── Stage 1: sweep sigma_in ───────────────────────────────────────────────
    si_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    so_fixed      = 0.9
    print(f"\n{'─'*80}")
    print(f"Stage 1 — sweep sigma_in, sigma_out={so_fixed} fixed  (source grid 101×101)")
    print(f"  Target σ₀/σ₁ = {shipped_ratio_01:.4f}")
    print(f"{'─'*80}")

    stage1_results = []
    for si in si_candidates:
        t0   = time.time()
        eigs = focus_eigenvalues(si, so_fixed, n_source=101)
        err  = spectrum_shape_error(eigs, s_shipped_norm)
        elapsed = time.time() - t0
        stage1_results.append((err, si, so_fixed, eigs))
        _report_row(si, so_fixed, eigs, err, elapsed)

    best1         = min(stage1_results, key=lambda x: x[0])
    best_si_stage1 = best1[1]
    print(f"\n  Best sigma_in from stage 1: {best_si_stage1}  "
          f"(error={best1[0]:.4f})")

    # ── Stage 2: sweep sigma_out with best sigma_in ───────────────────────────
    so_candidates = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    print(f"\n{'─'*80}")
    print(f"Stage 2 — sweep sigma_out, sigma_in={best_si_stage1} fixed  (source grid 101×101)")
    print(f"{'─'*80}")

    stage2_results = []
    for so in so_candidates:
        t0   = time.time()
        eigs = focus_eigenvalues(best_si_stage1, so, n_source=101)
        err  = spectrum_shape_error(eigs, s_shipped_norm)
        elapsed = time.time() - t0
        stage2_results.append((err, best_si_stage1, so, eigs))
        _report_row(best_si_stage1, so, eigs, err, elapsed)

    best2 = min(stage2_results, key=lambda x: x[0])
    print(f"\n  Best (sigma_in, sigma_out) from stage 2: "
          f"({best2[1]}, {best2[2]})  (error={best2[0]:.4f})")

    # ── Fine sweep around the best stage-2 result ─────────────────────────────
    best_si_final = best2[1]
    best_so_final = best2[2]

    # Also try refining sigma_in around best_si_stage1
    si_fine = sorted(set([
        round(best_si_stage1 + d, 2)
        for d in [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
        if 0.0 <= best_si_stage1 + d < best_so_final
    ]))
    if si_fine:
        print(f"\n{'─'*80}")
        print(f"Stage 3 — fine sigma_in sweep around {best_si_stage1}, "
              f"sigma_out={best_so_final}")
        print(f"{'─'*80}")
        stage3_results = list(stage2_results)   # include current best
        for si in si_fine:
            t0   = time.time()
            eigs = focus_eigenvalues(si, best_so_final, n_source=101)
            err  = spectrum_shape_error(eigs, s_shipped_norm)
            elapsed = time.time() - t0
            stage3_results.append((err, si, best_so_final, eigs))
            _report_row(si, best_so_final, eigs, err, elapsed)
        best3 = min(stage3_results, key=lambda x: x[0])
        best_si_final = best3[1]
        best_so_final = best3[2]
        print(f"\n  Best after fine sweep: "
              f"sigma_in={best_si_final}, sigma_out={best_so_final}  "
              f"(error={best3[0]:.4f})")

    # ── Re-run best config at full 201×201 for final report ──────────────────
    print(f"\n{'─'*80}")
    print(f"Final — re-running best config at 201×201 source grid ...")
    print(f"{'─'*80}")
    t0        = time.time()
    eigs_full = focus_eigenvalues(best_si_final, best_so_final, n_source=201)
    err_full  = spectrum_shape_error(eigs_full, s_shipped_norm)
    elapsed   = time.time() - t0
    _report_row(best_si_final, best_so_final, eigs_full, err_full, elapsed)

    print(f"\n{'═'*80}")
    print(f"SWEEP RESULT")
    print(f"  Best sigma_in  = {best_si_final}")
    print(f"  Best sigma_out = {best_so_final}")
    print(f"  Spectrum error = {err_full:.6f}  (at 201×201)")
    print(f"  Target  σ₀/σ₁ = {shipped_ratio_01:.4f}")
    eig01 = eigs_full[0] / eigs_full[1] if eigs_full[1] > 0 else np.nan
    print(f"  Recon   σ₀/σ₁ = {eig01:.4f}")
    print(f"\n  Full eigenvalue comparison (shipped vs recon, normalised):")
    print(f"  {'k':>3}  {'shipped':>10}  {'recon':>10}  {'ratio':>8}")
    for k in range(N_KEEP):
        r = s_shipped_norm[k] / eigs_full[k] if eigs_full[k] > 0 else np.nan
        print(f"  {k:>3}  {s_shipped_norm[k]:>10.6f}  {eigs_full[k]:>10.6f}  {r:>8.4f}")
    print(f"{'═'*80}")

    return best_si_final, best_so_final


# ===========================================================================
# 10.  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="TCC reconstruction + verification")
    parser.add_argument("--nc",         default=None,
                        help="Path to LithoBench NetCDF file (pixelILT + litho variables)")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index to load from the NetCDF file (default 0)")
    parser.add_argument("--pt",         default=None,
                        help="Path to LSO .pt result file for mask (M_bin key)")
    parser.add_argument("--n-medium",   type=float, default=1.44,
                        help="Immersion medium refractive index (default 1.44 = water at 193nm)")
    parser.add_argument("--dz",         type=float, default=DZ_DEF,
                        help=f"Defocus distance in nm (default {DZ_DEF} nm)")
    parser.add_argument("--n-source",   type=int,   default=N_SOURCE,
                        help=f"Source grid side length (default {N_SOURCE})")
    parser.add_argument("--out-dir",    default=None,
                        help="Directory to save reconstructed kernels (.pt) and summary (.txt). "
                             "Created if it does not exist.")
    parser.add_argument("--no-verify",  action="store_true",
                        help="Skip aerial image verification (eigenvalue report only)")
    args = parser.parse_args()

    n_med = args.n_medium
    Dz    = args.dz

    if NA > n_med:
        print(f"WARNING: NA={NA} > n_medium={n_med} — valid only for immersion lithography.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shipped kernels (needed for sweep target and final comparison) ───
    print("=" * 70)
    print("TCC Reconstruction — ICCAD-2013 parameters")
    print(f"  lambda={LAM} nm,  NA={NA},  pixel={PIXEL} nm,  N={N_GRID}")
    print(f"  n_medium={n_med},  Dz(defocus)={Dz} nm")
    print(f"  sigma_in={SIGMA_IN}, sigma_out={SIGMA_OUT} (fixed source annulus)")
    print("=" * 70)

    print("\n[1] Loading shipped kernels ...")
    k_focus   = torch.load(KERNEL_DIR/"kernels/focus.pt",   map_location="cpu",
                           weights_only=False).permute(2, 0, 1).numpy()   # (24,35,35)
    s_focus   = torch.load(KERNEL_DIR/"scales/focus.pt",    map_location="cpu",
                           weights_only=False).numpy()                     # (24,)
    k_defocus = torch.load(KERNEL_DIR/"kernels/defocus.pt", map_location="cpu",
                           weights_only=False).permute(2, 0, 1).numpy()
    s_defocus = torch.load(KERNEL_DIR/"scales/defocus.pt",  map_location="cpu",
                           weights_only=False).numpy()

    sidx_f = np.argsort(s_focus)[::-1]
    sidx_d = np.argsort(s_defocus)[::-1]
    print(f"    Focus   scales (top-5): {s_focus[sidx_f[:5]]}")
    print(f"    Focus   total power:    {s_focus.sum():.4f}")
    print(f"    Defocus total power:    {s_defocus.sum():.4f}")

    # ── Fixed ICCAD-2013 source geometry ─────────────────────────────────────
    sigma_in_use  = SIGMA_IN
    sigma_out_use = SIGMA_OUT

    print(f"\n  Using sigma_in={sigma_in_use},  sigma_out={sigma_out_use}")

    # ── Kernel frequency grid ─────────────────────────────────────────────────
    print("\n[2] Building kernel frequency grid ...")
    fk, df = make_kernel_freq_grid()
    print(f"    df = {df:.6f} cyc/nm,  range ±{17*df:.6f} cyc/nm")
    print(f"    NA/lambda = {NA/LAM:.6f} cyc/nm  ({NA/LAM/df:.2f} bins)")

    # ── Source grid ───────────────────────────────────────────────────────────
    print(f"\n[3] Building source grid ({args.n_source}×{args.n_source}) ...")
    fs, area_el = make_source_grid(sigma_in_use, sigma_out_use, n_source=args.n_source)
    print(f"    Annular source: {fs.shape[0]} points,  area element={area_el:.4e} (cyc/nm)^2")

    # ── Assemble TCC: focus ───────────────────────────────────────────────────
    print("\n[4] Assembling focus TCC (Dz=0) ...")
    TCC_focus = assemble_tcc(fs, area_el, fk, Dz=0.0, n_medium=n_med)
    herm_err  = float(np.max(np.abs(TCC_focus - TCC_focus.conj().T)))
    print(f"    Hermitian error: {herm_err:.2e}")

    sigma_f_recon, h_f_recon = tcc_to_socs(TCC_focus, N_KEEP)
    sigma_f_norm, h_f_norm   = normalise_to_shipped(
        sigma_f_recon, h_f_recon.reshape(N_KEEP, K_SIZE, K_SIZE),
        s_focus[sidx_f], k_focus[sidx_f],
    )
    h_f_norm = h_f_norm.reshape(N_KEEP, K_SIZE, K_SIZE)

    print(f"    Normalisation factor (shipped/recon): "
          f"{s_focus[sidx_f].sum() / sigma_f_recon.sum():.6e}")
    print(f"    After normalisation, total power: {sigma_f_norm.sum():.4f}")
    print(f"\n    Per-mode comparison (shipped vs recon, focus):")
    print(f"    {'k':>3}  {'shipped':>10}  {'recon':>10}  {'ratio':>8}")
    for i in range(N_KEEP):
        r = s_focus[sidx_f[i]] / sigma_f_norm[i] if sigma_f_norm[i] > 0 else float("nan")
        print(f"    {i:>3}  {s_focus[sidx_f[i]]:>10.4f}  {sigma_f_norm[i]:>10.4f}  {r:>8.4f}")

    # ── Assemble TCC: defocus ─────────────────────────────────────────────────
    print(f"\n[5] Assembling defocus TCC (Dz={Dz} nm, n={n_med}) ...")
    TCC_defocus = assemble_tcc(fs, area_el, fk, Dz=Dz, n_medium=n_med)
    sigma_d_recon, h_d_recon = tcc_to_socs(TCC_defocus, N_KEEP)
    sigma_d_norm, h_d_norm   = normalise_to_shipped(
        sigma_d_recon, h_d_recon.reshape(N_KEEP, K_SIZE, K_SIZE),
        s_defocus[sidx_d], k_defocus[sidx_d],
    )
    h_d_norm = h_d_norm.reshape(N_KEEP, K_SIZE, K_SIZE)
    print(f"    Shipped vs recon defocus (top-5):")
    for i in range(5):
        r = s_defocus[sidx_d[i]] / sigma_d_norm[i] if sigma_d_norm[i] > 0 else float("nan")
        print(f"      k={i}: {s_defocus[sidx_d[i]]:.4f} / {sigma_d_norm[i]:.4f} = {r:.4f}")

    # ── Save reconstructed kernels ────────────────────────────────────────────
    if out_dir is not None:
        print(f"\n[6] Saving reconstructed kernels to {out_dir} ...")
        # Save in the same layout as shipped: (35, 35, 24) = permute(1,2,0)
        torch.save(
            torch.from_numpy(h_f_norm).permute(1, 2, 0),
            out_dir / "recon_kernels_focus.pt",
        )
        torch.save(torch.from_numpy(sigma_f_norm.astype(np.float32)),
                   out_dir / "recon_scales_focus.pt")
        torch.save(
            torch.from_numpy(h_d_norm).permute(1, 2, 0),
            out_dir / "recon_kernels_defocus.pt",
        )
        torch.save(torch.from_numpy(sigma_d_norm.astype(np.float32)),
                   out_dir / "recon_scales_defocus.pt")
        print(f"    Saved recon_kernels_{{focus,defocus}}.pt and recon_scales_{{focus,defocus}}.pt")

    # ── Early exit if --no-verify ─────────────────────────────────────────────
    if args.no_verify:
        print("\n[--no-verify] Skipping aerial image verification.")
        return

    # ── Load test mask ────────────────────────────────────────────────────────
    print(f"\n[7] Loading test mask ...")
    litho_gt = None   # ground-truth aerial from NetCDF, if available
    if args.nc:
        mask, litho_gt = load_mask_nc(args.nc, sample_idx=args.sample_idx)
        print(f"    Loaded from NetCDF (sample {args.sample_idx}):  mask {mask.shape},  "
              f"litho_gt {'present' if litho_gt is not None else 'absent'}")
    elif args.pt:
        mask = load_mask_pt(args.pt)
        print(f"    Loaded from .pt: {args.pt}  shape={mask.shape}")
    else:
        mask = make_synthetic_mask(N_GRID, N_GRID)
        print(f"    Using synthetic binary mask  shape={mask.shape}")

    # Upsample to 2048×2048 if needed (e.g. 256×256 pre-downsampled input)
    if mask.shape != (N_GRID, N_GRID):
        import torch.nn.functional as F_torch
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_t = F_torch.interpolate(mask_t, size=(N_GRID, N_GRID),
                                     mode="bilinear", align_corners=False)
        mask   = mask_t.squeeze().numpy()
        print(f"    Bilinear upsampled → ({N_GRID}, {N_GRID})")

    # ── Aerial: shipped focus ─────────────────────────────────────────────────
    print("\n[8] Computing aerial images ...")
    aerial_shipped = socs_aerial(mask, k_focus[sidx_f], s_focus[sidx_f])
    print(f"    shipped:      mean={aerial_shipped.mean():.5f},  "
          f"min={aerial_shipped.min():.5f},  max={aerial_shipped.max():.5f}")

    # ── Aerial: reconstructed focus ───────────────────────────────────────────
    aerial_recon = socs_aerial(mask, h_f_norm, sigma_f_norm)
    print(f"    recon:        mean={aerial_recon.mean():.5f},  "
          f"min={aerial_recon.min():.5f},  max={aerial_recon.max():.5f}")

    # ── Aerial: NetCDF ground truth (LithoSim) ────────────────────────────────
    if litho_gt is not None:
        if litho_gt.shape != (N_GRID, N_GRID):
            import torch.nn.functional as F_torch
            lg_t    = torch.from_numpy(litho_gt).unsqueeze(0).unsqueeze(0).float()
            lg_t    = F_torch.interpolate(lg_t, size=(N_GRID, N_GRID),
                                          mode="bilinear", align_corners=False)
            litho_gt = lg_t.squeeze().numpy()
        print(f"    litho_gt:     mean={litho_gt.mean():.5f},  "
              f"min={litho_gt.min():.5f},  max={litho_gt.max():.5f}")

    # ── Defocus aerials ───────────────────────────────────────────────────────
    aerial_def_shipped = socs_aerial(mask, k_defocus[sidx_d], s_defocus[sidx_d])
    aerial_def_recon   = socs_aerial(mask, h_d_norm, sigma_d_norm)

    # ── Metrics helper ────────────────────────────────────────────────────────
    def metrics(ref, pred, label):
        diff   = ref - pred
        rms    = float(np.sqrt((diff**2).mean()))
        mae    = float(np.abs(diff).mean())
        maxae  = float(np.abs(diff).max())
        rel    = float((np.abs(diff) / (np.abs(ref) + 1e-12)).mean())
        corr   = float(np.corrcoef(ref.ravel(), pred.ravel())[0, 1])
        scale  = float(ref.mean() / (pred.mean() + 1e-20))
        rms_sc = float(np.sqrt(((ref - pred * scale)**2).mean()))
        print(f"\n    {label}")
        print(f"      RMS error:            {rms:.6f}")
        print(f"      MAE:                  {mae:.6f}")
        print(f"      Max abs error:        {maxae:.6f}")
        print(f"      Mean relative error:  {rel:.4%}")
        print(f"      Pearson correlation:  {corr:.8f}")
        print(f"      Scale factor (ref/pred): {scale:.6f}")
        print(f"      RMS after scale corr: {rms_sc:.6f}")
        return dict(rms=rms, mae=mae, maxae=maxae, rel=rel, corr=corr,
                    scale=scale, rms_sc=rms_sc)

    print("\n[9] VERIFICATION METRICS")
    m_shipped_recon = metrics(aerial_shipped, aerial_recon,
                              "Focus: shipped kernels vs recon kernels (Hopkins)")
    if litho_gt is not None:
        m_gt_shipped = metrics(litho_gt, aerial_shipped,
                               "Focus: LithoBench GT (LithoSim) vs shipped kernels [sanity]")
        m_gt_recon   = metrics(litho_gt, aerial_recon,
                               "Focus: LithoBench GT (LithoSim) vs recon kernels (Hopkins)")
        fig_path = (out_dir / "aerial_comparison.png") if out_dir is not None else Path.cwd() / "aerial_comparison.png"
        save_aerial_comparison_figure(fig_path, litho_gt, aerial_recon, aerial_shipped)
    metrics(aerial_def_shipped, aerial_def_recon,
            "Defocus: shipped kernels vs recon kernels")

    # ── Save results ──────────────────────────────────────────────────────────
    if out_dir is not None:
        print(f"\n[10] Saving arrays and summary to {out_dir} ...")
        np.save(out_dir / "aerial_shipped_focus.npy",  aerial_shipped.astype(np.float32))
        np.save(out_dir / "aerial_recon_focus.npy",    aerial_recon.astype(np.float32))
        np.save(out_dir / "aerial_shipped_defocus.npy", aerial_def_shipped.astype(np.float32))
        np.save(out_dir / "aerial_recon_defocus.npy",   aerial_def_recon.astype(np.float32))
        if litho_gt is not None:
            np.save(out_dir / "litho_gt.npy", litho_gt.astype(np.float32))

        summary_lines = [
            "TCC reconstruction summary",
            f"sigma_in={sigma_in_use}  sigma_out={sigma_out_use}",
            f"n_medium={n_med}  Dz={Dz} nm",
            f"nc_path={args.nc}  sample_idx={args.sample_idx}",
            "",
            "Focus: shipped vs recon",
            f"  RMS={m_shipped_recon['rms']:.6f}  MAE={m_shipped_recon['mae']:.6f}  "
            f"corr={m_shipped_recon['corr']:.8f}  scale={m_shipped_recon['scale']:.6f}  "
            f"RMS_sc={m_shipped_recon['rms_sc']:.6f}",
        ]
        if litho_gt is not None:
            summary_lines += [
                "Focus: GT vs shipped (sanity)",
                f"  RMS={m_gt_shipped['rms']:.6f}  corr={m_gt_shipped['corr']:.8f}",
                "Focus: GT vs recon (Hopkins verification)",
                f"  RMS={m_gt_recon['rms']:.6f}  corr={m_gt_recon['corr']:.8f}",
                f"Comparison figure: {fig_path}",
            ]
        with open(out_dir / "summary.txt", "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"    Saved summary.txt and aerial .npy arrays")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
