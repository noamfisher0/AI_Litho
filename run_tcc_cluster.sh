#!/usr/bin/env bash
# run_tcc_cluster.sh
#
# Run TCC reconstruction + verification on the Euler cluster.
# Source your .venv before launching this script, or run from an interactive
# node where the venv is already active.
#
# Usage (interactive node, venv already sourced):
#   bash scripts/run_tcc_cluster.sh
#
# Or as a batch job (add your SLURM header above this line):
#   #SBATCH --job-name=tcc_reconstruct
#   #SBATCH --time=01:00:00
#   #SBATCH --mem=32G
#   #SBATCH --cpus-per-task=4
#   #SBATCH --output=/cluster/scratch/nfisher/tcc_results/slurm_%j.out

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=/cluster/home/nfisher/cameleon
NC=/cluster/work/math/camlab-data/tmp_share/LithoBenchData/LithoBenchData-compressed/LithoBench-MetalSet.nc
OUT=/cluster/scratch/nfisher/tcc_results

mkdir -p "$OUT"

SCRIPT="$REPO/scripts/tcc_reconstruct.py"
PYTHON=python   # .venv must already be sourced; or replace with full path

cd "$REPO"

echo "=================================================="
echo "Fixed ICCAD 2013 TCC reconstruction"
echo "  Uses sigma_in=0.3 and sigma_out=0.9"
echo "  Builds focus + defocus TCCs at 201x201"
echo "  Runs SOCS aerials on a 2048x2048 mask from:"
echo "  $NC"
echo "  Compares recon kernels vs shipped kernels vs LithoBench GT"
echo "  Saves an aerial comparison figure and summary in $OUT"
echo "  Expected runtime: ~20-30 min"
echo "=================================================="

$PYTHON "$SCRIPT" \
    --nc         "$NC" \
    --sample-idx 0 \
    --n-source   201 \
    --out-dir    "$OUT" \
    2>&1 | tee "$OUT/verify.log"

echo ""
echo "All done. Results in $OUT/"
echo "  verify.log       — reconstruction and verification output"
echo "  summary.txt      — key metrics"
echo "  recon_kernels_*.pt / recon_scales_*.pt — reconstructed kernel tensors"
echo "  aerial_*.npy     — aerial images for all four conditions"
