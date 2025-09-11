#!/bin/bash
#SBATCH --job-name=test_sr_no_dna
#SBATCH --output=logs/test_sr_no_dna_%j.out
#SBATCH --error=logs/test_sr_no_dna_%j.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Exit on any error
set -euo pipefail

# Fix bash configuration warning
export BASHRCSOURCED=1

# Load environment
source ~/.bashrc
conda activate muse

# Set up environment variables
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_VISIBLE_DEVICES=0

# Disable wandb for testing
export WANDB_MODE=offline
export WANDB_SILENT=true

# Change to working directory
cd /scratch/rnd-rojas/Manan
mkdir -p logs

echo "=========================================="
echo "Super-Resolution Testing (No DNA Context)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'GPU info unavailable')"
echo "=========================================="

# Test 1: Best checkpoint
echo ""
echo "Testing super-resolution on best checkpoint..."
python test_super_resolution_nodna_downsampled.py \
    --checkpoint checkpointbest.pt \
    --timesteps 28 \
    --assume-preprocessed \
    --expects same \
    --num-batches 4
    # --edit-frac 0.6 \
    # --focus band \
    # --band-frac 0.12 \
    # --seeded \
    

#Test 2: Latest epoch
# echo ""
# echo "Testing super-resolution on epoch 12..."
# python test_super_resolution_nodna_downsampled.py \
#     --edit-frac 0.6 \
#     --focus band \
#     --band-frac 0.12 \
#     --epoch 12 \
#     --timesteps 18 \
#     --assume-preprocessed \
#     --expects same \
#     --num-batches 4

# # Test 3: Additional epochs for comparison
# for epoch in 8 10 11; do
#     echo ""
#     echo "Testing super-resolution on epoch $epoch..."
#     python test_super_resolution_nodna_downsampled.py \
#         --epoch $epoch \
#         --timesteps 36 \
#         --assume-preprocessed \
#         --expects same \
#         --num-batches 2
# done

echo ""
echo "=========================================="
echo "Super-Resolution Testing Complete!"
echo "=========================================="
echo "Generated files:"
echo "- sr_comparison_no_dna_batch_*.png"
echo "- sr_methods_comparison_no_dna_batch_*.png"
echo "=========================================="
