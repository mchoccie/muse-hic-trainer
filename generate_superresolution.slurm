#!/bin/bash
#SBATCH --job-name=superres_gen
#SBATCH --output=logs/superres_gen_%j.out
#SBATCH --error=logs/superres_gen_%j.err
#SBATCH --partition=gpuqm
#SBATCH --nodelist=g15
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load environment or activate conda
source ~/.bashrc
conda activate muse

cd /scratch/rnd-rojas/Manan

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up proxy for wandb (if needed)
export PROXY_PORT=1337
ssh -D $PROXY_PORT -qCNf coe-hpc2
export http_proxy=socks5h://localhost:1337
export https_proxy=socks5h://localhost:1337

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=== Super-Resolution Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# First, try to install missing dependency if needed
echo "Checking for enformer-pytorch..."
pip install enformer-pytorch

# Run super-resolution generation
echo "Running super-resolution generation..."

# Test with dataset sample (easiest)
python generate_superresolution.py \
    --checkpoint checkpointbest.pt \
    --sample-idx 0 \
    --output generated_sr_sample0.png

# Test with another sample
python generate_superresolution.py \
    --checkpoint checkpointbest.pt \
    --sample-idx 5 \
    --output generated_sr_sample5.png

# Test with epoch 12 checkpoint
python generate_superresolution.py \
    --epoch 12 \
    --sample-idx 0 \
    --output generated_sr_epoch12.png

echo "=== Super-Resolution Generation Complete ==="

# Clean up proxy
pkill -f "ssh -D $PROXY_PORT" 