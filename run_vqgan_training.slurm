#!/bin/bash
#SBATCH --job-name=vqgan_hic_training
#SBATCH --output=logs/vqgan_training_%j.out
#SBATCH --error=logs/vqgan_training_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpuqm
#SBATCH --nodelist=g16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@institution.edu


# Initialize conda and activate environment
eval "$(conda shell.bash hook)"
conda activate muse

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set working directory
cd /scratch/rnd-rojas/Manan

export PROXY_PORT=1337
ssh -D $PROXY_PORT -qCNf coe-hpc2
export http_proxy=socks5h://localhost:1337
export https_proxy=socks5h://localhost:1337

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create results directory if it doesn't exist
mkdir -p ./vq_highres_results_gpt5

# Run the training script
echo "Starting VQGAN training..."
python trainVQGan.py

echo "Training completed!"
