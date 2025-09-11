#!/bin/bash
#SBATCH --job-name=muse_advanced
#SBATCH --output=logs/muse_advanced_%j.out
#SBATCH --error=logs/muse_advanced_%j.err
#SBATCH --partition=gpuqm
#SBATCH --nodelist=g18
#SBATCH --time=96:00:00             # 4 days for full training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20          # more CPUs for DNA encoding and data loading
#SBATCH --mem=256G                  # increased memory for large models
#SBATCH --mail-type=ALL             # email notifications
#SBATCH --mail-user=your.email@institution.edu  # replace with your email

# Load environment
source ~/.bashrc
conda activate muse

cd /scratch/rnd-rojas/Manan

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up proxy for wandb
export PROXY_PORT=1337
ssh -D $PROXY_PORT -qCNf coe-hpc2
export http_proxy=socks5h://localhost:1337
export https_proxy=socks5h://localhost:1337

# Set CUDA device and memory settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1       # for better error reporting
export OMP_NUM_THREADS=20           # match CPU count

# Set PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Print system info
echo "=== System Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Check if required files exist
# echo "=== Checking Required Files ==="
# required_files=(
#     "muse_pipeline_improved.py"
#     "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset.npy"
#     "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset.npy"
#     "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_window_coords.npy"
#     "/scratch/rnd-rojas/Manan/hg19.fa"
#     "/scratch/rnd-rojas/Manan/qv_results4/vae.best_srcc.pt"
# )

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done

# Check GPU memory
echo "=== GPU Memory Check ==="
nvidia-smi

# Start training with error handling
echo "=== Starting Muse Training ==="
start_time=$(date +%s)

python muse_pipeline_downsampled.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "=== Training Completed Successfully ==="
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Total training time: $((duration / 3600)) hours, $(((duration % 3600) / 60)) minutes"
else
    echo "=== Training Failed ==="
    exit 1
fi

# Clean up proxy
pkill -f "ssh -D $PROXY_PORT"

echo "=== Job Finished ===" 