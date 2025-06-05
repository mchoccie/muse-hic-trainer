#!/bin/bash

#!/bin/bash
#SBATCH --job-name=vqgan_train
#SBATCH --output=logs/vqgan_%j.out
#SBATCH --error=logs/vqgan_%j.err
#SBATCH --partition=mgpuq            # adjust this to your cluster's GPU partition name
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load environment or activate conda
source ~/.bashrc
conda activate muse  # or your env name

cd /scratch/rnd-rojas/Manan
# Run your script
python trainVQGan.py