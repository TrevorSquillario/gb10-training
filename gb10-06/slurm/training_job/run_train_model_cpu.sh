#!/bin/bash
#SBATCH --job-name=gpu_train
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --output=/tmp/lab_hpc_%j.log

# 2. Load Anaconda (provides the Python interpreter your venv likely uses)
module load anaconda/25.11.1-1

# 3. Activate your Python VENV by path
# Replace this path with the actual path to your venv
source /home/trevor/venv/gb10-training/bin/activate

# 5. Run training
python train_model.py