#!/bin/bash
#SBATCH --job-name=gpu_train
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --output=/tmp/lab_hpc_%j.log

# 1. Load your specific environment
module load anaconda/25.11.1-1
source /home/trevor/venv/gb10-training/bin/activate

# 2. Fix the library path for your ARM64 system
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# 3. CRITICAL: Unset conflicting visibility variables 
# Let Slurm's cgroups handle it.
unset CUDA_VISIBLE_DEVICES

# 4. Verification Check (will appear in your log)
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()} | Count: {torch.cuda.device_count()}')"

# 5. Run training
python train_model.py --device cuda