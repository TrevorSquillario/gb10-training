#!/bin/bash
#SBATCH --job-name=nvidia-smi-test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# SLURM GPU Test Job
# This job requests 1 GPU and runs nvidia-smi to display GPU information
#
# Submit with: sbatch nvidia-smi-job.sh
# Check status: squeue
# View output: cat slurm-<jobid>.out

echo "=========================================="
echo "SLURM GPU Test Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Display allocated resources
echo "Allocated Resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo "  GPUs: $SLURM_GPUS"
echo "  GPU Devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Check if CUDA is available
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "WARNING: No CUDA devices visible"
    echo "CUDA_VISIBLE_DEVICES is not set"
else
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Run nvidia-smi
echo "=========================================="
echo "nvidia-smi Output"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    
    # Show detailed GPU info
    echo "=========================================="
    echo "Detailed GPU Information"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,utilization.memory --format=csv
    echo ""
    
    # Show process information
    echo "=========================================="
    echo "GPU Process Information"
    echo "=========================================="
    nvidia-smi pmon -c 1
    echo ""
else
    echo "ERROR: nvidia-smi command not found"
    echo "NVIDIA drivers may not be installed or available on this node"
    exit 1
fi

# Optional: Run a simple CUDA test if available
if command -v nvidia-smi &> /dev/null; then
    echo "=========================================="
    echo "Running GPU Compute Test"
    echo "=========================================="
    # Simple GPU memory bandwidth test
    nvidia-smi --query-gpu=memory.total --format=csv,noheader | while read mem; do
        echo "GPU Memory: $mem"
    done
fi

echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
