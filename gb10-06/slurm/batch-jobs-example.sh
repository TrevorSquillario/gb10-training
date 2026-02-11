#!/bin/bash
#SBATCH --job-name=gpu-array
#SBATCH --output=slurm-array-%A-%a.out
#SBATCH --error=slurm-array-%A-%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-3

# SLURM Array Job Example
# This demonstrates running multiple similar jobs with different parameters
#
# Submit with: sbatch batch-jobs-example.sh
# Check status: squeue
# View outputs: ls slurm-array-*.out

echo "=========================================="
echo "SLURM Array Job - Task $SLURM_ARRAY_TASK_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Display allocated GPU
echo "GPU Information:"
nvidia-smi -L
echo ""

# Simulate different workloads based on task ID
case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Task 0: Running GPU memory test"
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
        ;;
    1)
        echo "Task 1: Running GPU utilization monitor"
        nvidia-smi dmon -c 5
        ;;
    2)
        echo "Task 2: Running nvidia-smi in continuous mode"
        for i in {1..5}; do
            echo "Iteration $i:"
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv
            sleep 2
        done
        ;;
    3)
        echo "Task 3: Running full nvidia-smi report"
        nvidia-smi -q
        ;;
esac

echo ""
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
