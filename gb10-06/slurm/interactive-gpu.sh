#!/bin/bash
# Interactive GPU Session Example
#
# This script demonstrates how to request an interactive session with GPU access
# Useful for development, testing, and debugging

echo "=========================================="
echo "SLURM Interactive GPU Session"
echo "=========================================="
echo ""
echo "This script will request an interactive session with GPU access"
echo ""

# Request interactive session
# Adjust resources as needed:
#   --gres=gpu:1      - Request 1 GPU
#   --gres=gpu:2      - Request 2 GPUs
#   --cpus-per-task=4 - Request 4 CPU cores
#   --mem=16G         - Request 16GB RAM
#   --time=02:00:00   - Session duration (2 hours)

echo "Requesting interactive session with:"
echo "  1 GPU"
echo "  4 CPU cores"
echo "  16GB RAM"
echo "  2 hour time limit"
echo ""
echo "Run this command:"
echo ""
echo "srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash"
echo ""
echo "Or for a more feature-rich shell:"
echo ""
echo "srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -i"
echo ""
echo "Once in the session, you can:"
echo "  - Run nvidia-smi to check GPU"
echo "  - Test CUDA applications"
echo "  - Run Python scripts with PyTorch/TensorFlow"
echo "  - Install packages in your home directory"
echo ""
echo "To exit the session, type 'exit' or press Ctrl+D"
echo ""

# Uncomment to automatically start the session
# srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -i
