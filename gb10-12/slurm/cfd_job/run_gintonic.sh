#!/bin/bash
#SBATCH --job-name=ginTonic_CFD
#SBATCH --partition=debug
#SBATCH --nodes=1 
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/lab_hpc_%j.log

# 1. Load the OpenFOAM environment
source /opt/openfoam12/etc/bashrc

# 2. Navigate to the case directory
# Note: Ensure this directory exists on the node's local storage
cd /tmp/cases-12/ginTonicCHT

# 3. Parallel Execution
# Using srun ensures Slurm distributes the 20 tasks across the cores.
# We pass -parallel to the solver so it knows to look for decomposed processors.
echo "Starting OpenFOAM 'ginTonic' simulation at $(date)"

srun --mpi=pmix chtMultiRegionFoam -parallel

echo "Simulation finished at $(date)"