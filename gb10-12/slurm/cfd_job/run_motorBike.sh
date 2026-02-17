#!/bin/bash
#SBATCH --job-name=motorBike
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:30:00
#SBATCH --partition=debug  # Change this to your cluster's partition name

# Load OpenFOAM environment
source /opt/openfoam12/etc/bashrc

# Run
./Allrun

# Reconstruct for post-processing
#reconstructPar

touch motorBike.foam