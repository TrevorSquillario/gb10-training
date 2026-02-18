#!/bin/bash
#SBATCH --job-name=LYSO_SIM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        
#SBATCH --gres=gpu:1             
#SBATCH --partition=debug
#SBATCH --time=02:00:00          # Reduced time; Lysozyme is fast!
#SBATCH --output=log.out
#SBATCH --error=log.err

# 1. Clean environment
module purge

# 2. Load your custom GROMACS module
module load gromacs/2026.0

# 3. Environment Check
echo "Running GROMACS from: $(which gmx)"
gmx --version | grep "Precision"

# 4. Run the Production MD
gmx mdrun -v \
    -s ~/gb10/scratch/GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP/stmv/output.tpr \
    -deffnm ~/gb10/scratch/GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP/stmv/output \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu \
    -pme gpu \
    -bonded gpu \
    -update cpu \
    -nsteps 10000 \
    -resethway \
    -nstlist 100 \
    -notunepme \
    -pin on