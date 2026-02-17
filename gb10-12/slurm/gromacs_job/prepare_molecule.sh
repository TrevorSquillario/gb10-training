#!/bin/bash

# Used for molecules from https://www.rcsb.org

# --- Configuration ---
PDB_ID="1GD6"
TOPOLOGY="topol.top"

# Move to tmp directory for processing
if [ -d "/tmp/${PDB_ID}" ]; then
    echo "--- /tmp/${PDB_ID} exists; removing to ensure clean workspace ---"
    rm -rf "/tmp/${PDB_ID}"
fi
mkdir -p "/tmp/${PDB_ID}"
cd "/tmp/${PDB_ID}" || { echo "Failed to enter /tmp/${PDB_ID}"; exit 1; }

module purge
echo "--- Load GROMACS module ---"
module load gromacs/2026.0

echo "--- Step 1: Downloading PDB ---"
wget -nc https://files.rcsb.org/download/${PDB_ID}.pdb

echo "--- Step 2: Cleaning PDB ---"
grep -v HOH ${PDB_ID}.pdb > ${PDB_ID}_clean.pdb

echo "--- Step 3: Running pdb2gmx ---"
gmx pdb2gmx -f ${PDB_ID}_clean.pdb -o ${PDB_ID}_processed.gro -water spce -ff oplsaa

echo "--- Step 4: Defining Box ---"
# Note: Increase -d 1.0 to 2.0 if you want a larger system that takes longer to run
gmx editconf -f ${PDB_ID}_processed.gro -o ${PDB_ID}_newbox.gro -c -d 1.0 -bt cubic

echo "--- Step 5: Solvating ---"
gmx solvate -cp ${PDB_ID}_newbox.gro -cs spc216.gro -o ${PDB_ID}_solv.gro -p $TOPOLOGY

echo "--- Step 6: Creating MDP Parameters (Minim & Production) ---"
# Minimization MDP
cat << EOF > minim.mdp
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

# Production MDP (100ps run)
cat << EOF > nvt.mdp
integrator              = md
nsteps                  = 50000     ; 2fs * 50,000 = 100 ps
dt                      = 0.002
nstlist                 = 20
cutoff-scheme           = Verlet
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1 0.1
ref_t                   = 300 300
constraints             = h-bonds
pbc                     = xyz
EOF

echo "--- Step 7: Adding Ions ---"
gmx grompp -f minim.mdp -c ${PDB_ID}_solv.gro -p $TOPOLOGY -o ions.tpr -maxwarn 1
echo 13 | gmx genion -s ions.tpr -o ${PDB_ID}_final.gro -p $TOPOLOGY -pname NA -nname CL -neutral

echo "--- Step 8: Final Assembly (Minimization) ---"
gmx grompp -f minim.mdp -c ${PDB_ID}_final.gro -p $TOPOLOGY -o md_0_1.tpr

echo "--- Step 9: Final Assembly (Production) ---"
# We use the final.gro from step 7 as the starting coordinates
gmx grompp -f nvt.mdp -c ${PDB_ID}_final.gro -p $TOPOLOGY -o production.tpr

echo "--- Files prepared in /tmp/${PDB_ID} ---"
echo "------------------------------------------------------"
echo "SUCCESS: production.tpr is ready."
echo "Run: gmx mdrun -v -deffnm production -nb gpu -pme gpu -bonded gpu -update gpu"
echo "------------------------------------------------------"