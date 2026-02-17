# Lesson 12: GPU Job Orchestration with SLURM

This lesson covers using SLURM to manage and schedule GPU workloads on the GB10, including installation, job submission, troubleshooting, and running common HPC and ML jobs.

## SLURM

### SLURM GPU Architecture
```
┌─────────────────────────────────────────────────┐
│              SLURM Cluster                      │
│  ┌──────────────────────────────────────────┐   │
│  │      slurmctld (Controller)              │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  slurmd (Compute Node with GPUs)         │   │
│  │  ┌────────────────────────────────────┐  │   │
│  │  │  Job Queue                         │  │   │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐        │  │   │
│  │  │  │Job 1 │ │Job 2 │ │Job 3 │        │  │   │
│  │  │  └──────┘ └──────┘ └──────┘        │  │   │
│  │  └────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────┐  │   │
│  │  │  GPU Resources (via GRES)          │  │   │
│  │  │  ┌──────────────────────────────┐  │  │   │
│  │  │  │  GPU 0  │  GPU 1  │  GPU 2   │  │  │   │
│  │  │  └──────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Comparison: Kubernetes vs SLURM

| Feature | Kubernetes | SLURM |
|---------|-----------|-------|
| **Primary Use** | Microservices, web apps | HPC, batch jobs |
| **GPU Sharing** | Time-slicing, MIG | Exclusive or shared |
| **Scheduling** | Real-time, declarative | Batch queue, priority |
| **Networking** | Service mesh, ingress | InfiniBand, high-speed |
| **State** | Stateless preferred | Stateful jobs common |
| **Auto-scaling** | Built-in HPA/VPA | Manual configuration |
| **Best For** | AI inference, serving | Training, simulations |

### Install
```bash
# Install dependencies
sudo apt-y install environment-modules build-essential cmake git m4 flex bison zlib1g-dev libboost-system-dev libboost-thread-dev 

# Install SLURM
cd slurm
sudo ./install.sh
srun -n1 hostname
```

### Useful commands
```bash
squeue
scancel 1
sinfo
# Submit background job
sbatch run_train_model.sh
# Submit interactive job
srun run_train_model.sh
```

### Troubleshooting

***Job stuck in queue***
```bash
sudo pkill -9 slurmctld
sudo pkill -9 slurmd
sudo pkill -9 slurmstepd

# Check if anything is still running (should return empty)
ps aux | grep slurm | grep -v grep

# Wipe the state locations again
sudo rm -rf /var/spool/slurm/ctld/*
sudo rm -rf /var/spool/slurm/d/*

# Restart services
sudo systemctl start slurmctld
sudo systemctl start slurmd
sudo chmod 755 /var/spool/slurm
```

### Run traditional CFD based job. CPU only.
```bash
# Install OpenFOAM Foundation v12
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt -y install openfoam12 paraview

# Download the example mesh https://holzmann-cfd.com/community/training-cases/gin-tonic
wget -P /tmp https://holzmann-cfd.com/OpenFOAMCases/017_ginTonic/ginTonicCHT-12.tar.gz
tar -xvf /tmp/ginTonicCHT-12.tar.gz -C /tmp
cd /tmp/cases-12/ginTonicCHT
cp -r 0.orig 0

vi system/decomposeParDict

numberOfSubdomains 20;
method scotch;

simpleCoeffs
{
    n               (2 1 2);
    delta           0.001;
}

# Submit batch job to SLURM
sbatch ~/git/gb10-training/gb10-06/slurm/cfd_job/run_gintonic.sh
# This will output a job number, replace that in the tail command
tail -f /tmp/lab_hpc_28.log
htop
# This will take 20-30 minutes to run
```

### Run LLM training job
```bash
wget -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-py313_25.11.1-1-Linux-aarch64.sh
chmod u+x /tmp/Miniconda3-py313_25.11.1-1-Linux-aarch64.sh
sudo bash /tmp/Miniconda3-py313_25.11.1-1-Linux-aarch64.sh -b -p /opt/anaconda/25.11.1-1

# Create environment module (Used with the `module load` command)
sudo chown -R root:${USER} /etc/environment-modules/modules
sudo chmod -R 775 /etc/environment-modules/modules
mkdir -p /etc/environment-modules/modules/anaconda
cat << EOF > /etc/environment-modules/modules/anaconda/25.11.1-1
#%Module1.0
##
## Anaconda Modulefile
##
proc ModulesHelp { } {
    puts stderr "\tThis module adds Anaconda to your environment."
}

module-whatis   "Name: Anaconda"
module-whatis   "Version: 25.11.1-1"

set root "/opt/anaconda/25.11.1-1"

# Set Path
prepend-path    PATH            $root/bin

# The magic for conda activate
if { [ module-info mode load ] } {
    puts stdout "source $root/etc/profile.d/conda.sh;"
}
EOF

# Reinstall torch with GPU support
source ~/venv/gb10-training/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

sbatch ~/git/gb10-training/gb10-06/slurm/training_job/run_train_model_cpu.sh
# This will output a job number, replace that in the tail command
tail -f /tmp/lab_hpc_28.log

sbatch ~/git/gb10-training/gb10-06/slurm/training_job/run_train_model_gpu.sh
# This will output a job number, replace that in the tail command
tail -f /tmp/lab_hpc_28.log
```

### Run GROMACS job

GROMACS (GROningen MAchine for Chemical Simulations) is a free, open-source, and high-performance software package designed for molecular dynamics simulations, primarily for studying biomolecules like proteins, lipids, and nucleic acids. It simulates Newton's equations of motion for systems with millions of particles, offering exceptional speed for non-bonded interactions. It is widely used in drug design and biophysics. 

https://www.gromacs.org
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/gromacs

#### Compile the software
```bash
cd ~/git
wget https://ftp.gromacs.org/gromacs/gromacs-2026.0.tar.gz
tar xfz gromacs-2026.0.tar.gz
cd gromacs-2026.0
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON -DGMX_GPU=CUDA -DCMAKE_CUDA_ARCHITECTURES=native
make
make check
sudo make install
```

#### Create the environment module
```bash
mkdir -p /etc/environment-modules/modules/gromacs
cat << EOF > /etc/environment-modules/modules/gromacs/2026.0
#%Module1.0
##
## GROMACS 2026.0 Modulefile
##
proc ModulesHelp { } {
    puts stderr "\tThis module adds GROMACS 2026.0 to your environment."
    puts stderr "\tIt handles binary paths, libraries, and man pages via GMXRC."
}

module-whatis   "Name: GROMACS"
module-whatis   "Version: 2026.0"

# The root directory where GROMACS is installed
set root "/usr/local/gromacs"

# Set standard paths for direct module control (optional but recommended)
prepend-path    PATH            \$root/bin
prepend-path    LD_LIBRARY_PATH \$root/lib64
prepend-path    MANPATH         \$root/share/man

# The magic for GMXRC (equivalent to 'source /usr/local/gromacs/bin/GMXRC')
if { [ module-info mode load ] } {
    puts stdout "source \$root/bin/GMXRC;"
}

# Optional: Clean up if the module is unloaded
if { [ module-info mode remove ] } {
    puts stdout "unset GMXBIN; unset GMXLDLIB; unset GMXMAN; unset GMXDATA;"
}
EOF
```

#### Submit the job
```bash
cd /tmp
wget https://zenodo.org/record/3893789/files/GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP.tar.gz 
tar xf GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP.tar.gz 
cd GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP/stmv

echo -e "\ncompressed-x-grps   = System" >> pme_nvt.mdp
sed -i '/nstxout-compressed/c\nstxout-compressed  = 5000' pme_nvt.mdp
sed -i 's/constraints.*/constraints              = all-bonds/' pme_nvt.mdp
sed -i 's/nstlist.*/nstlist                  = 100/' pme_nvt.mdp

/usr/local/gromacs/bin/gmx editconf -f conf.gro -o output.pdb
/usr/local/gromacs/bin/gmx grompp -f pme_nvt.mdp -c conf.gro -p topol.top -o output.tpr

sbatch ~/git/gb10-training/gb10-06/slurm/gromacs_job/gromacs_run.sh
tail -f gmx_output_40.log

# When the job completes you should have some new output.* files 
ls -halt | head

# Copy the output.gro and output.xtc to your laptop
```

#### Run the visualization

Install VMD on your Windows laptop (You have to build it from source if you want to install it on the GB10)
https://www.ks.uiuc.edu/Development/Download/download.cgi

Launch VMD from the Start Menu

Load the Structure:
   - Go to File > New Molecule.
   - Click Browse and select output.gro.
   - Click Load. You should see a "cloud" of dots in the - display window.

Load the Trajectory (The Movement):
   - In the VMD Main window, left click output.gro, now right-click it.
   - Select Load Data Into Molecule.
   - Browse and select your trajectory file output.xtc.
   - Click Load. 

Make it Look "Real-World":
Hide the Water
   - If the virus is surrounded by water, which blocks our view.
   - Go to Graphics > Representations.
   - In the "Selected Atoms" box, delete all and type: protein.
   - Press Enter. Now only the virus is visible.
Choose a Drawing Style
   - Change Drawing Method to NewCartoon.
   - Change Coloring Method to Secondary Structure.
   - Change Material to Transparent.
   - Click the Create Rep button.
   - In Selected Atoms, type nucleic.
   - Set Drawing Method to VDW or QuickSurf.
   - Set Coloring Method to ColorID and pick a bright color like Yellow.



