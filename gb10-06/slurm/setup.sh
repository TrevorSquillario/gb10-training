#!/bin/bash
set -e

echo "=========================================="
echo "SLURM Setup for GPU Workloads"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "Cannot detect OS"
    exit 1
fi

echo "Detected OS: $OS $VER"

# Install SLURM and dependencies
echo "Installing SLURM packages..."
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    apt-get update
    apt-get install -y \
        slurm-wlm \
        slurm-client \
        slurmd \
        slurmctld \
        munge \
        libmunge-dev \
        build-essential \
        mailutils \
        libpam0g-dev \
        libmariadb-dev \
        libmysqlclient-dev
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ]; then
    yum install -y epel-release
    yum install -y \
        slurm \
        slurm-slurmd \
        slurm-slurmctld \
        munge \
        munge-devel \
        gcc \
        make
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Create SLURM user
if ! id -u slurm >/dev/null 2>&1; then
    echo "Creating slurm user..."
    useradd -r -d /var/lib/slurm -s /bin/false slurm
fi

# Create required directories
echo "Creating SLURM directories..."
mkdir -p /var/spool/slurm/d
mkdir -p /var/spool/slurm/ctld
mkdir -p /var/log/slurm
mkdir -p /etc/slurm

chown -R slurm:slurm /var/spool/slurm
chown -R slurm:slurm /var/log/slurm
chmod 755 /var/spool/slurm
chmod 755 /var/log/slurm

# Setup Munge (authentication)
echo "Configuring Munge authentication..."
if [ ! -f /etc/munge/munge.key ]; then
    dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi

# Start munge
systemctl enable munge
systemctl restart munge
systemctl status munge --no-pager

# Detect GPUs
echo "Detecting GPUs..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "Found $GPU_COUNT GPU(s): $GPU_NAME"
else
    GPU_COUNT=0
    GPU_NAME="none"
    echo "No NVIDIA GPUs detected"
fi

# Get system information
HOSTNAME=$(hostname -s)
CPUS=$(nproc)
MEM_MB=$(free -m | awk '/^Mem:/{print $2}')

# Generate SLURM configuration
echo "Generating SLURM configuration..."
cat > /etc/slurm/slurm.conf <<EOF
# SLURM Configuration - Auto-generated on $(date)
ClusterName=gb10-cluster
SlurmctldHost=$HOSTNAME

# Authentication
AuthType=auth/munge
CryptoType=crypto/munge

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# Logging
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldDebug=info
SlurmdDebug=info

# Process tracking
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup

# Timeouts
MessageTimeout=60
SlurmctldTimeout=300
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# State preservation
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d

# Return to service
ReturnToService=2

# MPI
MpiDefault=none

# Accounting
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none

# Node definition
NodeName=$HOSTNAME CPUs=$CPUS RealMemory=$MEM_MB State=UNKNOWN Gres=gpu:$GPU_COUNT

# Partition definition
PartitionName=gpu Nodes=$HOSTNAME Default=YES MaxTime=INFINITE State=UP
EOF

# Generate GRES configuration (GPU resources)
if [ $GPU_COUNT -gt 0 ]; then
    echo "Configuring GPU resources..."
    cat > /etc/slurm/gres.conf <<EOF
# GPU Resources Configuration
# Auto-generated on $(date)
NodeName=$HOSTNAME Name=gpu File=/dev/nvidia[0-$((GPU_COUNT-1))]
EOF
fi

# Generate cgroup configuration
echo "Configuring cgroups..."
cat > /etc/slurm/cgroup.conf <<EOF
CgroupAutomount=yes
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
EOF

# Set permissions
chown slurm:slurm /etc/slurm/slurm.conf
chown slurm:slurm /etc/slurm/gres.conf 2>/dev/null || true
chown slurm:slurm /etc/slurm/cgroup.conf
chmod 644 /etc/slurm/*.conf

# Enable and start SLURM services
echo "Starting SLURM services..."
systemctl enable slurmctld
systemctl enable slurmd
systemctl restart slurmctld
systemctl restart slurmd

# Wait a moment for services to start
sleep 3

# Check service status
echo ""
echo "=========================================="
echo "Service Status"
echo "=========================================="
systemctl status munge --no-pager | head -5
echo "---"
systemctl status slurmctld --no-pager | head -5
echo "---"
systemctl status slurmd --no-pager | head -5

# Set node to IDLE state
echo ""
echo "Setting node to IDLE state..."
scontrol update nodename=$HOSTNAME state=idle

# Verify configuration
echo ""
echo "=========================================="
echo "SLURM Configuration Summary"
echo "=========================================="
scontrol show config | grep -E "ClusterName|SlurmctldHost"
echo ""
echo "Node Information:"
scontrol show node $HOSTNAME
echo ""
echo "Partition Information:"
scontrol show partition

if [ $GPU_COUNT -gt 0 ]; then
    echo ""
    echo "GPU Resources:"
    scontrol show node $HOSTNAME | grep -i gres
fi

echo ""
echo "=========================================="
echo "SLURM Setup Complete!"
echo "=========================================="
echo ""
echo "Test commands:"
echo "  sinfo                    # View cluster info"
echo "  squeue                   # View job queue"
echo "  sbatch job.sh            # Submit a job"
echo "  srun hostname            # Run command"
echo "  scontrol show node       # Node details"
echo ""
echo "Example GPU job submission:"
echo "  sbatch --gres=gpu:1 nvidia-smi-job.sh"
echo ""
