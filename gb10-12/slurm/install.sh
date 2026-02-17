#!/bin/bash
set -e

# SLURM setup for single-node Ubuntu 24.04 (Teaching/Demo use)
# Optimized for Cgroup v2 and includes resource safety buffers.

echo "Starting SLURM setup for Ubuntu 24.04..."

if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run as root or with sudo"
  exit 1
fi

# 1. System Updates & Package Installation
echo "Installing Slurm, Munge, and Cgroup dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
  slurm-wlm slurmctld slurmd munge libmunge2 cgroup-tools nvidia-cuda-toolkit-gcc

# 2. Create slurm user if missing (usually handled by apt, but good for safety)
if ! id -u slurm >/dev/null 2>&1; then
  useradd -r -m -d /var/lib/slurm -s /usr/sbin/nologin slurm || true
fi

# 3. Directory Setup & Permissions
echo "Configuring directories..."
mkdir -p /var/spool/slurm/ctld /var/spool/slurm/d /var/log/slurm /etc/slurm
chown -R slurm:slurm /var/spool/slurm /var/log/slurm /etc/slurm
chmod 755 /etc/slurm
chmod 750 /var/spool/slurm /var/log/slurm

# 4. Munge Security Setup
if [ ! -f /etc/munge/munge.key ]; then
  echo "Generating Munge key..."
  dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key
  chown munge:munge /etc/munge/munge.key
  chmod 400 /etc/munge/munge.key
fi

systemctl enable --now munge
sleep 1 # Ensure munge socket is initialized

# 5. Resource Detection
HOSTNAME=$(hostname -s)
CPUS=$(nproc)
# Calculate RealMemory: Total RAM minus 1GB buffer for OS stability
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
MEM_MB=$(( TOTAL_MEM - 1024 ))

# GPU Detection logic
GPU_COUNT=0
GRES_LINE=""
if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
  if [ "$GPU_COUNT" -gt 0 ]; then
    GRES_LINE="Gres=gpu:$GPU_COUNT"
    echo "GPUs detected: $GPU_COUNT. Configuring GRES..."
    cat > /etc/slurm/gres.conf <<EOF
NodeName=$HOSTNAME Name=gpu File=/dev/nvidia[0-$((GPU_COUNT-1))]
EOF
  fi
fi

# 6. Writing slurm.conf (Cgroup v2 / 24.04 compatible)
echo "Generating slurm.conf..."
cat > /etc/slurm/slurm.conf <<EOF
# Minimal slurm.conf for single-node demo (Ubuntu 24.04)
ClusterName=localcluster
SlurmctldHost=$HOSTNAME
AuthType=auth/munge
CryptoType=crypto/munge
SlurmUser=slurm
SlurmdUser=root

# Scheduling & Resource Management
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
GresTypes=gpu

# Logging & State
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d

# Process Tracking (Cgroup v2)
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup

# Node Definitions
NodeName=$HOSTNAME CPUs=$CPUS RealMemory=$MEM_MB $GRES_LINE State=UNKNOWN
PartitionName=debug Nodes=$HOSTNAME Default=YES MaxTime=INFINITE State=UP

MailProg=/usr/bin/true
EOF

# 7. Writing cgroup.conf (Mandatory for 24.04)
echo "Generating cgroup.conf..."
cat > /etc/slurm/cgroup.conf <<EOF
CgroupMountpoint=/sys/fs/cgroup
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
EOF

# 8. Final Permissions Fix
chown slurm:slurm /etc/slurm/*.conf
chmod 644 /etc/slurm/*.conf

# 9. Launch Services
echo "Starting Slurm Daemons..."
systemctl enable --now slurmctld slurmd

# 10. Verification
echo "--------------------------------------------------"
echo "Setup complete! Checking status..."
sleep 2
sinfo

echo ""
echo "Try running a test job:"
echo "srun --pty -n1 bash"
if [ "$GPU_COUNT" -gt 0 ]; then
  echo "To test GPU: srun --gres=gpu:1 nvidia-smi"
fi