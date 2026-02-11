#!/bin/bash
set -e

# Simplified SLURM setup for single-node Ubuntu (teaching/demo use)
# - Focuses on Ubuntu 22.04
# - Minimal dependencies
# - Auto-detects CPU/RAM/GPU
# - Uses Munge + cgroups
# - Does NOT install DB accounting, mail, or build tools

echo "Simplified SLURM setup (Ubuntu single-node)"

if [ "$EUID" -ne 0 ]; then
  echo "Run as root or with sudo"
  exit 1
fi

if [ ! -f /etc/os-release ]; then
  echo "Unsupported OS: cannot detect /etc/os-release"
  exit 1
fi

. /etc/os-release
if [ "$ID" != "ubuntu" ]; then
  echo "This script is written for Ubuntu. Adjust packages for other distros."
  exit 1
fi

echo "Updating apt and installing minimal packages..."
apt-get update
apt-get install -y --no-install-recommends \
  slurm-wlm slurmctld slurmd munge libmunge2 munge-tools cgroup-tools

# Create slurm user if missing
if ! id -u slurm >/dev/null 2>&1; then
  useradd -r -m -d /var/lib/slurm -s /usr/sbin/nologin slurm || true
fi

# Create directories
mkdir -p /var/spool/slurm /var/spool/slurm/d /var/log/slurm /etc/slurm
chown -R slurm:slurm /var/spool/slurm /var/log/slurm
chmod 750 /var/spool/slurm /var/log/slurm

# Setup Munge key (simple single-node)
if [ ! -f /etc/munge/munge.key ]; then
  dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key
  chown munge:munge /etc/munge/munge.key
  chmod 400 /etc/munge/munge.key
fi

systemctl enable --now munge || true

# Detect resources
HOSTNAME=$(hostname -s)
CPUS=$(nproc)
MEM_MB=$(free -m | awk '/^Mem:/{print $2}')

# Detect GPUs (optional)
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Write minimal slurm.conf
cat > /etc/slurm/slurm.conf <<EOF
# Minimal slurm.conf for single-node demo
ClusterName=localcluster
SlurmctldHost=$HOSTNAME
AuthType=auth/munge
CryptoType=crypto/munge
SchedulerType=sched/backfill
SelectType=select/cons_res
SelectTypeParameters=CR_Core
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldDebug=info
SlurmdDebug=info
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup
# Timeouts
MessageTimeout=60
SlurmctldTimeout=120
SlurmdTimeout=300
# State
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
# Node
NodeName=$HOSTNAME CPUs=$CPUS RealMemory=$MEM_MB State=UNKNOWN
PartitionName=debug Nodes=$HOSTNAME Default=YES MaxTime=INFINITE State=UP
EOF

# Add GRES if GPUs present
if [ "$GPU_COUNT" -gt 0 ]; then
  cat > /etc/slurm/gres.conf <<EOF
NodeName=$HOSTNAME Name=gpu File=/dev/nvidia[0-$((GPU_COUNT-1))]
EOF
  # Append Gres to slurm.conf line
  sed -i "s/NodeName=$HOSTNAME CPUs=/NodeName=$HOSTNAME CPUs=/; s/State=UNKNOWN/& Gres=gpu:$GPU_COUNT/" /etc/slurm/slurm.conf || true
fi

# Basic cgroup.conf
cat > /etc/slurm/cgroup.conf <<EOF
CgroupAutomount=yes
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
EOF

# Permissions
chown slurm:slurm /etc/slurm/slurm.conf || true
chown slurm:slurm /etc/slurm/gres.conf 2>/dev/null || true
chown slurm:slurm /etc/slurm/cgroup.conf || true
chmod 644 /etc/slurm/*.conf 2>/dev/null || true

# Enable and start services
systemctl enable --now slurmctld slurmd || true

sleep 2

echo ""
echo "SLURM simplified setup complete. Verify with:"
echo "  systemctl status slurmctld slurmd munge"
echo "  sinfo"
echo "  srun --pty -n1 bash -c 'hostname; nproc; free -m'"
if [ "$GPU_COUNT" -gt 0 ]; then
  echo "  sbatch --gres=gpu:1 /path/to/nvidia-smi-job.sh"
fi

echo "Note: This script is for single-node demo/teaching. For production, review security, accounting, and HA options."
