#!/bin/bash
set -e

echo "=========================================="
echo "SLURM Uninstall Script"
echo "=========================================="
echo "WARNING: This will completely remove SLURM and all related data"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

# Confirmation prompt
read -p "Are you sure you want to uninstall SLURM? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Uninstall cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 1: Stopping SLURM services"
echo "=========================================="

# Stop and disable services
services=("slurmd" "slurmctld" "slurmdbd" "munge")
for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        echo "Stopping $service..."
        systemctl stop "$service" || true
    fi
    if systemctl is-enabled --quiet "$service" 2>/dev/null; then
        echo "Disabling $service..."
        systemctl disable "$service" || true
    fi
done

echo ""
echo "=========================================="
echo "Step 2: Cancelling running jobs"
echo "=========================================="

# Try to cancel all jobs before shutdown
if command -v scancel &> /dev/null; then
    echo "Cancelling all running jobs..."
    scancel --state=RUNNING || true
    sleep 2
fi

echo ""
echo "=========================================="
echo "Step 3: Removing SLURM packages"
echo "=========================================="

# Detect OS and remove packages
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS"
    exit 1
fi

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    echo "Removing SLURM packages (Ubuntu/Debian)..."
    apt-get remove -y \
        slurm-wlm \
        slurm-client \
        slurmd \
        slurmctld \
        slurmdbd \
        slurm-wlm-* \
        libslurm* \
        2>/dev/null || true
    
    # Optionally remove munge (prompt first)
    read -p "Remove Munge authentication? (yes/no): " remove_munge
    if [ "$remove_munge" = "yes" ]; then
        apt-get remove -y munge libmunge-dev libmunge2 || true
    fi
    
    # Clean up
    apt-get autoremove -y || true
    apt-get autoclean || true
    
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ]; then
    echo "Removing SLURM packages (CentOS/RHEL/Rocky)..."
    yum remove -y slurm slurm-* || true
    
    read -p "Remove Munge authentication? (yes/no): " remove_munge
    if [ "$remove_munge" = "yes" ]; then
        yum remove -y munge munge-devel || true
    fi
    
    yum autoremove -y || true
else
    echo "Warning: Unsupported OS: $OS"
    echo "Please manually remove SLURM packages"
fi

echo ""
echo "=========================================="
echo "Step 4: Removing SLURM directories"
echo "=========================================="

directories=(
    "/var/spool/slurm"
    "/var/spool/slurm/d"
    "/var/spool/slurm/ctld"
    "/var/log/slurm"
    "/etc/slurm"
    "/var/lib/slurm"
    "/run/slurm"
    "/tmp/slurm"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir..."
        rm -rf "$dir"
    fi
done

echo ""
echo "=========================================="
echo "Step 5: Removing SLURM user and group"
echo "=========================================="

if id -u slurm >/dev/null 2>&1; then
    echo "Removing slurm user..."
    userdel slurm || true
fi

if getent group slurm >/dev/null 2>&1; then
    echo "Removing slurm group..."
    groupdel slurm || true
fi

echo ""
echo "=========================================="
echo "Step 6: Cleaning up systemd"
echo "=========================================="

# Remove systemd service files if they exist
systemd_files=(
    "/etc/systemd/system/slurmd.service"
    "/etc/systemd/system/slurmctld.service"
    "/etc/systemd/system/slurmdbd.service"
    "/lib/systemd/system/slurmd.service"
    "/lib/systemd/system/slurmctld.service"
    "/lib/systemd/system/slurmdbd.service"
)

for file in "${systemd_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file..."
        rm -f "$file"
    fi
done

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "=========================================="
echo "Step 7: Cleaning up environment"
echo "=========================================="

# Remove any SLURM-related environment variables from common profile files
profile_files=(
    "/etc/profile.d/slurm.sh"
    "/etc/environment"
)

for file in "${profile_files[@]}"; do
    if [ -f "$file" ]; then
        if grep -q "SLURM" "$file"; then
            echo "Cleaning SLURM variables from $file..."
            sed -i '/SLURM/d' "$file"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Step 8: Optional cleanup"
echo "=========================================="

# Ask about munge key
if [ -f /etc/munge/munge.key ]; then
    read -p "Remove Munge key? (yes/no): " remove_key
    if [ "$remove_key" = "yes" ]; then
        echo "Removing munge key..."
        rm -f /etc/munge/munge.key
        
        # Stop munge service
        systemctl stop munge || true
        systemctl disable munge || true
    fi
fi

# Ask about job output files
echo ""
read -p "Search and list SLURM job output files? (yes/no): " list_outputs
if [ "$list_outputs" = "yes" ]; then
    echo "Searching for slurm-*.out files..."
    find /home -name "slurm-*.out" -type f 2>/dev/null | head -20
    find /home -name "slurm-*.err" -type f 2>/dev/null | head -20
    
    read -p "Remove all SLURM output files? (yes/no): " remove_outputs
    if [ "$remove_outputs" = "yes" ]; then
        echo "Removing SLURM output files..."
        find /home -name "slurm-*.out" -type f -delete 2>/dev/null || true
        find /home -name "slurm-*.err" -type f -delete 2>/dev/null || true
        echo "Output files removed"
    fi
fi

echo ""
echo "=========================================="
echo "SLURM Uninstall Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ SLURM services stopped and disabled"
echo "  ✓ SLURM packages removed"
echo "  ✓ SLURM directories cleaned"
echo "  ✓ SLURM user removed"
echo "  ✓ Systemd configuration cleaned"
echo ""
echo "Note: Some configuration files may remain in user home directories"
echo "Note: Munge may still be installed if you chose to keep it"
echo ""
echo "Verification commands:"
echo "  systemctl status slurmd     # Should show 'not found'"
echo "  which sbatch                # Should show 'not found'"
echo "  ls -la /etc/slurm          # Should not exist"
echo ""
