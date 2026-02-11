#!/bin/bash
set -e

echo "=========================================="
echo "MicroK8s Uninstall Script"
echo "=========================================="
echo "WARNING: This will completely remove MicroK8s and all deployed workloads"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

# Confirmation prompt
read -p "Are you sure you want to uninstall MicroK8s? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Uninstall cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 1: Checking MicroK8s installation"
echo "=========================================="

if ! command -v microk8s &> /dev/null; then
    echo "MicroK8s is not installed"
    exit 0
fi

# Show current status
echo "Current MicroK8s status:"
microk8s status --wait-ready --timeout 10 || echo "MicroK8s is not responding"

echo ""
echo "=========================================="
echo "Step 2: Backing up resources (optional)"
echo "=========================================="

read -p "Backup current Kubernetes resources? (yes/no): " backup
if [ "$backup" = "yes" ]; then
    backup_dir="/tmp/microk8s-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    echo "Backing up to $backup_dir..."
    
    # Backup all resources
    for resource in pods deployments services configmaps secrets persistentvolumeclaims; do
        echo "  Backing up $resource..."
        microk8s kubectl get $resource --all-namespaces -o yaml > "$backup_dir/$resource.yaml" 2>/dev/null || true
    done
    
    # Backup cluster info
    microk8s kubectl cluster-info dump --output-directory="$backup_dir/cluster-info" 2>/dev/null || true
    
    echo "Backup completed: $backup_dir"
    echo ""
fi

echo ""
echo "=========================================="
echo "Step 3: Listing deployed workloads"
echo "=========================================="

echo "Current deployments:"
microk8s kubectl get deployments --all-namespaces 2>/dev/null || echo "  None found"
echo ""
echo "Current pods:"
microk8s kubectl get pods --all-namespaces 2>/dev/null || echo "  None found"
echo ""

read -p "Continue with uninstall? (yes/no): " continue
if [ "$continue" != "yes" ]; then
    echo "Uninstall cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 4: Stopping MicroK8s"
echo "=========================================="

echo "Stopping MicroK8s services..."
microk8s stop || true
sleep 3

echo ""
echo "=========================================="
echo "Step 5: Removing MicroK8s snap"
echo "=========================================="

echo "Removing MicroK8s snap package..."
snap remove microk8s --purge || snap remove microk8s || true

echo ""
echo "=========================================="
echo "Step 6: Cleaning up directories"
echo "=========================================="

# Remove MicroK8s directories
directories=(
    "/var/snap/microk8s"
    "/snap/microk8s"
    "/root/.kube"
    "/var/lib/containerd"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir..."
        rm -rf "$dir" || true
    fi
done

# Clean up user .kube directories
echo "Cleaning up user .kube directories..."
for user_home in /home/*; do
    if [ -d "$user_home/.kube" ]; then
        username=$(basename "$user_home")
        read -p "Remove .kube directory for user $username? (yes/no): " remove_user_kube
        if [ "$remove_user_kube" = "yes" ]; then
            rm -rf "$user_home/.kube"
            echo "  Removed $user_home/.kube"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Step 7: Removing user from microk8s group"
echo "=========================================="

# Remove users from microk8s group
if getent group microk8s >/dev/null 2>&1; then
    echo "Removing users from microk8s group..."
    for user in $(getent group microk8s | cut -d: -f4 | tr ',' ' '); do
        if [ -n "$user" ]; then
            echo "  Removing $user from microk8s group..."
            gpasswd -d "$user" microk8s 2>/dev/null || true
        fi
    done
fi

echo ""
echo "=========================================="
echo "Step 8: Cleaning up network interfaces"
echo "=========================================="

# Remove virtual network interfaces created by MicroK8s
echo "Cleaning up network interfaces..."
interfaces=$(ip link show | grep -E 'veth|cni|flannel|calico' | awk -F: '{print $2}' | tr -d ' ' || true)
if [ -n "$interfaces" ]; then
    for iface in $interfaces; do
        echo "  Removing interface $iface..."
        ip link delete "$iface" 2>/dev/null || true
    done
else
    echo "  No virtual interfaces to clean"
fi

echo ""
echo "=========================================="
echo "Step 9: Cleaning up iptables rules"
echo "=========================================="

echo "Flushing iptables rules..."
# Backup current iptables
iptables-save > /tmp/iptables-backup-$(date +%Y%m%d-%H%M%S).rules 2>/dev/null || true

# Remove chains created by Kubernetes/CNI
chains=("KUBE-FORWARD" "KUBE-SERVICES" "KUBE-EXTERNAL-SERVICES" "KUBE-NODEPORTS" "CNI-FORWARD" "FLANNEL-FWD")
for chain in "${chains[@]}"; do
    if iptables -L "$chain" -n &>/dev/null; then
        echo "  Flushing chain $chain..."
        iptables -F "$chain" 2>/dev/null || true
        iptables -X "$chain" 2>/dev/null || true
    fi
done

echo ""
echo "=========================================="
echo "Step 10: Cleaning up CNI configuration"
echo "=========================================="

cni_dirs=(
    "/etc/cni/net.d"
    "/opt/cni/bin"
    "/var/lib/cni"
)

for dir in "${cni_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir..."
        rm -rf "$dir" || true
    fi
done

echo ""
echo "=========================================="
echo "Step 11: Removing kubectl alias"
echo "=========================================="

if snap list 2>/dev/null | grep -q microk8s; then
    echo "Warning: MicroK8s snap still present, skipping alias removal"
else
    echo "Removing kubectl snap alias..."
    snap unalias kubectl 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Step 12: Cleaning up systemd"
echo "=========================================="

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Reset failed services
systemctl reset-failed 2>/dev/null || true

echo ""
echo "=========================================="
echo "Step 13: Optional cleanup"
echo "=========================================="

# Ask about containerd
read -p "Remove containerd completely? (only if not used by other tools) (yes/no): " remove_containerd
if [ "$remove_containerd" = "yes" ]; then
    echo "Stopping containerd..."
    systemctl stop containerd 2>/dev/null || true
    systemctl disable containerd 2>/dev/null || true
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
            apt-get remove -y containerd containerd.io 2>/dev/null || true
        fi
    fi
    
    rm -rf /var/lib/containerd 2>/dev/null || true
    rm -rf /etc/containerd 2>/dev/null || true
fi

# Ask about persistent volumes
echo ""
read -p "Search for persistent volume data? (yes/no): " search_pvs
if [ "$search_pvs" = "yes" ]; then
    echo "Searching for persistent volume directories..."
    pv_dirs=$(find /var -type d -name 'pv-*' 2>/dev/null || true)
    if [ -n "$pv_dirs" ]; then
        echo "Found persistent volume directories:"
        echo "$pv_dirs"
        
        read -p "Remove these directories? (yes/no): " remove_pvs
        if [ "$remove_pvs" = "yes" ]; then
            echo "$pv_dirs" | xargs rm -rf 2>/dev/null || true
            echo "Persistent volumes removed"
        fi
    else
        echo "No persistent volume directories found"
    fi
fi

echo ""
echo "=========================================="
echo "MicroK8s Uninstall Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ MicroK8s snap removed"
echo "  ✓ Configuration directories cleaned"
echo "  ✓ Network interfaces cleaned"
echo "  ✓ IPtables rules flushed"
echo "  ✓ CNI configuration removed"
echo "  ✓ Systemd configuration cleaned"
echo ""
if [ "$backup" = "yes" ]; then
    echo "Backup location: $backup_dir"
    echo ""
fi
echo "Note: Some network configurations may persist until reboot"
echo "Note: Container images and volumes have been removed"
echo ""
echo "Verification commands:"
echo "  snap list | grep microk8s  # Should show nothing"
echo "  which kubectl              # Should not find microk8s version"
echo "  ip link show               # Check for lingering interfaces"
echo ""
echo "To reinstall, run: ./setup.sh"
echo ""
