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

echo "Resetting MicroK8s (unmounting storage and clearing state)..."
microk8s reset --destroy-storage || true
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
interfaces=$(ip link show | grep -E 'veth|cni|flannel|calico|vxlan' | awk -F: '{print $2}' | tr -d ' ' || true)
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

echo "Step 9: Surgical cleanup of iptables (Keeping Docker rules)..."

echo "  Backup iptables to /tmp..."
iptables-save > /tmp/iptables-backup-$(date +%Y%m%d-%H%M%S).rules 2>/dev/null || true
iptables-legacy-save > /tmp/iptables-legacy-backup-$(date +%Y%m%d-%H%M%S).rules 2>/dev/null || true

echo "Step 9: Surgical cleanup of MicroK8s networking..."

# 1. Clean the 'nf_tables' (The ones you just dumped)
# We only want to remove the specific lines mentioning 10.1.0.0/16
echo "  Removing MicroK8s pod rules from nf_tables..."
iptables -D FORWARD -s 10.1.0.0/16 -m comment --comment "generated for MicroK8s pods" -j ACCEPT 2>/dev/null || true
iptables -D FORWARD -d 10.1.0.0/16 -m comment --comment "generated for MicroK8s pods" -j ACCEPT 2>/dev/null || true

# 2. Clean the 'legacy' tables (Where the hidden KUBE/CNI chains live)
# This is the "ghost" firewall that MicroK8s actually uses
echo "  Cleaning up legacy Kubernetes/CNI chains..."
# We use the legacy-specific command to flush and delete
K8S_CHAINS=$(iptables-legacy-save | grep '^:' | cut -d' ' -f1 | cut -d':' -f2 | grep -E 'KUBE-|CNI-|FLANNEL-' || true)

for chain in $K8S_CHAINS; do
    echo "    Cleaning legacy chain: $chain"
    iptables-legacy -F "$chain" 2>/dev/null || true
    iptables-legacy -X "$chain" 2>/dev/null || true
done

# 3. Handle the NAT table in legacy mode (NodePorts)
K8S_NAT_CHAINS=$(iptables-legacy -t nat -S 2>/dev/null | grep -E 'KUBE-|CNI-' | grep '^:' | cut -d' ' -f2 || true)
for chain in $K8S_NAT_CHAINS; do
    iptables-legacy -t nat -F "$chain" 2>/dev/null || true
    iptables-legacy -t nat -X "$chain" 2>/dev/null || true
done

echo "  ✓ Kubernetes rules removed. Docker rules preserved."

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
echo "MicroK8s Uninstall Complete!"
echo "=========================================="
echo ""

