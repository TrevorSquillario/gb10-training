#!/bin/bash
set -e

echo "=========================================="
echo "MicroK8s Setup for GPU Workloads"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

# Install microk8s
echo "Installing MicroK8s..."
snap install microk8s --classic --channel=1.35/stable

# Determine target user (prefer the invoking sudo user)
TARGET_USER="${SUDO_USER:-$USER}"
HOME_DIR="$(getent passwd "$TARGET_USER" | cut -d: -f6 || true)"
if [ -z "$HOME_DIR" ]; then
    # Fallbacks
    if [ "$TARGET_USER" = "root" ]; then
        HOME_DIR="/root"
    else
        HOME_DIR="/home/$TARGET_USER"
    fi
fi

if id -u "$TARGET_USER" >/dev/null 2>&1; then
    echo "Adding user $TARGET_USER to microk8s group..."
    usermod -a -G microk8s "$TARGET_USER" || true
else
    echo "Warning: user $TARGET_USER not found; skipping group addition"
fi

# Ensure kube config directory exists and set ownership
KUBE_DIR="$HOME_DIR/.kube"
mkdir -p "$KUBE_DIR"
chown -R "$TARGET_USER:$TARGET_USER" "$KUBE_DIR" || true

# Wait for microk8s to be ready
echo "Waiting for MicroK8s to be ready..."
microk8s status --wait-ready

# Enable required addons
echo "Enabling required addons..."
microk8s enable dns
microk8s enable helm3
microk8s enable storage

# Setup kubectl alias
echo "Setting up kubectl alias..."
snap alias microk8s.kubectl kubectl

# Generate kubeconfig for target user
echo "Generating kubeconfig for $TARGET_USER..."
microk8s config > "$KUBE_DIR/config"
chown "$TARGET_USER:$TARGET_USER" "$KUBE_DIR/config" || true

echo ""
echo "=========================================="
echo "MicroK8s Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in for group changes to take effect"
echo "2. Or run: newgrp microk8s"
echo "3. Verify with: kubectl get nodes"
echo "4. Install GPU Operator: kubectl apply -f ../kubernetes/gpu-operator.yaml"
echo ""
