# Lesson 6: GPU Orchestration with Kubernetes and SLURM

This lesson covers GPU orchestration using two popular frameworks:
- **Kubernetes (microk8s)** with NVIDIA GPU Operator and time-slicing
- **SLURM** workload manager for HPC environments

## Quick Start

### Kubernetes (microk8s)

#### Install and configure microk8s
```bash
cd ~/git/gb10-training/gb10-06/microk8s
sudo ./setup.sh

# Logout of your SSH session and back in to refresh your groups

# Verify installation
microk8s kubectl version
microk8s kubectl get nodes
microk8s kubectl get pods -A
```

#### Install GPU Operator
```bash
microk8s helm3 repo add nvidia https://helm.ngc.nvidia.com/nvidia
microk8s helm3 repo update

# https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html#microk8s
microk8s helm install gpu-operator -n gpu-operator --create-namespace \
  nvidia/gpu-operator $HELM_OPTIONS \
    --version=v25.10.1 \
    --set toolkit.env[0].name=CONTAINERD_CONFIG \
    --set toolkit.env[0].value=/var/snap/microk8s/current/args/containerd-template.toml \
    --set toolkit.env[1].name=CONTAINERD_SOCKET \
    --set toolkit.env[1].value=/var/snap/microk8s/common/run/containerd.sock \
    --set toolkit.env[2].name=RUNTIME_CONFIG_SOURCE \
    --set-string toolkit.env[2].value=file=/var/snap/microk8s/current/args/containerd.toml

microk8s kubectl -n gpu-operator get pods
microk8s kubectl -n gpu-operator logs -l app=nvidia-device-plugin-daemonset
microk8s kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
microk8s kubectl -n gpu-operator describe pod nvidia-device-plugin-daemonset-mmvrc
```

#### Remove gpu-operator
```bash
microk8s helm3 uninstall gpu-operator -n gpu-operator
microk8s kubectl delete namespace gpu-operator --ignore-not-found
microk8s kubectl get crds | awk '/nvidia|gpu/ {print $1}' | xargs -r microk8s kubectl delete crd
microk8s kubectl delete clusterrole nvidia-device-plugin --ignore-not-found
microk8s kubectl delete clusterrolebinding nvidia-device-plugin --ignore-not-found
microk8s kubectl delete clusterrole -l app.kubernetes.io/managed-by=gpu-operator --ignore-not-found
microk8s kubectl delete clusterrolebinding -l app.kubernetes.io/managed-by=gpu-operator --ignore-not-found
sudo rm -f /etc/containerd/conf.d/00-nvidia.toml
sudo systemctl restart snap.microk8s.daemon-containerd.service
sudo microk8s stop && sudo microk8s start
```

#### Configure GPU time-slicing
Instructions to apply time-slicing:

1. Apply the time slicing config:
   ```
   kubectl apply -f ../kubernetes/time-slicing-config.yaml
   ```
2. Verify GPU resources are multiplied:
   ```
   kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'
   ```
   
   You should see "4" instead of "1" if you have 1 physical GPU
3. Restart device plugin pods to apply changes:
   ```
   kubectl delete pods -n gpu-operator -l app=nvidia-device-plugin-daemonset
   ```
4. Check logs:
   ```
   kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
   ```

Notes:
- Time-slicing allows 4 pods to share a single GPU
- Each pod thinks it has exclusive access
- GPU memory is NOT isolated - pods share memory
- Performance depends on workload characteristics
- Best for burst workloads or development/testing

#### Deploy nvidia-smi test (4 replicas sharing GPU)
```bash
kubectl apply -f ../kubernetes/nvidia-smi-deployment.yaml
# If you need to restart the deployment
# kubectl rollout restart deployment nvidia-smi-test

kubectl get deployment nvidia-smi-test -n default
kubectl rollout status deployment/nvidia-smi-test -n default
kubectl get pods -l app=nvidia-smi-test -n default -o wide

kubectl logs -l app=nvidia-smi-test -n default --all-containers=true

# or exec into a pod to run nvidia-smi interactively:
kubectl exec -it $(kubectl get pods -l app=nvidia-smi-test -o name | head -1) -- nvidia-smi
```

#### Deploy vLLM inference server
```bash
kubectl apply -f ../kubernetes/vllm-deployment.yaml
```

### SLURM

```bash
# Install SLURM
cd slurm
sudo ./setup.sh

# Submit GPU test job
sbatch nvidia-smi-job.sh

# Check job status
squeue

# View job output
cat slurm-*.out
```

## Learning Objectives

- Understand GPU resource management in Kubernetes
- Configure GPU time-slicing for multi-tenancy
- Deploy AI inference workloads on Kubernetes
- Set up SLURM for HPC GPU scheduling
- Submit and manage GPU jobs in SLURM

## Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with drivers installed
- Root/sudo access
- At least 8GB RAM
- 50GB free disk space

## Architecture Overview

### Kubernetes GPU Architecture
```
┌─────────────────────────────────────────────────┐
│           Kubernetes Cluster                    │
│  ┌──────────────────────────────────────────┐   │
│  │  GPU Operator (manages GPU resources)    │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  Time-Slicing Config (4x multiplier)     │   │
│  └──────────────────────────────────────────┘   │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐        │
│  │ Pod 1 │ │ Pod 2 │ │ Pod 3 │ │ Pod 4 │        │
│  │  GPU  │ │  GPU  │ │  GPU  │ │  GPU  │        │
│  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘        │
│      └──────┬──┴──────┬──────┴─────┘            │
│             │         │                         │
│      ┌──────▼─────────▼──────┐                  │
│      │  Physical GPU (1x)    │                  │
│      └───────────────────────┘                  │
└─────────────────────────────────────────────────┘
```

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

## Comparison: Kubernetes vs SLURM

| Feature | Kubernetes | SLURM |
|---------|-----------|-------|
| **Primary Use** | Microservices, web apps | HPC, batch jobs |
| **GPU Sharing** | Time-slicing, MIG | Exclusive or shared |
| **Scheduling** | Real-time, declarative | Batch queue, priority |
| **Networking** | Service mesh, ingress | InfiniBand, high-speed |
| **State** | Stateless preferred | Stateful jobs common |
| **Auto-scaling** | Built-in HPA/VPA | Manual configuration |
| **Best For** | AI inference, serving | Training, simulations |

