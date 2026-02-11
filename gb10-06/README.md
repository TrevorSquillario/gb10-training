# Lesson 6: GPU Container Orchestration with Kubernetes and SLURM

This lesson covers container orchestration using two popular frameworks:
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
microk8s status
```

#### Install GPU Operator

Reference: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html#microk8s

```bash
microk8s helm3 repo add nvidia https://helm.ngc.nvidia.com/nvidia
microk8s helm3 repo update

microk8s helm install gpu-operator -n gpu-operator --create-namespace \
  nvidia/gpu-operator $HELM_OPTIONS \
    --version=v25.10.1 \
    --set toolkit.env[0].name=CONTAINERD_CONFIG \
    --set toolkit.env[0].value=/var/snap/microk8s/current/args/containerd-template.toml \
    --set toolkit.env[1].name=CONTAINERD_SOCKET \
    --set toolkit.env[1].value=/var/snap/microk8s/common/run/containerd.sock \
    --set toolkit.env[2].name=RUNTIME_CONFIG_SOURCE \
    --set-string toolkit.env[2].value=file=/var/snap/microk8s/current/args/containerd.toml

# Verify
microk8s kubectl -n gpu-operator get pods
microk8s kubectl -n gpu-operator logs -l app=nvidia-device-plugin-daemonset
microk8s kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
microk8s kubectl -n gpu-operator describe pod nvidia-device-plugin-daemonset-mmvrc
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

#### Deploy nvidia-smi test pod to check if the container can see the GPU
```bash
kubectl apply -f ../kubernetes/nvidia-smi.yaml
kubectl logs pod/nvidia-smi

# Remove the pod
kubectl delete -f ../kubernetes/nvidia-smi.yaml
```

#### Deploy test vLLM cluster (2 replicas sharing GPU)
```bash
# Make your models available at /mnt/models for Kubernetes HostPath mounting
sudo ln -s /home/$USER/gb10/models /mnt/models

# Create the deployment
kubectl apply -f ../kubernetes/vllm-deployment.yaml
# If you need to restart the deployment
# kubectl rollout restart deployment  vllm-cluster

kubectl get deployment  vllm-cluster -n default
kubectl rollout status deployment/vllm-cluster -n default
kubectl get pods -l app=vllm-demo -n default -o wide

# View logs for all pods in deployment
kubectl logs -l app=vllm-demo -n default 

# Test vLLM service (Wait till you see the message in the logs: "INFO:     Application startup complete.")
curl -X POST http://localhost:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'

```

#### Setup Prometheus and Grafana for DCGM and vLLM Metrics
```bash
microk8s enable observability
microk8s kubectl get svc -n observability

# Add servicemonitor endpoints for prometheus to scrape /metric endpoints
kubectl apply -f ../kubernetes/dcgm-exporter-servicemonitor.yaml
kubectl apply -f ../kubernetes/vllm-metrics-servicemonitor.yaml
kubectl get servicemonitor --all-namespaces

# Fix for microk8s kublet path
microk8s kubectl patch ds nvidia-dcgm-exporter -n gpu-operator --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/volumes/0/hostPath/path", "value": "/var/snap/microk8s/common/var/lib/kubelet/pod-resources"}]'
# Verify
microk8s kubectl get ds nvidia-dcgm-exporter -n gpu-operator -o jsonpath='{.spec.template.spec.volumes[?(@.name=="pod-gpu-resources")].hostPath.path}'

# Expose the grafana service on NodePort 32000
microk8s kubectl patch svc kube-prom-stack-grafana -n observability -p '{"spec": {"type": "NodePort", "ports": [{"port": 80, "nodePort": 32000}]}}'

# Verify and test
kubectl get svc -n observability | grep grafana
# Open a browser to http://<gb10-ip>:32000/login
# Default user/pass: admin/prom-operator
```

#### Install the NVIDIA DCGM Exporter and vLLM Dashboard
1. Download the dashboard `json` files from https://github.com/TrevorSquillario/gb10-training/tree/main/gb10-06/grafana
2. Open Grafana
3. Hover over the Dashboards icon on the left and select `+ Import`
4. Click `Upload JSON file` and select `dcgm_exporter_grafana.json` 
5. On the Prometheus dropdown select `Prometheus (default)`. Leave others at defaults.
6. Repeat this for `vllm_grafana.json`
7. Go to Dashboards > General and look for `NVIDIA DCGM Exporter Dashboard` and `vLLM`

#### Cleanup vLLM cluster deployment
```bash
# Delete the deployment (which will remove the pods)
kubectl delete -f ../kubernetes/vllm-deployment.yaml
kubectl delete -f ../kubernetes/vllm-metrics-servicemonitor.yaml
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

