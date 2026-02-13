# Lesson 6: GPU Container Orchestration with Kubernetes and SLURM

This lesson covers container orchestration using two popular frameworks:
- **Kubernetes (microk8s)** with NVIDIA GPU Operator and time-slicing
- **SLURM** workload manager for HPC environments

## Quick Start

### Kubernetes (microk8s)

#### Install and configure microk8s
```bash
cd ~/git/gb10-training/gb10-06/microk8s
sudo ./install.sh

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

#### Uninstall
```bash
cd ~/home/trevor~/git/gb10-training/gb10-06/microk8s
sudo ./uninstall.sh
```

### SLURM

#### Install
```bash
# Install dependencies
sudo apt-y install environment-modules build-essential cmake git m4 flex bison zlib1g-dev libboost-system-dev libboost-thread-dev 

# Install SLURM
cd slurm
sudo ./install.sh
srun -n1 hostname
```

#### Useful commands
```bash
squeue
scancel 1
sinfo
# Submit background job
sbatch run_train_model.sh
# Submit interactive job
srun run_train_model.sh
```

#### Troubleshooting

***Job stuck in queue***
```bash
sudo pkill -9 slurmctld
sudo pkill -9 slurmd
sudo pkill -9 slurmstepd

# 2. Check if anything is still running (should return empty)
ps aux | grep slurm | grep -v grep

# 3. Wipe the state locations again
sudo rm -rf /var/spool/slurm/ctld/*
sudo rm -rf /var/spool/slurm/d/*

# 4. Restart services
sudo systemctl start slurmctld
sudo systemctl start slurmd
sudo chmod 755 /var/spool/slurm
```

#### Run traditional CFD based job. CPU only.
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

#### Run LLM training job
```bash
wget -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-py313_25.11.1-1-Linux-aarch64.sh
chmod u+x /tmp/Miniconda3-py313_25.11.1-1-Linux-aarch64.sh
sudo bash /tmp/Miniconda3-py313_25.11.1-1-Linux-aarch64.sh -b -p /opt/anaconda/25.11.1-1

# Create environment module (Used with the `module load` command)
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

