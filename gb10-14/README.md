Lesson 14: Dual GB10

# Dual GB10 Setup

* This is designed around 1 DAC cable connected to port 1 of each node. We're using the 192.168.10.x network.

## Configure Network
List infiniband interfaces and make note of which is UP
```ibdev2netdev```

```
# Node 1
sudo tee /etc/netplan/40-cx7.yaml > /dev/null EOF
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    # ----------------------------------------------------
    # Primary Interfaces (Domain 0000) - Assign IPs Here
    # ----------------------------------------------------
    # Physical Port 1 (Network 1)
    enp1s0f0np0:
      dhcp4: no
      addresses:
        - 192.168.10.10/24
      mtu: 9000
      link-local: []

    # Physical Port 2 (Network 2)
    enp1s0f1np1:
      dhcp4: no
      addresses:
        - 192.168.20.10/24
      mtu: 9000
      link-local: []

    # ----------------------------------------------------
    # Socket Direct Interfaces (Domain 0002) - Leave Empty
    # ----------------------------------------------------
    # Handled automatically by Mellanox hardware/driver
    enP2p1s0f0np0:
      dhcp4: no
      mtu: 9000
      link-local: []

    enP2p1s0f1np1:
      dhcp4: no
      mtu: 9000
      link-local: []
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```
```
# Node 2
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    # ----------------------------------------------------
    # Primary Interfaces (Domain 0000) - Assign IPs Here
    # ----------------------------------------------------
    # Physical Port 1 (Network 1)
    enp1s0f0np0:
      dhcp4: no
      addresses:
        - 192.168.10.11/24
      mtu: 9000
      link-local: []

    # Physical Port 2 (Network 2)
    enp1s0f1np1:
      dhcp4: no
      addresses:
        - 192.168.20.11/24
      mtu: 9000
      link-local: []

    # ----------------------------------------------------
    # Socket Direct Interfaces (Domain 0002) - Leave Empty
    # ----------------------------------------------------
    # Handled automatically by Mellanox hardware/driver
    enP2p1s0f0np0:
      dhcp4: no
      mtu: 9000
      link-local: []

    enP2p1s0f1np1:
      dhcp4: no
      mtu: 9000
      link-local: []
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

## Node Setup
Enable bi-directional passwordless SSH
```
ssh-copy-id -i ~/.ssh/id_ed25519.pub $USER@192.168.0.30
ssh-copy-id -i ~/.ssh/id_ed25519.pub $USER@192.168.0.31
```
Enable passworless sudo
```
sudo tee /etc/sudoers.d/trevor > /dev/null <<EOF 
trevor ALL=(ALL) NOPASSWD:ALL
EOF
```

## Ansible
I provided a set of roles to help with some of the setup. It's not 100% but it's a place to start.

```
pip install virtualenv
mkdir ~/venv
python -m virtualenv ~/venv/ansible-dell
source ~/venv/ansible-dell/bin/activate

pip install ansible

cd ansible

# Edit the `gb10.yaml` file, review the roles and uncomment the ones you need.
ansible-playbook gb10.yaml
```

## EarlyOOM
Used to prevent kernel panics from Out of Memory events. Kills llm process if <2% available RAM remaining.
```
sudo apt install earlyoom
sudo vi /etc/default/earlyoom
# -m 2    : trigger at 2% available memory (~2.5 GB on 128 GB DGX Spark)
# -s 80   : swap backstop (earlyoom v1.7 uses AND logic — triggers when RAM < 2% AND swap < 80%)
# --prefer: processes to kill first on OOM (inference workloads)
# --avoid : processes to protect from OOM kill
EARLYOOM_ARGS="-m 2 -s 80 --prefer '(vllm|VLLM|sglang|llama-server|llama-cli|trtllm|tritonserver|ray|python3|python)' --avoid '(systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager)'"

vi ~/.bash_rc
alias earlyoom='journalctl -u earlyoom -f'
```

You can also use and provided Ansible playbook `ansible-playbook gb10.yaml --tag earlyoom`

Monitor the earlyoom logs with `journalctl -u earlyoom -f` or if installed from Ansible there will be an `earlyoom` bash alias.

## Configure NCCL (Run on Both Nodes)

I'm not 100% sure this is necessary but if you want to test NCCL directly on the GB10 here it is.

```
# Install MPI (OpenMPI)
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev

# Install dependencies and build NCCL
git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

# Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

# Install NCCL test suite
git clone https://github.com/NVIDIA/nccl-tests.git  ~/nccl-tests/
cd ~/nccl-tests
make MPI=1
```
### Run a NCCL Test (On Head Node)
```
# Set network interface environment variables (use your active interface)
export UCX_NET_DEVICES=enp1s0f0np0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export OMPI_MCA_btl_tcp_if_include=enp1s0f0np0

# Run the all_gather performance test across both nodes
mpirun -np 2 -H 192.168.10.10:1,192.168.10.11:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

If this test hangs ensure you have SSH key authentication setup properly by trying ssh 192.168.10.11 from the head node. You shouldn't be prompted for a password. Also be sure you ran Configure NCCL (Run on Both Nodes)

Output
```
# nccl-tests version 2.18.5 nccl-headers=22809 nccl-library=22809
# Collective test starting: all_gather_perf
# nThread 1 nGpus 1 minBytes 17179869184 maxBytes 17179869184 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  37755 on promaxgb10-4c75 device  0 [000f:01:00] NVIDIA GB10
#  Rank  1 Group  0 Pid  30808 on promaxgb10-4c99 device  0 [000f:01:00] NVIDIA GB10
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
 17179869184    2147483648     float    none      -1   374425   45.88   22.94       0   373792   45.96   22.98       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.9611
#
# Collective test concluded: all_gather_perf
#
```

### Infiniband Benchmark

Spark 2:
```
ibdev2netdev
ib_write_bw -d rocep1s0f0 --report_gbits -q 4 -R --force-link IB

************************************
* Waiting for client to connect... *
************************************
```
Spark 1:

```
ib_write_bw 192.168.10.11 -d rocep1s0f0 --report_gbits -q 4 -R --force-link IB

---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : rocep1s0f0
 Number of qps   : 4            Transport type : IB
 Connection type : RC           Using SRQ      : OFF
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 1
 Mtu             : 1024[B]
 Link type       : IB
 Max inline data : 0[B]
 rdma_cm QPs     : ON
 Data ex. method : rdma_cm
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x066e PSN 0xf840
 local address: LID 0000 QPN 0x066f PSN 0x3a9d2a
 local address: LID 0000 QPN 0x0670 PSN 0x52282c
 local address: LID 0000 QPN 0x0671 PSN 0xc9420b
 remote address: LID 0000 QPN 0x012f PSN 0x3362c8
 remote address: LID 0000 QPN 0x0130 PSN 0xc82912
 remote address: LID 0000 QPN 0x0131 PSN 0x2929f4
 remote address: LID 0000 QPN 0x0132 PSN 0x4eb433
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 65536      20000            111.62             111.61             0.212877
---------------------------------------------------------------------------------------
```

Latency
```
ib_write_lat -d rocep1s0f0 --report_gbits -R --force-link IB
ib_write_lat 192.168.10.11 -d rocep1s0f0 --report_gbits -R --force-link IB
```

## Llama-Benchy LLM Benchmark
https://github.com/eugr/llama-benchy

```
# Install UV (Like python-pip but better)
curl -LsSf https://astral.sh/uv/install.sh | sh

uvx llama-benchy --base-url http://localhost:8090/v1 --model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 --enable-prefix-caching --latency-mode generation --depth 32768 --concurrency 1 2 4 --runs 1
```

## Model Runners

Each of these can be used with a single GB10 or a multi-node cluster

### eugr/spark-vllm-docker

https://github.com/eugr/spark-vllm-docker
```
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker

# Automatically detect nodes and 200GB network
./build-and-copy.sh -c

# Build with specific git reference
./build-and-copy.sh --rebuild-vllm --vllm-ref ace95c9cf 
./build-and-copy.sh --rebuild-vllm --vllm-ref release/v0.22.1

# Run recipe (from the ./recipes directory)
./run-recipe.sh gemma4-26b-a4b --setup
```

### eugr/spark-vllm-docker recipe for deepseek-ai/DeepSeek-V4-Flash-0731

Until this PR [#304](https://github.com/eugr/spark-vllm-docker/pull/304) gets merged you have to use the fork that contains the recipe and patches to use DFlash speculative decoding.

```
cd ~/git
git clone -b deepseek-v4-flash-dspark-perf-recipe https://github.com/vedcsolution/spark-vllm-docker

# We need to copy in the custom recipe
cp ~/git/gb10-training/gb10-14/deepseek-v4-flash-dspark-0731.yaml ~/git/spark-vllm-docker/recipes/
```

This is going to take 30-40 minutes on the first run. It has to compile everything from source then copy the docker image to the worker node. Subsequent runs are much faster.

```
cd ~/git/spark-vllm-docker
./build-and-copy.sh --vllm-ref releases/v0.25.1 --rebuild-vllm -c 192.168.10.11
```

This is going to use up almost all of the RAM available so be sure to stop any other containers that are using GPU VRAM like ollama. You can monitor available RAM with the `earlyoom` tool or `free -m` and look for `available`.

If you've made it this far, cross your fingers and send it. 
```
./run-recipe.sh deepseek-v4-flash-dspark-0731
```

### sparkun

https://github.com/spark-arena/sparkrun
https://spark-arena.com 
```
# This will install sparkrun and setup your cluster for you
uvx sparkrun setup 

# Run model from spark-arena 
sparkrun run @eugr/step-3.7-flash-nvfp4 --port 8090

# Run from local file
sparkrun run ./step-3.7-flash-nvfp4.yaml --port 8090

sparkrun stop ./step-3.7-flash-nvfp4.yaml --port 8090
sparkrun show ./step-3.7-flash-nvfp4.yaml
```

### gb10-codegen

https://github.com/TrevorSquillario/gb10-codegen

```
git clone https://github.com/TrevorSquillario/gb10-codegen.git
cd gb10-codegen
```

#### Dual GB10

```
cd vllm-ray

# Download model to ~/.cache/huggingface and copy to node-2 using 200GB network
./hf_download.sh -c 192.168.10.11 google/gemma-4-26B-A4B-it

# Start vllm ray cluster
./start_cluster.sh <node-2 IP> <model listed in entrypoint.sh> <vLLM docker container tag. Ex: vllm/vllm-openai:*v0.22.1*>
./start_cluster.sh 192.168.10.11 gemma4 v0.22.1
```

vLLM serve commands for each model is located in `entrypoint.sh`. The `gemma4` in the `./start_cluster` command is in the `case` statement.

#### Single Node
```
docker compose --profile gemma4-nvfp4 up
```