## Troubleshooting

No space available on disk

https://docs.docker.com/reference/cli/docker/system/prune/

This will remove all the docker images no associated to a running container.

```bash
docker system prune -a
```

OOM (OutofMemory) Errors

It could be that the model is just too large to fit into RAM. If not clear the cache.
```bash
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
```

## How to Reinstall the NVIDIA DGX Operating System
Download the US image from the Dell support site, here: 
https://www.dell.com/support/home/en-us/drivers/driversdetails?driverid=ypgdr&oscode=dgx7&productcode=dell-pro-max-fcm1253-micro 

Burn this image to a USB drive, needs to be at least 16GB.   

For Windows you can use RUFUS, get it here: 
https://rufus.ie/en/#download 

Note you need administrator rights to use RUFUS. 

For Linux use DD to create the USB drive using the following command 

```sudo dd if=/path/to/image.iso of=/dev/sdX bs=4M status=progress oflag=sync```

To boot from the USB drive press <F7> when the Dell logo appears at power on.  Choose the boot device.  Typically something like: 

```UEFI: USB USB Hard Drive, Partition 2.```

Let the installer go through the process, it will fail. 
Accept the prompt and drop to a bash shell, 
Execute the following command to get a listing of the nvme devices in the Pro Max GB10. 

```
fdisk –l | grep nvme 

Disk /dev/nvme0n1: 3.64TiB, 4000078700016, 7814037168 sectors 
```

In this case the main storage device is nvme0n1, so to remove the partitions you will need to use FDISK and remove the partitions. 

You will get a warming saying that the partition is in use.  Ignore and proceed.  You will be at the fdisk prompt. 

```
fdisk /dev/nvme01

Command (m for help)? 
Enter p to print out the current partition table.  Verify that the selected device is /dev/nvme0n1 

root@promaxgb10-4c70:~# fdisk /dev/nvme0n1 
Welcome to fdisk (util-linux 2.39.3). 
Changes will remain in memory only, until you decide to write them. 
Be careful before using the write command. 

This disk is currently in use - repartitioning is probably a bad idea. 
It's recommended to umount all file systems, and swapoff all swap 
partitions on this disk. 

Command (m for help): p 

Disk /dev/nvme0n1: 3.64 TiB, 4000787030016 bytes, 7814037168 sectors 
Disk model: ESL04TBTLCZ-27J4-TYN                     
Units: sectors of 1 * 512 = 512 bytes 
Sector size (logical/physical): 512 bytes / 512 bytes 
I/O size (minimum/optimal): 512 bytes / 512 bytes 
Disklabel type: gpt 
Disk identifier: 2A851E5F-24D8-470D-BB8F-2E5B71ED7030 

Device           Start        End    Sectors  Size Type 
/dev/nvme0n1p1    2048    1050623    1048576  512M EFI System 
/dev/nvme0n1p2 1050624 7814033407 7812982784  3.6T Linux filesystem 

Command (m for help):  

Type d to delete the partitions, continue until all the paritions are deleted. 

Type w to write your changes to disk. 
```

This is absolutely data destructive with no method to recover.  Make sure this is what you need to do. 

## Dual GB10 Setup
### Configure Network
List infiniband interfaces and make note of which is connected
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
        - 192.168.1.10/24
      link-local: []

    # Physical Port 2 (Network 2)
    enp1s0f1np1:
      dhcp4: no
      addresses:
        - 192.168.2.10/24
      link-local: []

    # ----------------------------------------------------
    # Socket Direct Interfaces (Domain 0002) - Leave Empty
    # ----------------------------------------------------
    # Handled automatically by Mellanox hardware/driver
    enP2p1s0f0np0:
      dhcp4: no
      link-local: []

    enP2p1s0f1np1:
      dhcp4: no
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
        - 192.168.1.11/24
      link-local: []

    # Physical Port 2 (Network 2)
    enp1s0f1np1:
      dhcp4: no
      addresses:
        - 192.168.2.11/24
      link-local: []

    # ----------------------------------------------------
    # Socket Direct Interfaces (Domain 0002) - Leave Empty
    # ----------------------------------------------------
    # Handled automatically by Mellanox hardware/driver
    enP2p1s0f0np0:
      dhcp4: no
      link-local: []

    enP2p1s0f1np1:
      dhcp4: no
      link-local: []
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

### Configure NCCL (Run on Both Nodes)
```
# Enable bi-directional passwordless SSH
ssh-copy-id -i ~/.ssh/id_ed25519.pub trevor@192.168.0.30
ssh-copy-id -i ~/.ssh/id_ed25519.pub trevor@192.168.0.33

sudo tee /etc/sudoers.d/trevor > /dev/null <<EOF 
trevor ALL=(ALL) NOPASSWD:ALL
EOF

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
mpirun -np 2 -H 192.168.1.10:1,192.168.1.11:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

### Benchmark

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
ib_write_bw 192.168.1.11 -d rocep1s0f0 --report_gbits -q 4 -R --force-link IB

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
ib_write_lat 192.168.1.11 -d rocep1s0f0 --report_gbits -R --force-link IB
```

## NVIDIA PyTorch Container Comparison

| **Feature**            | **25.02‑py3**                | **25.09‑py3**                        | **25.10‑py3**               |
|------------------------|------------------------------|--------------------------------------|-----------------------------|
| PyTorch Version        | 2.7.0a0                      | 2.9.0a0                              | 2.9.0a0                     |
| CUDA Toolkit           | 12.8                         | 12.9 (Early CUDA 13)                 | 13.0.2                      |
| Python Version         | 3.12.3                       | 3.12.3                               | 3.12.3                      |
| Blackwell Status       | Early Access / Beta          | Optimized                            | Production Stable           |
| RAPIDS Included        | Yes                          | No (Removed)                         | No (Removed)                |

## Simple LLM Benchmark
```bash
cd cd ~/git/gb10-training/appendix
source ~/venv/gb10-training/bin/activate
pip install openai

python llm_benchmark.py
```
You will be presented with 3 options. You can't run these at the same time. Start Ollama choose 1, stop Ollama. Start trtllm, choose 2, stop trtllm, etc. When you're finished...
```bash
```bash
python llm_benchmark.py --report

python ~/git/gb10-training/appendix/llm_benchmark.py --help
usage: llm_benchmark.py [-h] [--report] [--csv CSV] [--model MODEL] [--url URL]

Simple LLM benchmark tool

options:
  -h, --help     show this help message and exit
  --report       Show aggregated report from results CSV and exit
  --csv CSV      Path to results CSV file
  --model MODEL  Override model for all endpoints (optional)
  --url URL      Override URL for all endpoints (optional)

```

## Manually Add Model to Ollama

*Note: 
```The Q8 (8-big) quant I selected for this model requires 243GB of RAM so it's not actually going to run on the GB10. If you want to use this model you'll need to choose the IQ4_XS version```

Start by installing the hf cli and download the model.

```bash
# Ensure our python venv is started
source ~/venv/gb10-training/bin/activate
pip install -U "huggingface_hub[cli]"
# FYI: This model is 268GB. We'll include a specific subdirectory for the the 8-bit quant
hf download unsloth/MiniMax-M2.1-GGUF \
  --include "*Q8_0*" \
  --local-dir ~/gb10/models/MiniMax-M2.1-GGUF_Q8_0
```
For models split into parts Ollama requires that they are merged into one file

```bash
# Run a temporary container to use the llama.cpp tools
docker run --rm \
  -v ~/gb10/models:/models \
  --entrypoint /llm/llama-gguf-split \
  amperecomputingai/llama.cpp:latest \
  --merge \
  /models/MiniMax-M2.1-GGUF_Q8_0/Q8_0/MiniMax-M2.1-Q8_0-00001-of-00005.gguf \
  /models/MiniMax-M2.1-GGUF_Q8_0/MiniMax-M2.1-Q8_0-merged.gguf
```

Create the model definition
```bash
cd ~/gb10/models/MiniMax-M2.1-GGUF_Q8_0
vi Modelfile

FROM ./MiniMax-M2.1-Q8_0-merged.gguf

# This template handles standard chat AND tool calling for MiniMax
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}
{{- if .Tools }}
When you need to use a tool, you must respond in JSON format:
{"name": "function_name", "parameters": {"arg": "value"}}
Available tools:
{{ .Tools }}
{{- end }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ if .Thinking }}<think>
{{ .Thinking }}</think>
{{ end }}{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<think>"
PARAMETER stop "</think>"
PARAMETER temperature 1
PARAMETER num_ctx 32768
```

Create the model using the `ollama` command in the container 
```bash
docker exec -it ollama /bin/bash
# CD to the /models volume mount we specified in the compose.yaml. This corresponds to your ~/gb10/models directory on the host.
cd /models/MiniMax-M2.1-GGUF_Q8_0
ollama create minimax-q8 -f Modelfile

# Verify the model is working with
ollama run minimax-q8
```

## Prometheus /metrics Endpoint for Ollama
https://github.com/NorskHelsenett/ollama-metrics

## Remote Access via Sunshine/Moonight Streaming

This is much better than RDP or VNC. Supports high resolutions and GPU rendering

https://github.com/eelbaz/dgx-spark-headless-sunshine

## Other Self-Hosted Apps Worth Trying
- Frigate https://frigate.video/
- Immich https://immich.app/
- changedetection.io https://github.com/dgtlmoon/changedetection.io
- Paperless-ngx + Paperless-AI
- Self-hosted web based Photoshop https://www.photopea.com