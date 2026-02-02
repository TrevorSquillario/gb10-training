# Week 1: Foundational Setup & The Blackwell Advantage

The objective of Week 1 is to move your team from "Hardware Enthusiasts" to "Remote Developers." By the end of this week, every SE will have a professional development environment that feels like they are working on their local laptop, while actually leveraging the petaflop of power inside the GB10.

## 1. Hardware & Architecture Overview

Before touching the keyboard, the team needs to understand the "Why" behind the GB10.

- The Grace Blackwell (GB10) Superchip: Explain the NVLink-C2C (Chip-to-Chip) interconnect. Unlike a standard PC where the CPU and GPU fight over a narrow PCIe bus, the GB10 has 900 GB/s of coherent memory bandwidth.
- Unified Memory (128GB): This is the killer feature. Explain that the CPU and GPU share this pool, allowing them to run 200-billion parameter models that would normally require a full server rack.
- The FP4 Precision: Introduce the 2nd Gen Transformer Engine. Explain that Blackwell is the first architecture to support 4-bit Floating Point (FP4), which doubles inference speed without losing accuracy.

## 2. Hands-on Lab: Remote Development Setup

This lab standardizes the environment using NVIDIA Sync and VS Code.

### Step A: Establishing Connectivity

We will use mDNS (hostname.local) for easy discovery on your 1GbE network.

- Install NVIDIA Sync: Have everyone download and install the NVIDIA Sync tool on their laptops.
- Discovery: Launch the app and "Add Device." Use the unique hostname found on the physical sticker of the GB10 (e.g., `spark-a1b2.local`).
- SSH Key Exchange: NVIDIA Sync will handle the creation of SSH keys so the team never has to type a password again.

### Step B: VS Code Remote-SSH (The Professional Way)

Working in a raw terminal is fine for "Heavy Tech" users, but VS Code Remote-SSH is the equalizer for "Light Tech" users.

- Extension Install: Install the **Remote - SSH** extension in local VS Code.
- The Magic Connection:
  1. Press `F1` -> `Remote-SSH: Connect to Host`
  2. Select your GB10.
  3. VS Code will now install a "Server" on the GB10 (ARM64 version).

Verification: Open the integrated terminal in VS Code and run:

```bash
# Verify connection and GPU/CPU visibility
nvidia-smi
```

If you see the Blackwell GPU and the Grace CPU listed, you are officially connected.

### Step C: Bash Customizations (GitHub-Ready)

To make the 12-week journey easier, we will standardize the shell.

Clone the course repo:

```bash
git clone https://github.com/your-org/gb10-training.git
```

Add these aliases to `~/.bashrc` to help "Light Tech" users navigate:

```bash
# Helpful aliases for the course
alias docs="cd ~/gb10-training"
alias gpustats="watch -n 1 nvidia-smi"
alias dashboard="ssh -L 11000:localhost:11000 user@spark-xxxx.local"
```

---

🌟 Week 1 Challenge: The "First Flight"

Task: Use your newly configured VS Code to create a file named `hello_blackwell.py`. Write a simple script that prints the total available VRAM on the system.

- Heavy Tech (PyTorch):

```python
# Example: hello_blackwell.py (Heavy Tech)
import torch

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"Total VRAM: {total_vram / (1024**3):.2f} GB")
else:
    print("No CUDA GPU detected")
```

- Light Tech (pynvml):

```python
# Example: hello_blackwell.py (Light Tech)
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
mem = nvmlDeviceGetMemoryInfo(handle)
print(f"Total VRAM: {mem.total / (1024**3):.2f} GB")
```

## Resources for Week 1

- Playbook: VS Code on DGX Spark
- Documentation: DGX Spark User Guide (PDF)
- NVIDIA Sync for DGX Spark: This video provides a step-by-step walkthrough of setting up NVIDIA Sync to enable password-less SSH and remote development access, which is the primary goal of your first week.