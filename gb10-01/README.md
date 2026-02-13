# Lesson 1: Foundational Setup & The Blackwell Advantage

The objective of Lesson 1 is to move your team from "Hardware Enthusiasts" to "Remote Developers." By the end of this Lesson, every SE will have a professional development environment that feels like they are working on their local laptop, while actually leveraging the petaflop of power inside the GB10.

## Hardware & Architecture Overview

Before touching the keyboard, the team needs to understand the "Why" behind the GB10.

- The Grace Blackwell (GB10) Superchip: The NVLink-C2C (Chip-to-Chip) interconnect. Unlike a standard PC where the CPU and GPU fight over a narrow PCIe bus, the GB10 has 300 GB/s of coherent memory bandwidth.
- Unified Memory (128GB): This is the killer feature. The CPU and GPU share this pool, allowing them to run 200-billion parameter models that would normally require a full server rack. You essential get 128GB of VRAM but at a slower speed than the GB200s 8TB/s.
- The FP4 Precision: Introduce the 2nd Gen Transformer Engine. Blackwell is the first architecture to support 4-bit Floating Point (FP4), which doubles inference speed without losing accuracy.

## Hands-on Lab: Remote Development Setup

This lab provides several options for establishing connectivity to your GB10.

### First Time Setup

Connect your GB10 and perform initial setup using either the HDMI Port with a USB keyboard and mouse or WiFi hotspot and your phone (for headless setup). 
Reference [Dell Setup Guide](https://www.dell.com/support/kbdoc/en-us/000398800/dell-pro-max-with-gb-10-fcm1253-initial-setup-out-of-the-box-experience) 
- If using WiFi hotspot turn on Airplane mode, then ensure wifi is on and connect to the hostspot listed on the sticker.
- Make note of the IP Address or access your ISP Gateway to view the DHCP Clients

### Connect to the DGX Dashboard

#### Option 1:  Use NVIDIA Sync

- Install NVIDIA Sync: Download (https://build.nvidia.com/spark/connect-to-your-spark/sync) and install the NVIDIA Sync tool on your laptop.
- Discovery: Launch the app and "Add Device." Use the unique hostname found on the physical sticker of the GB10 (e.g., `promaxgb10-a1b2.local`).
- SSH Key Exchange: NVIDIA Sync will handle the creation of SSH keys so the team never has to type a password again.
- SSH Tunnels are setup for direct access to the DGX Dashboard and Jupyter Notebook

#### Option 2: SSH to the IP and start the ssh tunnel

Start SSH tunnel from remote port 11000 to local port 11000 for DGX Dashboard and 11002 for Jupyter Notebook

```bash
ssh -L 11000:localhost:11000 -L 11002:localhost:11002 <username>@<your-gb10-ip>
```

### Verification 

#### Open the terminal SSH to the device and run:

```bash
# Verify connection and GPU/CPU visibility
nvidia-smi
```

If you see the Blackwell GPU and the Grace CPU listed, you are officially connected.

#### Access the DGX Dashboard
[http://localhost:11000](http://localhost:11000)

### Setup Visual Studio Code with Remote SSH to your GB10

1. Install Visual Studio Code from the Company Portal or https://code.visualstudio.com/download
2. Open VSCode and select Terminal in the top bar. This will open a PowerShell terminal. 
3. Select the Extensions tab on the left or hit `Ctrl + Shift + P` and search for Install Extension
4. Search for Remote - SSH and install

```bash
# Get the SSH Public Key from your laptop (Windows)
ssh-keygen
type ~/.ssh/id_rsa.pub

# Get the SSH Public Key from your laptop (Linux)
cat ~/.ssh/*.pub

# Add SSH Key to GB10
ssh <username>@<your-gb10-ip>
vi ~/.ssh/authorized_keys
o
# Right click to paste
wq

# Shouldn't need the but just in case
chmod 600 ~/.ssh/authorized_keys
chown <username>:<username> ~/.ssh/authorized_keys
```

### Download Course Files

Clone the course repo:

```bash
mkdir ~/git
cd ~/git
git clone https://github.com/TrevorSquillario/gb10-training

# If there are changes to the repo and you want to ensure you are using the latest version run
cd ~/git/gb10-training
git pull
```

### Updating your GB10

*There is a BIOS update from factory version

https://www.dell.com/support/kbdoc/en-us/000379162/how-to-upgrade-the-bios-and-the-firmware-on-a-dell-pro-max-with-the-grace-blackwell-system

```bash
sudo apt update
sudo apt upgrade
sudo fwupdmgr refresh
sudo fwupdmgr upgrade
```

### Install useful Linux utilities
```bash
sudo apt install tree nvtop htop
```

### Setup your Python Virtual Environment

```bash
mkdir ~/venv
python3 -m venv ~/venv/gb10-training
source ~/venv/gb10-training/bin/activate
```

### (Optional) Customize `~/.bashrc` by adding commands to the end of the file:

```bash
vi ~/.bashrc

source ~/venv/gb10-training/bin/activate
alias gpustats="watch -n 1 nvidia-smi"
cd ~/git
```

### (Optional) Customize `/etc/inputrc` to enable easy history search with Ctrl + Up/Down

```bash
sudo vi /etc/inputrc
ggdG
```

```bash
# do not bell on tab-completion
set bell-style none

set meta-flag on
set input-meta on
set convert-meta off
set output-meta on

set completion-ignore-case on
# Completed names which are symbolic links to
# directories have a slash appended.
set mark-symlinked-directories on

$if mode=emacs
        # Home/End Fix
        "\e[1~": beginning-of-line
        "\e[4~": end-of-line

        # for linux console and RH/Debian xterm
        "\e[1;5A": history-search-backward
        "\e[1;5B": history-search-forward
        "\e[1;5C": forward-word
        "\e[1;5D": backward-word

        "\e[3~": delete-char
        "\e[2~": quoted-insert

        # for putty
        "\eOA": history-search-backward
        "\eOB": history-search-forward
        "\eOC": forward-word
        "\eOD": backward-word

        # for rxvt
        "\e[8~": end-of-line
        "\eOc": forward-word
        "\eOd": backward-word

        # for non RH/Debian xterm, can't hurt for RH/DEbian xterm
        "\eOH": beginning-of-line
        "\eOF": end-of-line

        # for freebsd console
        "\e[H": beginning-of-line
        "\e[F": end-of-line
$endif
```

```bash
wq!
```

Exit out of your SSH session and start a new one. Type part of a command and press Ctrl + Up to search through history

## Lesson 1 Challenge: The "First Flight"

Task: 

1. Access the DGX Dashboard
2. Start the Jupyter Notebook container
3. Wait for the container to start and logs to finish. You should be given a url like `http://localhost:11002/lab?token=asdfa234asdfb`
4. Open the Jupyter Notebook in your browser
5. Under Notebook click Python 3
6. Copy the following python code into the notebook
7. Click the Run button at the top or press Shift + Enter

```python
import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    
    print(f"--- Standard Summary ---")
    print(f"Device Name: {props.name}")
    print(f"Total Unified Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Multiprocessor Count: {props.multi_processor_count}")
    
    print(f"\n--- All Internal Properties ---")
    # We filter out methods and private attributes to show just the data fields
    for attr in dir(props):
        if not attr.startswith('_'):
            value = getattr(props, attr)
            print(f"{attr}: {value}")
else:
    print("No CUDA GPU detected.")
```