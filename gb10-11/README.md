# Lesson 11: Enterprise Use Cases

## CUDA Accelerated ffmpeg

### HWACCEL Transcode with Scaling

https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html

This allows you to load a 4k video and scale it down to multiple resolutions at the same time. The 4k video is stored in VRAM and transcoded simultaneously to lower resolutions.

1:N HWACCEL Transcode with Scaling
The following command reads file input.mp4 and transcodes it to two different H.264 videos at various output resolutions and bit rates. Note that while using the GPU video encoder and decoder, this command also uses the scaling filter (scale_npp) in FFmpeg for scaling the decoded video output into multiple desired resolutions. Doing this ensures that the memory transfers (system memory to video memory and vice versa) are eliminated, and that transcoding is performed with the highest possible performance on the GPU hardware.

Input: input.mp4

Outputs: 1080p, 720p (audio same as input)

```bash
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4
-vf scale_npp=1920:1080 -c:a copy -c:v h264_nvenc -b:v 5M output1.mp4
-vf scale_npp=1280:720 -c:a copy -c:v h264_nvenc -b:v 8M output2.mp4
```

Simple enough right, not so much. The difficult part is compiling ffmpeg with the options to support this. I build a custom container to handle this as there isn't one publicly available. 

```bash
cd gb10-11/ffmpeg-nvec
docker compose up

docker exec -it ffmpeg-nvenc /bin/bash
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i 4k_sample.mkv -vf scale_cuda=1280:720 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
```

### VMAF

https://developer.nvidia.com/blog/calculating-video-quality-using-nvidia-gpus-and-vmaf-cuda/

VMAF evaluates video quality using key elementary metrics from a reference and a distorted image:

- Visual information fidelity (VIF): Quantifies the preservation of original content, reflecting perceived information loss.
- Additive distortion measurement (ADM): Assesses structural changes and texture degradation. It is notably sensitive to additive distortions such as noise.
- Motion features: Crucial for appraising motion-rendering quality in dynamic scenes. 

## The Billion Row Challenge (RAPIDS / cuDF)

Overview
- Goal: show how GPU acceleration (RAPIDS/cuDF) can replace large CPU clusters for big-data ETL and analytics by processing "billion-row" scale datasets interactively on a GB10.
- Demonstration dataset: NYC Taxi trip data (multiple CSVs/parquet files). The lesson runs an identical workflow on CPU (pandas) and GPU (cuDF) to compare run-time and memory behavior.

Why GB10
- A GB10 with large GPU memory allows loading entire large datasets into GPU memory (e.g., 128 GB VRAM), removing the need for distributed clusters and demonstrating a dramatic speedup.

Prerequisites
- NVIDIA drivers + CUDA installed and working (`nvidia-smi`).
- Conda or Miniconda installed on the host.

Quick setup (recommended: `mamba` + `conda` channels)
1. Check CUDA version:

```bash
nvidia-smi
# Note the CUDA driver version and choose a matching RAPIDS cudatoolkit.
```

2. Install `mamba` and create an environment (replace `CUDATOOLKIT` with a version that matches your system, e.g. `12.1`):

```bash
conda install -n base -c conda-forge mamba -y
mamba create -n rapids -c rapidsai -c nvidia -c conda-forge \
	rapids python=3.10 cudatoolkit=CUDATOOLKIT -y
conda activate rapids
```

Note: RAPIDS releases must match your CUDA version. If unsure, see RAPIDS install docs: https://rapids.ai/start.html

Download dataset (example: place multiple month CSVs into `data/`)

```bash
mkdir -p data
# Download a few months of NYC taxi CSVs into data/ (example source)
# e.g. https://s3.amazonaws.com/nyc-tlc/trip+data/ or other public parquet sources
# For a fast experiment, download 3-12 months to create a large multi-GB dataset.
```

Example scripts

1) `gpu_run.py` — RAPIDS/cuDF (GPU)

```python
import time
import cudf
import glob

start = time.perf_counter()
df = cudf.read_csv(sorted(glob.glob('data/*.csv')))
# simple example transform + aggregation
df['tpep_pickup_datetime'] = cudf.to_datetime(df['tpep_pickup_datetime'])
df = df[df['passenger_count'] > 0]
agg = df.groupby('PULocationID').agg({'trip_distance': 'mean', 'total_amount': 'sum'})
print(agg.head())
print('Elapsed (GPU):', time.perf_counter() - start)
```

2) `cpu_run.py` — pandas (CPU)

```python
import time
import pandas as pd
import glob

start = time.perf_counter()
df = pd.concat((pd.read_csv(f) for f in sorted(glob.glob('data/*.csv'))), ignore_index=True)
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df = df[df['passenger_count'] > 0]
agg = df.groupby('PULocationID').agg({'trip_distance': 'mean', 'total_amount': 'sum'})
print(agg.head())
print('Elapsed (CPU):', time.perf_counter() - start)
```

Steps to run the lesson
- Ensure `data/` contains the CSV(s).
- With the `rapids` conda env active, run the GPU script:

```bash
python gpu_run.py
```

- In a separate conda env (or after installing `pandas` in the same env), run the CPU script:

```bash
conda create -n pandas-env python=3.10 pandas -y
conda activate pandas-env
python cpu_run.py
```

What to highlight for students
- Show GPU memory usage with `nvidia-smi` while `gpu_run.py` runs and note that the entire dataset can be resident in GPU memory.
- Compare the printed elapsed times: expect the GPU run to be dramatically faster for large datasets and many-row operations.
- Discuss trade-offs: RAPIDS API differences vs pandas, IO formats (Parquet often preferable), and ecosystem (Dask + cuDF for out-of-core / multi-GPU).

Optional extensions
- Convert CSVs to Parquet and re-run (Parquet improves IO and speeds up GPU reads).
- Use Dask-cuDF to scale across multiple GPUs or a small cluster.

Expected "Aha!" moment
- Students will observe a large wall-clock speedup (e.g., minutes → seconds) for complex joins/aggregations when run on the GB10 GPU versus CPU, demonstrating the practical value of GPU-accelerated data engineering.



## Cybersecurity: Password Entropy & Auditing (Hashcat)

Overview
- Goal: teach password-entropy concepts and practical auditing by creating local test hashes and measuring how quickly the GB10 can recover them with `hashcat`.
- Lesson: "Why Length Matters" — show how short (6-char) passwords can be recovered in milliseconds while longer (11+ char) passwords are infeasible to brute-force.

Why GB10
- The GB10's architecture delivers very high integer-hash throughput, producing very large `hashes/sec` numbers in `hashcat` benchmarks — a striking demo for students.

Ethics & safety
- Run experiments only on hashes you create locally for teaching purposes. Do not attempt to crack hashes that you do not own or otherwise lack explicit authorization to test.

Prerequisites
- `hashcat` installed (system package or release from https://hashcat.net/hashcat/).
- A local wordlist such as `/usr/share/wordlists/rockyou.txt` (installable from `wordlists` packages) or a custom list.

Create test hashes (local only)
- MD5 (example: password `password`):

```bash
echo -n 'password' | md5sum | awk '{print $1}' > hashes_md5.txt
# hashes_md5.txt now contains: 5f4dcc3b5aa765d61d8327deb882cf99
```

- SHA1 (example):

```bash
echo -n 'password' | sha1sum | awk '{print $1}' > hashes_sha1.txt
```

- bcrypt (example - Python) — hashcat mode supports full bcrypt hashes; generate them locally:

```bash
python - <<'PY' > hashes_bcrypt.txt
import bcrypt
print(bcrypt.hashpw(b'password123', bcrypt.gensalt()).decode())
PY
```

Basic `hashcat` examples (local test hashes)
- Benchmark hashcat to see native throughput for your GB10 (no target hashes):

```bash
hashcat -b
```

- Crack an MD5 hash using a wordlist (mode `-m 0`, attack `-a 0`):

```bash
hashcat -m 0 -a 0 hashes_md5.txt /usr/share/wordlists/rockyou.txt --potfile-path=hashcat.pot
```

- Bruteforce a 6-character lowercase password (mask attack):

```bash
# ?l = lowercase, mask ?l?l?l?l?l?l is 6 chars
hashcat -m 0 -a 3 hashes_md5.txt ?l?l?l?l?l?l
```

- Attempt an 11-character mixed-case+digits password (note: infeasible for full brute force without targeted rules):

```bash
# Example mask for 11 chars: ?a?a?a?a?a?a?a?a?a?a?a (very large search space)
hashcat -m 0 -a 3 hashes_md5.txt ?a?a?a?a?a?a?a?a?a?a?a
```

Teaching notes and demo flow
- Start with `hashcat -b` to show hashes/sec baseline for MD5/SHA1 on the GB10.
- Run the 6-char mask attack and show near-instant recovery for typical weak passwords.
- Show the exponential jump in time/space required when increasing length (try 6 → 8 → 11) and discuss why length beats complexity for entropy.
- Compare wordlist attacks vs mask/bruteforce and discuss targeted rules (hybrid attacks) as practical red-team techniques.

What to highlight for students
- Use `watch -n 0.5 nvidia-smi` during GPU runs to show GPU utilization and temperature.
- Point out `hashcat` statistics printed on-screen: `Progress`, `Recovered`, `Rejected` and `Hashes/sec`.
- Discuss defensive takeaways: use long, unique passphrases and multi-factor authentication; educate about password managers.

Optional exercises
- Build custom wordlists from leaked datasets (local, educational only) and compare success rates.
- Use `--increment` mode or tuned masks to demonstrate how attackers optimize search strategies.

Expected "Aha!" moment
- Watch the `hashes/sec` counter reach very large values on the GB10 for fast hashing algorithms, and contrast that with the infeasibility of brute-forcing long passphrases.

## Blender

```bash
sudo apt install blender
```

https://www.blender.org/download/demo-files/

## FreeCAD

```bash
sudo apt install freecad
```