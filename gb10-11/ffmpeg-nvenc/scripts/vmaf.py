import subprocess
import os

https://developer.nvidia.com/blog/calculating-video-quality-using-nvidia-gpus-and-vmaf-cuda/

# Configuration
input_file = "4k_sample.mkv"
model_path = "/opt/build/vmaf/model/vmaf_v0.6.1.json" # Update to your path
output_log = "vmaf_scores.json"

# The VMAF-CUDA Pipeline:
# 1. Decode reference [0:v] and distorted [1:v] in GPU
# 2. Scale reference if needed (VMAF requires matching resolutions)
# 3. Run libvmaf_cuda
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
    "-i", input_file,  # The Reference
    
    # Define the encoding ladder (Distorted)
    "-c:v", "av1_nvenc", "-preset", "p4", "-b:v", "6M", 
    "-f", "null", "-" # We output to null for a pure stress test
    
    # The VMAF-CUDA Filtergraph
    # [0:v] is the raw input, [v:0] is the newly encoded stream
    "-filter_complex", 
    f"[0:v]scale_npp=1920:1080[ref];" # Match the 1080p target
    f"[v:0][ref]libvmaf_cuda=model=path={model_path}:log_fmt=json:log_path={output_log}"
]

print("🔥 Launching VMAF-CUDA Stress Test...")
subprocess.run(ffmpeg_cmd)