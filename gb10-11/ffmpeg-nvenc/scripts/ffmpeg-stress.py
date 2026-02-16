"""Refactored ffmpeg ladder script.

This script runs a single ffmpeg invocation producing multiple outputs
with CUDA hwaccel and `scale_npp` for scaling. By default it keeps the
two resolutions used in the NVIDIA example: 1920x1080 and 1280x720.

https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html
"""

import argparse
import shutil
import subprocess
import sys

# Keep the resolutions from the NVIDIA example and associated bitrates.
RESOLUTIONS = [
    {"name": "4k",   "res": "3840:2160", "bitrate": "15M"},
    {"name": "1080p", "res": "1920:1080", "bitrate": "6M"},
    {"name": "720p",  "res": "1280:720",  "bitrate": "3M"},
    {"name": "480p",  "res": "854:480",   "bitrate": "1.5M"},
    {"name": "360p",  "res": "640:360",   "bitrate": "800k"}
]


def build_ffmpeg_cmd(input_file, outputs, codec="h264_nvenc", benchmark=False):
    cmd = [
        "ffmpeg", "-y", "-vsync", "0",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", input_file,
    ]

    for out in outputs:
        w, h = out["res"].split(":")
        # Options placed before the output file apply to that output.
        cmd += ["-vf", f"scale_cuda={w}:{h}", "-c:a", "copy", "-c:v", codec, "-b:v", out["bitrate"], out["filename"]]

    if benchmark:
        cmd.append("-benchmark")

    return cmd


def check_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH", file=sys.stderr)
        sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(description="Run ffmpeg ladder with NVIDIA scale_npp and NVENC.")
    p.add_argument("input", help="Input video file")
    p.add_argument("--prefix", default="output", help="Output filename prefix (default: output)")
    p.add_argument("--codec", default="h264_nvenc", help="Encoder to use (default: h264_nvenc)")
    p.add_argument("--benchmark", action="store_true", help="Append -benchmark to ffmpeg")
    return p.parse_args()


def main():
    args = parse_args()
    check_ffmpeg_available()

    outputs = []
    for r in RESOLUTIONS:
        outputs.append({"name": r["name"], "res": r["res"], "bitrate": r["bitrate"], "filename": f"{args.prefix}_{r['name']}.mp4"})

    cmd = build_ffmpeg_cmd(args.input, outputs, codec=args.codec, benchmark=args.benchmark)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()