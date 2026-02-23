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
https://www.dell.com/support/kbdoc/en-bh/000382042/how-to-reinstall-the-nvidia-dgx-operating-system-on-dell-pro-max-with-grace-blackwell-systems


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

## Sunshine/Moonight Streaming
https://github.com/seanGSISG/dgx-spark-sunshine-setup

## Other Self-Hosted Apps Worth Trying
- Frigate https://frigate.video/
- Immich https://immich.app/
- changedetection.io https://github.com/dgtlmoon/changedetection.io
- Paperless-ngx + Paperless-AI
- Self-hosted web based Photoshop https://www.photopea.com