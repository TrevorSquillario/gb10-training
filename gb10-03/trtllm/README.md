# TensorRT LLM Server (trtllm-serve) Benchmarking
aiperf profile \
  --model nvidia/Llama-3.3-70B-Instruct-NVFP4 \
  --url http://localhost:8001 \
  --endpoint-type chat \
  --request-rate 1 \
  --request-count 1 \
  --streaming

# Test with curl
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'

# Ollama Benchmarking

hf download unsloth/Llama-3.3-70B-Instruct-GGUF --include "*Llama-3.3-70B-Instruct-Q8_0*" --local-dir ~/models/Llama-3.3-70B-Instruct-GGUF_Q8_0

# Run a temporary container to use the llama.cpp tools to merge the split GGUF files into a single GGUF file that Ollama can use.
docker run --rm \
  -v ~/models:/models \
  --entrypoint /llm/llama-gguf-split \
  amperecomputingai/llama.cpp:latest \
  --merge \
  /models/Llama-3.3-70B-Instruct-GGUF_Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf \
  /models/Llama-3.3-70B-Instruct-GGUF_Q8_0/Llama-3.3-70B-Instruct-Q8_0-merged.gguf

aiperf profile \
  --model unsloth/Llama-3.3-70B-Instruct-GGUF \
  --url http://localhost:11434 \
  --endpoint-type chat \
  --request-rate 1 \
  --request-count 1 \
  --streaming

curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.3:70b",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'