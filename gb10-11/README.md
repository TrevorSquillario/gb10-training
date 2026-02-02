# Week 11: Enterprise Deployment with NVIDIA NIMs

**Objective:** Move from manual configuration to Enterprise AI. Deploy NVIDIA NIM (Inference Microservices), which are pre-packaged, "one-click" containers that wrap the world's most popular models in production-grade code. On the GB10, NIMs are the "Easy Button" for transitioning a lab project into a scalable customer solution.

## 1. Why NIMs on the GB10?

Think of NIM as the "Professional Wrapper" for everything we've learned. While SGLang (Week 10) is a raw engine, a NIM is a complete service.

- **Standardized APIs:** Every NIM exposes an industry-standard OpenAI-compatible API. This means if you build an app using a Llama-3 NIM, you can swap it for a Mistral NIM without changing a single line of your application code.
- **Architecture-Aware Optimization:** When you launch a NIM on your GB10, it automatically detects the Blackwell architecture and selects the most optimized TensorRT-LLM or vLLM backend specifically for the SM 12.1 instruction set.
- **Enterprise Security:** NIMs are part of the NVIDIA AI Enterprise stack, meaning they come with security patches and are validated to run 24/7 in production environments.

## 2. Hands-on Lab: Deploying a Llama-3 NIM

We will deploy the Llama-3.1-8B-Instruct NIM, which is officially supported on the DGX Spark.

### Step A: Authentication (NGC)

NIMs are hosted on the NVIDIA GPU Cloud (NGC). You will need your API key from Week 9.

```bash
# Login to the NVIDIA container registry
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Step B: Launching the NIM

This command pulls the NIM and starts the microservice. We map the port to 8000 to differentiate it from our other servers.

```bash
export MODEL_NAME="meta/llama-3.1-8b-instruct"

docker run -it --rm --name llama3-nim \
  --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

`-v ~/.cache/nim`: This ensures that once the model is downloaded, it stays on your GB10 even if you stop the container.

### Step C: Testing the production API

Open a second terminal and send a request to your new enterprise endpoint:

```bash
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role":"user", "content":"What are the three core benefits of NVIDIA NIM?"}]
  }'
```

## 3. Workflow: The "Self-Healing" Demo

In a production environment, if a service crashes, it must restart.

- **The Demo:** Show the customer how the NIM handles errors. Stop the container (`docker stop llama3-nim`) and restart it.
- **Observation:** Notice how quickly it comes back online. Because the model weights are already cached in your `~/.cache/nim` folder, the "Time to Ready" is minimal, which is a key requirement for enterprise SLAs.

---

🌟 **Week 11 Challenge: The "API Switcheroo"**

**Task:** Prove the portability of NIM.

1. Create a simple Python script that asks an AI to summarize a text using the URL `http://localhost:8000/v1`.
2. Deploy a second NIM (e.g., Mistral-7B) on a different port (e.g., 8001).
3. Change only the port number in your script.

**Goal:** Show that your "Customer Application" works perfectly with two different AI models without any code refactoring.

---

## Resources for Week 11

- Playbook: NIM on Spark Documentation
- Catalogue: Explore the NVIDIA NIM API Inventory

> **Next Step:** Ready for Week 12: The Graduation Showcase, where you'll put all these tools together for a final customer pitch demo?

**Video Resource:** Deploying Generative AI in Production with NVIDIA NIM. This livestream demonstration covers common use cases and provides a live walkthrough of simplifying AI deployment using the NVIDIA NIM microservice containers.
