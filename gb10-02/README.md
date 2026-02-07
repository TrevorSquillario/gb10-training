# Session 2: Containerization & The NVIDIA Container Stack

**Objective:** Master the NVIDIA Container Toolkit and learn how to run GPU-accelerated containers on the GB10. On the GB10 we don't install AI frameworks "on the metal"—we containerize them to keep the DGX OS clean and to switch between CUDA or LLM engine versions quickly.

## 1. The Container Strategy for Sales Engineers

- **Environment drift is the enemy.** Containers ensure that if a demo works on your GB10, it will work on the customer's hardware too.
- **Why Docker on Blackwell?** The GB10 uses an ARM64 architecture. Standard x86 Docker images won't work—use multi-arch or ARM64-specific images.
- **The NVIDIA Runtime:** Standard Docker doesn't "see" the GPU. The NVIDIA Container Toolkit (nvidia-container-runtime) maps the Blackwell GPU/CPU into the container so GPU workloads run as expected.

## 2. Hands-on Lab: Configuring the Engine

Most GB10 systems come with Docker pre-installed, but the "GPU hook" often needs a quick verification.

### Step A: Permission fix (run without sudo)

Add your user to the `docker` group so you can run Docker commands without `sudo`:

```bash
sudo usermod -aG docker $USER
```

Close your terminal and open a new one for the change to take effect. Verify by running:

```bash
docker ps
```

If you don't get a "permission denied" error, you're good to go.

### Step B: Verifying the GPU bridge

Run a smoke-test container to confirm the runtime can see the Blackwell GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu24.04 nvidia-smi
```

What to look for: the output should show the NVIDIA GB10 and a CUDA 13.0 driver.

## 3. Mastering the "Big Three" Commands

These three Docker patterns cover ~90% of typical AI demo workflows.

| Pattern   | Command example                                                       | Use case                                              |
|-----------|-----------------------------------------------------------------------|--------------------------------------------------------|
| The Probe | `docker run --rm -it --gpus all [image] /bin/bash`                     | "I want to go inside the container to inspect it."    |
| The Server| `docker run -d -p 8080:8080 --gpus all [image]`                       | "Run a web-based AI tool in the background."          |
| The Cleanup| `docker system prune -af`                                            | "Free space after downloading many models."           |

---

🌟 **Session 2 Challenge: The "Docker Model Runner"**

**Task:** Use the Docker AI plugin to pull a coding model optimized for ARM64 and run it locally.

Install the plugin (if not present):

```bash
sudo apt-get install docker-model-plugin
```

Pull the Qwen3-Coder model:

```bash
docker model pull ai/qwen3-coder
```

Verify the model is running and accessible at http://localhost:12434.

---

## Resources for Session 2

- Playbook: Docker Model Runner on DGX Spark
- Reference: NVIDIA Container Toolkit Docs

> **Pro tip:** Always check the **Architecture** field when pulling images from Docker Hub. If it doesn't say `arm64` or `aarch64`, it will not run on your GB10!
