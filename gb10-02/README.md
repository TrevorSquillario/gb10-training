# Lesson 2: Containerization & The NVIDIA Container Stack

**Objective:** Master the NVIDIA Container Toolkit and learn how to run GPU-accelerated containers on the GB10. On the GB10 we don't install AI frameworks "on the metal"—we containerize them to keep the DGX OS clean and to switch between CUDA or LLM engine versions quickly.

## The Container Strategy 

- **Environment drift is the enemy.** Containers ensure that if a demo works on your GB10, it will work on the customer's hardware too.
- **Why Docker on Blackwell?** The GB10 uses an ARM64 architecture. Standard x86 Docker images won't work—use multi-arch or ARM64-specific images.
- **The NVIDIA Runtime:** Standard Docker doesn't "see" the GPU. The NVIDIA Container Toolkit (nvidia-container-runtime) maps the Blackwell GPU/CPU into the container so GPU workloads run as expected.

## Hands-on Lab: Configuring the Engine

Most GB10 systems come with Docker pre-installed, but the "GPU hook" often needs a quick verification.

### Permission fix (run without sudo)

If your user isn't in the `docker` group, add your user to the group so you can run Docker commands without `sudo`:

```bash
groups | grep docker
sudo usermod -aG docker $USER
```

Close your terminal and open a new one for the change to take effect. Verify by running:

```bash
docker ps
```

If you don't get a "permission denied" error, you're good to go.

### Verifying the GPU bridge

First we'll make sure the NVIDIA Container Runtime is installed. The GB10 and all DGX systems come preinstalled with the Nvidia Container Toolkit. For other systems see [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
dpkg -l | grep nvidia-container-toolkit
docker info | grep -i runtime
```

Run a smoke-test container to confirm the runtime can see the Blackwell GPU. Then see that the container was downloaded and stored locally. 

```bash
docker run --rm  --gpus all   nvidia/cuda:13.1.1-base-ubuntu24.04 nvidia-smi
docker image ls
```

What to look for: the output should show the NVIDIA GB10 and a CUDA 13..1 driver.

#### Debugging

Generate debug logs for NVIDIA Support

Wait for completion: It may take 30–60 seconds. Once finished, it will output a file named nvidia-bug-report.log.gz in your current directory.

Note: If the script hangs, it usually means your GPU is in a "frozen" state. You can try running sudo `nvidia-bug-report.sh --safe-mode` to skip the hung components.

```bash
sudo nvidia-bug-report.sh
```

Enable debug logs for the Container Runtime. Find the line #debug = "/var/log/nvidia-container-runtime.log" and uncomment it (remove the #). Save and restart Docker/Containerd.

```bash
sudo vi /etc/nvidia-container-runtime/config.toml
sudo systemctl restart docker
journalctl -u docker | grep nvidia
```

## Useful Docker Commands

These common Docker commands cover most demo and troubleshooting workflows.

| Pattern       | Command example                                                       | Use case                                              |
|---------------|-----------------------------------------------------------------------|--------------------------------------------------------|
| The Probe     | `docker exec -it <container> /bin/bash`                                | Open an interactive shell inside a running container.  |
| List          | `docker ps -a`                                                         | List running and stopped containers.                   |
| Logs          | `docker logs -f <container>`                                           | Stream container logs (use `-n` to tail N lines).      |
| Run           | `docker run --rm -it --gpus all <image> bash`                          | Start an interactive GPU-enabled container.            |
| Images        | `docker images` (or `docker image ls`)                                 | List local images.                                     |
| Pull          | `docker pull <image>`                                                  | Download an image from a registry.                     |
| Inspect       | `docker inspect <container>`                                     | JSON low-level details about containers/images.        |
| Remove (cont) | `docker rm <container>`                                                 | Remove stopped container(s).                           |
| Remove (img)  | `docker rmi <image>`                                                   | Remove an image from local cache.                      |
| Cleanup       | `docker system prune -af`                                              | Remove unused data to free space.                      |

---

## Resources for Lesson 2

- Reference: NVIDIA Container Toolkit Docs — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html

> **Pro tip:** Always check the **Architecture** field when pulling images from registries. If it doesn't say `arm64` or `aarch64`, it will not run on your GB10!
