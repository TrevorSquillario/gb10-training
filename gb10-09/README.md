# Lesson 9: RAG (Retrieval-Augmented Generation) & Private Data

**Objective:** Master RAG, the industry-standard way to eliminate AI hallucinations. Teach your GB10 to answer questions using your private sales playbooks, PDFs, and technical documentation. Unlike cloud RAG, this entire pipeline stays on your local NVMe storage, ensuring customer-sensitive data never touches the internet.

## The RAG Architecture on Blackwell

Standard LLMs are "frozen" in time based on their training data. RAG gives them a "library card" to look up facts.

- **The CPU/GPU Handshake:** On the GB10, the Grace ARM CPU excels at text processing (chunking and embedding), while the Blackwell GPU handles the intensive vector search and final answer generation.
- **Vector Databases:** We will use FAISS (Facebook AI Similarity Search) or Milvus, both of which are optimized to run in the GB10's 128GB Unified Memory space.

## Hands-on Lab: n8n

```bash
mkdir ~/gb10/n8n
sudo chown -R 1000:1000 ~/gb10/n8n
sudo chmod -R 700 ~/gb10/n8n
cd gb10-09/n8n
cp .env.example .env
docker compose up

# Browse to http://<gb10-ip>:5678
```

### External Access
The `compose.yaml` is configured in local-only mode (HTTP). This means no access to Google services or using webhooks with external integrations like Slack. This would be used when Slack reaches out to your local n8n instance. 

If you want to use these services and expose n8n to the Internet using CloudFlare Tunnels or the Tailscale Funnel you should set the `environment` variable to `N8N_SECURE_COOKIE=true` in your `compose.yaml`. This will enable HTTPS. You will also have to provide the domain name in `.env`

### Using n8n

#### Navigating
- Use `Ctrl + Mouse Wheel` to zoom `Mouse Wheel` to pan up/down
- Use `Ctrl + Left Click Drag` to pan `Left Click Drag` to select nodes
- Use `Tab` to open the Nodes Panel

### Import lesson workflows 
```bash
docker exec -it -u node n8n n8n import:workflow --separate --input /workflows
```

### Workflow 1: Simple RAG Lookup

```bash
# Pull the text embedding model in Ollama
docker exec -it ollama ollama pull mxbai-embed-large
docker exec -it ollama ollama pull mixtral:8x22b

```

The ***SimpleRAG*** workflow reads all files from the `n8n/local-files` directory in this lesson folder, creates text embeddings using Ollama then adds them to the n8n in-memory vector database.

When you start a Chat session in the Workflow and ask `What is the racadm command to set the DNS servers?` it will use the Ollama model configured in the `Ollama Chat Model` node to query the in-memory vector database and return an answer.





## Hands-on Lab: The AI Workbench RAG App

*I couldn't get this to work. The chat app loads fine but errors when submitting a chat message. 

We will use NVIDIA AI Workbench, a desktop tool that automates the setup of complex multi-container RAG environments. The RAG App uses Tavily to perform web search queries to ground the model using RAG. 

### Initialize the project

1. Install NVIDIA AI Workbench on your laptop from https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/
2. Start an SSH session to your GB10 and install the service. https://docs.nvidia.com/ai-workbench/user-guide/latest/install/remote-install.html
```bash
mkdir -p $HOME/.nvwb/bin && \
curl -L https://workbench.download.nvidia.com/stable/workbench-cli/$(curl -L -s https://workbench.download.nvidia.com/stable/workbench-cli/LATEST)/nvwb-cli-$(uname)-$(uname -m) --output $HOME/.nvwb/bin/nvwb-cli && \
chmod +x $HOME/.nvwb/bin/nvwb-cli

sudo $HOME/.nvwb/bin/nvwb-cli install \
--noninteractive \
--accept \
--docker \
--drivers \
--uid 1000 \
--gid 1000
```
3. Open AI Workbench on your laptop, select Add Remote Machine > Use Existing Machine. Fill in the form for SSH Key File browse and select `C:\Users\<USERNAME>\.ssh\id_rsa`
4. Select Clone Project
```bash
Repository URL: https://github.com/NVIDIA/workbench-example-agentic-rag
Path: Leave as the default
```
5. You will see a message `Your environment has unconfigured environment variables`. Click Resolve Now. Add your NGC API Key in `NVIDIA_API_KEY`. 
6.  Create a free account on https://www.tavily.com. Clicking the email verification link will take you to the Dashboard. Copy the API Key and paste that into `TAVILY_API_KEY`. Click Continue.
7. Navigate to Environment > Project Container > Appplication > Chat and start the web application. A browser window will open automatically and load with the Gradio chat interface.
 

Official Documentation: https://build.nvidia.com/spark/rag-ai-workbench/instructions


#### Cleanup
```bash
sudo $HOME/.nvwb/bin/nvwb-cli uninstall

# The uninstall command removes the NVIDIA Container Runtime so we need to reinstall. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
rm -rf ~/.nvwb 
```




---

## Resources for Lesson 9

- Playbook: RAG App in AI Workbench https://build.nvidia.com/spark/rag-ai-workbench/instructions
