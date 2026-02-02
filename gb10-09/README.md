# Week 9: RAG (Retrieval-Augmented Generation) & Private Data

**Objective:** Master RAG, the industry-standard way to eliminate AI hallucinations. Teach your GB10 to answer questions using your private sales playbooks, PDFs, and technical documentation. Unlike cloud RAG, this entire pipeline stays on your local NVMe storage, ensuring customer-sensitive data never touches the internet.

## 1. The RAG Architecture on Blackwell

Standard LLMs are "frozen" in time based on their training data. RAG gives them a "library card" to look up facts.

- **The CPU/GPU Handshake:** On the GB10, the Grace ARM CPU excels at text processing (chunking and embedding), while the Blackwell GPU handles the intensive vector search and final answer generation.
- **Vector Databases:** We will use FAISS (Facebook AI Similarity Search) or Milvus, both of which are optimized to run in the GB10's 128GB Unified Memory space.

## 2. Hands-on Lab: The AI Workbench RAG App

We will use NVIDIA AI Workbench, a desktop tool that automates the setup of complex multi-container RAG environments.

### Step A: Initialize the project

Open AI Workbench on your laptop and point it to your GB10 location.

Clone the official Hybrid RAG project:

```
https://github.com/NVIDIA/workbench-example-hybrid-rag
```

AI Workbench will automatically build the environment, installing the vector DB and the retrieval service.

### Step B: Ingesting your sales docs

1. Open the Gradio Chat UI from the Workbench dashboard.
2. Navigate to the "Upload Documents" tab.
3. Drag and drop your PDFs (e.g., "GB10_Competitive_Analysis.pdf").

**Behind the scenes:** The GB10 is now "chunking" your PDF into small pieces, turning those pieces into math (embeddings), and storing them in your local vector database.

### Step C: The "grounded" chat

1. Switch back to the Chat tab and toggle "Use Vector Database" to **ON**.
2. **Prompt:** "What are the key performance metrics of the GB10 vs the previous generation according to our internal docs?"
3. **The Result:** The AI will provide a specific answer and—crucially—cite the source from the PDF you just uploaded.

## 3. Workflow: Hybrid vs. Fully Local

- **Cloud Mode:** Uses the GB10 as a client to hit NVIDIA's hosted NIM endpoints (fastest setup).
- **Local System:** Uses the GB10 for everything. This requires the most memory but is 100% private.

**Tip:** For the "Local System" mode on GB10, select the Mistral-7B or Llama-3-8B model for the best balance of speed and accuracy.

---

🌟 **Week 9 Challenge: The "Instant Expert"**

**Task:** Build a RAG system for a "Mock Customer Meeting."

1. Find a 20+ page technical manual or a long product roadmap.
2. Ingest it into your RAG app.
3. **The Test:** Try to "trick" the AI by asking a question that is not in its training data but is in the manual.

**Goal:** Verify that the AI answers using only the provided context and says "I don't know" if the information is missing, rather than making it up.

---

## Resources for Week 9

- Playbook: RAG App in AI Workbench
- Blog: Rethinking CPUs for Local RAG on DGX Spark

> **Pro Tip:** If you see an "Authorization Failed" error in the Chat UI, ensure your `NGC_API_KEY` is correctly set in the Workbench "Secrets" section. This key is required to download the optimized embedding models.

> **Next Step:** Ready for Week 10: High-Throughput Serving, where we learn how to make your GB10 handle 50+ simultaneous chat users at once?

**Video Resource:** Local Multimodal RAG Pipeline End-to-End Tutorial on DGX Spark. This video is a perfect companion to Week 9 as it provides a complete end-to-end tutorial on building a multimodal RAG pipeline specifically on the NVIDIA DGX Spark hardware you are using.