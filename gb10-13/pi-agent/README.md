# Pi Agent — Dockerized

This directory contains a Dockerized [Pi agent](https://github.com/example/pi-agent) configuration that mirrors the Hermes agent setup, connecting to the same local vLLM instance and MCP servers.

## Overview

Pi is a CLI-based AI agent. This Docker setup isolates Pi in a container while sharing:

- **LLM backend** — the same vLLM Gemma 4 instance at `http://192.168.0.30:8090/v1`
- **MCP servers** — `idrac-redfish-mcp` and `ome-mcp` via HTTP transport
- **Skills** — the same skill directories used by Hermes (`hermes-skills`, `idrac-hermes-skills`)
- **Network** — joins the `hermes_hermes-net` bridge network to reach sibling containers

## Directory Structure

```
pi/
├── agent/
│   ├── models.json       # Provider & model definitions (→ ~/.pi/agent/models.json)
│   ├── settings.json     # Default provider, model, system prompt (→ ~/.pi/agent/settings.json)
│   └── agents/           # Optional custom agent profiles (→ ~/.pi/agent/agents/)
├── config/
│   └── mcp.json          # MCP server definitions (→ ~/.config/mcp/mcp.json)
├── Dockerfile            # Container image definition
├── compose.yaml          # Docker Compose service definition
├── .gitignore
└── README.md
```

## Configuration Files

### 1. `agent/models.json` — Model & Provider

Defines the vLLM OpenAI-compatible endpoint and the Gemma 4 model with multimodal support (`input: ["text", "image"]`). No separate vision block needed since Gemma 4 is natively multimodal.

Key details:
- `compat.supportsDeveloperRole: false` — vLLM expects `system` role, not `developer`
- `reasoning: true` — enables Gemma 4's thinking/reasoning tokens

### 2. `agent/settings.json` — Global Agent Settings

Sets the default provider, model, and system prompt — equivalent to Hermes' `config.yaml` `model` and `agent` blocks.

### 3. `config/mcp.json` — MCP Servers

Connects to the same HTTP-based MCP servers as Hermes using `streamable-http` transport. Both `idrac` and `ome` are defined.

## Usage

### Build and start

```bash
cd pi
docker compose up -d
```

### Run interactively

```bash
docker compose run --rm pi
```

Or attach to the running container:

```bash
docker compose exec pi pi
```

### Verify connectivity

Inside the container, check that Pi detects your model:

```bash
pi /model
```

And inspect connected MCP tools:

```bash
pi /mcp
```

Add bash alias to launch pi in the container:
```pi
cat /etc/profile.d/pi.sh
pi() {
    docker exec -it pi-agent pi "$@"
}
```

## Adding Custom Agent Profiles

Place markdown files in `agent/agents/` (e.g., `ops.md`):

```markdown
---
name: ops
description: Systems and operations sub-agent
model: vllm/google/Gemma-4-26B-A4B-it
---
You are a professional operations assistant capable of precise tool execution, autonomous web scraping, and Python development.
```

These are mounted into `~/.pi/agent/agents/` inside the container.

## Packages

- npm:pi-messenger
- npm:pi-fork