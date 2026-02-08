# Session 5: Agentic Coding (Claude Code + Local Ollama)

**Objective:** Transition from "Chatting with AI" to "Working with an AI Agent." Set up Claude Code and bridge it to your local Ollama backend to create a private, hyper-fast "AI Pair Programmer" that can read files, run terminal commands, and fix bugs autonomously.

## What is Claude Code?

Unlike a standard Chatbot that just gives you code snippets, an Agent has "Tools."

- **File I/O:** Claude Code can read your entire codebase to understand context.
- **Terminal Access:** It can run `npm test`, `python script.py`, or `git commit`.
- **The Loop:** It thinks → Proposes a change → Runs a test → If it fails, it tries again.

Why the GB10? Agentic loops are "token heavy." They send massive system prompts and file contexts (often 15k+ tokens). The GB10’s 128GB of VRAM ensures you can maintain these large context windows without the AI "forgetting" the beginning of the task.

## Hands-on Lab: Setup & Integration
Since January 2026, Ollama natively supports the Anthropic Messages API, making this setup much simpler than it used to be.

### Install Claude Code
Run the native installer on your GB10.

```bash
curl -fsSL https://claude.ai/install.sh | bash
```
Wait for that to finish then run
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

Verify with `claude --version`.

### Configure the Local Bridge
We need to tell Claude Code to stop looking for Anthropic's servers and look at your local Ollama port (11434) instead.

Add these variables to your `~/.bashrc`:

```bash
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

Reload your shell: `source ~/.bashrc`.

### Selecting the Right Agentic Model
Not all models are good at "Tool Use." For the best results I recommend, which are specifically trained for agentic loops.

- `qwen3-coder-next`

Pull the model in Ollama:

```bash
docker exec -it ollama ollama pull qwen3-coder-next
```
3. Launching the Agent
Navigate to any code project directory on your GB10 and launch:

```bash
claude --model qwen3-coder-next
```

## Setup Claude Code in VSCode

Since your GB10 is already running Ollama and Claude Code, we just need to "plug in" VS Code.

### Extension install

1. Open VS Code on your laptop 
2. Select the Extensions tab on the left or hit Ctrl + Shift + P and search for Install Extension
3. Search for Remote - SSH and install
4. Search for Claude Code and install
6. Hit Ctrl + Shift + P and search for User Settings then select Preferences: Open User Settings
7. Go to Extensions > Claude Code and select
```
- Disable Login Prompt
- Select Model: qwen3-coder-next
- Use Terminal
```

## Connect to your GB10 using the Remote SSH Extension

First you need to setup SSH key authentication from your laptop to your GB10. This setup assumes you're on Windows.

1. In VSCode select the Terminal menu at the top, then New Terminal.
2. A PowerShell terminal window will appear at the bottom. Type `ssh-keygen` then just hit Enter several times to accept the defaults.
3. Enter `type ~/.ssh/id_rsa.pub | clip`. This will print the SSH public key and copy it to the clipboard in Windows.
4. SSH to your GB10
```bash
vi ~/.ssh/authorized_keys

o
# Paste the ssh key
wq
```

Then setup the connection in VSCode
1. Click the Remote Explorer icon in the left bar. Hover over to see the names.
2. Hover over the SSH section and click the + icon
3. A box will appear at the top center of the window. Enter `ssh <username>@<gb10-ip>`
4. It should log you in automatically using passwordless SSH key authentication. If it prompts you for a password, something it's setup properly. You can enter your password but you'll have to do it everytime.

## Using Claude Code

1. Select the Claude Code icon (orange sprite) in the top right
2. This will open the build-in VSCode Terminal and launch Claude Code in Terminal Mode. As of this writing it seems the extensions chat interface is broke when using a local LLM. We'll expore other options in future lessons.
3. Ensure the `qwen3-coder-next`is displayed in the top right. If not use `/model qwen3-coder-next` to select the model
4. 


### Useful Commands inside Claude Code

`/stats`: Shows you how many tokens you've used in the session.

`/compact`: Clears the "memory" of the conversation to save VRAM while keeping the current file context.

`/model`: Changes or selects a model to use

`/exit`: Exit session, you can just type `exit` as well

🌟 Session 5 Challenge: Editing and creating local files

Prompt: Create a test directory in the current path and create 3 text files. Fill the files with loren ipsum text

📚 Resources for Session 5
Playbook: Claude Code Local Setup

Documentation: Ollama Anthropic Compatibility

Pro Tip: If Claude Code feels "slow" to start, it’s usually because it is reading your .git history or large node_modules. Create a .claudeignore file in your project root to exclude those folders, just like a .gitignore.