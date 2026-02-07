# Session 5: Agentic Coding (Claude Code + Local Ollama)

**Objective:** Transition from "Chatting with AI" to "Working with an AI Agent." Set up Claude Code and bridge it to your local Ollama backend to create a private, hyper-fast "AI Pair Programmer" that can read files, run terminal commands, and fix bugs autonomously.

## 1. What is an "Agentic" Workflow?

Unlike a standard Chatbot that just gives you code snippets, an Agent has "Tools."

- **File I/O:** Claude Code can read your entire codebase to understand context.
- **Terminal Access:** It can run `npm test`, `python script.py`, or `git commit`.
- **The Loop:** It thinks → Proposes a change → Runs a test → If it fails, it tries again.

Why the GB10? Agentic loops are "token heavy." They send massive system prompts and file contexts (often 15k+ tokens). The GB10’s 128GB of VRAM ensures you can maintain these large context windows without the AI "forgetting" the beginning of the task.

2. Hands-on Lab: Setup & Integration
Since January 2026, Ollama natively supports the Anthropic Messages API, making this setup much simpler than it used to be.

Step A: Install Claude Code
Run the native installer on your GB10. (Ensure you have Node.js 18+ installed from Session 1).

Bash
curl -fsSL https://claude.ai/install.sh | bash
Verify with claude --version.

Step B: Configure the Local Bridge
We need to tell Claude Code to stop looking for Anthropic's servers and look at your local Ollama port (11434) instead.

Add these variables to your ~/.bashrc (or ~/.zshrc):

Bash
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
Reload your shell: source ~/.bashrc.

Step C: Selecting the Right Agentic Model
Not all models are good at "Tool Use." For the best results on the GB10, we will use Qwen3-Coder or GLM-4.7-Flash, which are specifically trained for agentic loops.

Pull the model in Ollama:

Bash
ollama pull qwen3-coder:32b
3. Launching the Agent
Navigate to any code project directory on your GB10 and launch:

Bash
claude --model qwen3-coder:32b
Common Commands inside Claude Code:

/stats: Shows you how many tokens you've used in the session.

/compact: Clears the "memory" of the conversation to save VRAM while keeping the current file context.

Fix the bug in the login controller: Claude will now search your files, find the error, and ask for permission to edit the file.

🌟 Session 5 Challenge: The "Autonomous Bug Fix"
Task: Use the Agent to create and fix a project.

Ask Claude: "Create a simple Python FastAPI app with one GET endpoint that returns a random quote. Also, create a unit test for it."

Once created, ask: "Run the tests." (Claude should run the command for you).

The Twist: Purposely break the code (e.g., delete a comma) and tell Claude: "The app is broken, find the error and fix it until the tests pass again."

Observe: Watch the terminal as Claude "loops"—reading the error log, editing the file, and re-running the test automatically.

📚 Resources for Session 5
Playbook: Claude Code Local Setup

Documentation: Ollama Anthropic Compatibility

Pro Tip: If Claude Code feels "slow" to start, it’s usually because it is reading your .git history or large node_modules. Create a .claudeignore file in your project root to exclude those folders, just like a .gitignore.