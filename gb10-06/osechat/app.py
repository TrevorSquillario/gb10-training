import logging
import os
import re
import subprocess
from typing import Optional

import chainlit as cl
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Tool response model
class NvidiaSmiStats(BaseModel):
    gpu_name: str
    temperature: str
    memory_usage: str
    raw_output: Optional[str] = None
    error: Optional[str] = None


# Define the agent with Ollama
model = os.getenv("OLLAMA_MODEL", "llama3.1")
logger.debug(f"Initializing agent with model: {model}")
logger.debug(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'Not set')}")
logger.debug(f"OLLAMA_API_KEY: {'Set' if os.getenv('OLLAMA_API_KEY') else 'Not set'}")

retries = 1
agent = Agent(
    f"ollama:{model}",
    deps_type=type(None),
    retries=retries,
    system_prompt=(
        "You are a helpful AI assistant with access to system GPU information. "
        "When users ask about GPU status, temperature, or memory usage, use the "
        "get_nvidia_smi_stats tool to retrieve current data and provide a clear summary. "
        "Do not include any extra parameters when calling the tool."
    ),
)
logger.debug("Agent initialized successfully")


def extract_agent_result_text(result) -> str:
    """Safely extract a displayable text from various Agent run result shapes."""
    # Common attribute names across different agent/result implementations
    candidates = [
        "text",
        "output",
        "response",
        "content",
        "final_text",
        "final_output",
        "answer",
        "message",
        "result",
        "generated_text",
    ]

    for attr in candidates:
        if hasattr(result, attr):
            val = getattr(result, attr)
            try:
                if isinstance(val, str):
                    return val
                # If it's an object with a content field (e.g., message-like)
                if hasattr(val, "content") and isinstance(val.content, str):
                    return val.content
                return str(val)
            except Exception:
                continue

    # Fallback to stringifying the whole result
    try:
        return str(result)
    except Exception:
        return "<unreadable result>"


@agent.tool
async def get_nvidia_smi_stats(ctx: RunContext, name: Optional[str] = None) -> NvidiaSmiStats:
    """
    Execute nvidia-smi command and parse GPU statistics including name, temperature, and memory usage.

    Returns:
        NvidiaSmiStats object containing GPU information
    """
    logger.debug("Executing get_nvidia_smi_stats tool")
    try:
        logger.debug("Running nvidia-smi command")
        # Run nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        logger.debug(f"nvidia-smi return code: {result.returncode}")
        logger.debug(f"nvidia-smi stdout: {result.stdout}")
        logger.debug(f"nvidia-smi stderr: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"nvidia-smi returned error: {result.stderr}")
            return NvidiaSmiStats(
                gpu_name="Unknown",
                temperature="N/A",
                memory_usage="N/A",
                error=f"nvidia-smi returned error: {result.stderr}"
            )

        # Parse the output
        output = result.stdout.strip()
        logger.debug(f"Parsing nvidia-smi output: {output}")

        # Example output: "NVIDIA GeForce RTX 3090, 45, 2048, 24576"
        parts = [part.strip() for part in output.split(",")]
        logger.debug(f"Parsed parts: {parts}")

        if len(parts) >= 4:
            gpu_name = parts[0]
            temperature = f"{parts[1]}°C"
            memory_used = parts[2]
            memory_total = parts[3]
            memory_usage = f"{memory_used}MB / {memory_total}MB"
            logger.debug(f"GPU stats - name: {gpu_name}, temp: {temperature}, usage: {memory_usage}")

            return NvidiaSmiStats(
                gpu_name=gpu_name,
                temperature=temperature,
                memory_usage=memory_usage,
                raw_output=output
            )
        else:
            logger.error(f"Unexpected nvidia-smi output format: {output}")
            return NvidiaSmiStats(
                gpu_name="Unknown",
                temperature="N/A",
                memory_usage="N/A",
                error="Could not parse nvidia-smi output",
                raw_output=output
            )

    except FileNotFoundError:
        logger.error("nvidia-smi command not found - NVIDIA drivers may not be installed")
        return NvidiaSmiStats(
            gpu_name="Unknown",
            temperature="N/A",
            memory_usage="N/A",
            error="nvidia-smi command not found. NVIDIA drivers may not be installed."
        )
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi command timed out")
        return NvidiaSmiStats(
            gpu_name="Unknown",
            temperature="N/A",
            memory_usage="N/A",
            error="nvidia-smi command timed out"
        )
    except Exception as e:
        logger.exception(f"Unexpected error in get_nvidia_smi_stats: {str(e)}")
        return NvidiaSmiStats(
            gpu_name="Unknown",
            temperature="N/A",
            memory_usage="N/A",
            error=f"Unexpected error: {str(e)}"
        )
        logger.debug(f"Parsing nvidia-smi output: {output}")

        # Example output: "NVIDIA GeForce RTX 3090, 45, 2048, 24576"
        parts = [part.strip() for part in output.split(",")]
        logger.debug(f"Parsed parts: {parts}")

        if len(parts) >= 4:
            gpu_name = parts[0]
            temperature = f"{parts[1]}°C"
            memory_used = parts[2]
            memory_total = parts[3]
            memory_usage = f"{memory_used}MB / {memory_total}MB"
            logger.debug(f"GPU stats - name: {gpu_name}, temp: {temperature}, usage: {memory_usage}")

            return NvidiaSmiStats(
                gpu_name=gpu_name,
                temperature=temperature,
                memory_usage=memory_usage,
                raw_output=output
            )
        else:
            logger.error(f"Unexpected nvidia-smi output format: {output}")
            return NvidiaSmiStats(
                gpu_name="Unknown",
                temperature="N/A",
                memory_usage="N/A",
                error="Could not parse nvidia-smi output",
                raw_output=output
            )

    except FileNotFoundError:
        logger.error("nvidia-smi command not found - NVIDIA drivers may not be installed")
        return NvidiaSmiStats(
            gpu_name="Unknown",
            temperature="N/A",
            memory_usage="N/A",
            error="nvidia-smi command not found. NVIDIA drivers may not be installed."
        )
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi command timed out")
        return NvidiaSmiStats(
            gpu_name="Unknown",
            temperature="N/A",
            memory_usage="N/A",
            error="nvidia-smi command timed out"
        )
    except Exception as e:
        logger.exception(f"Unexpected error in get_nvidia_smi_stats: {str(e)}")
        return NvidiaSmiStats(
            gpu_name="Unknown",
            temperature="N/A",
            memory_usage="N/A",
            error=f"Unexpected error: {str(e)}"
        )


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Log agent configuration for debugging
    logger.info(f"Chat started - Model: {model}")
    logger.info(f"Agent config: retries={retries}")

    await cl.Message(
        content="👋 Hello! I'm your AI assistant with GPU monitoring capabilities. Ask me about your GPU status!"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages."""
    logger.info(f"Received message: {message.content}")

    # Create a step to show the agent is processing
    result_text = None
    async with cl.Step(name="Agent Processing", type="tool") as step:
        step.input = message.content
        logger.debug("Agent step started")

        try:
            # Run the agent with the user's message
            logger.debug("Running agent with message")
            result = await agent.run(message.content)
            logger.debug(f"Agent completed, result type: {type(result)}")

            # Safely extract text from the agent result
            result_text = extract_agent_result_text(result)
            logger.debug(f"Agent result text: {result_text}")

            # Show the agent's response
            step.output = result_text

        except Exception as e:
            logger.exception(f"Error running agent: {str(e)}")
            step.output = f"Error: {str(e)}"
            await cl.Message(
                content=f"Sorry, I encountered an error: {str(e)}"
            ).send()
            return

    logger.info(f"Sending response: {result_text}")
    # Send the final response
    await cl.Message(content=result_text).send()


if __name__ == "__main__":
    # Note: Chainlit apps are typically run with: chainlit run app.py
    pass
