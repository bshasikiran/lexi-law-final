"""
NVIDIA LLM Wrapper — Reusable OpenAI-compatible client for NVIDIA API.
Supports both normal invoke and streaming responses with reasoning_content.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Explicitly load .env from project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

# NVIDIA API Configuration
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "openai/gpt-oss-120b"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Initialize the OpenAI-compatible client for NVIDIA
_client = None

def get_client() -> OpenAI:
    """Get or create the NVIDIA OpenAI client (singleton)."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
        )
    return _client


def invoke(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 1.0,
) -> str:
    """
    Non-streaming completion. Returns the full response text.
    
    Args:
        messages: List of message dicts [{"role": "...", "content": "..."}]
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        top_p: Nucleus sampling
    
    Returns:
        str: The complete response text
    """
    client = get_client()
    completion = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    return completion.choices[0].message.content or ""


def stream(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 1.0,
):
    """
    Streaming completion generator. Yields (event_type, content) tuples.
    
    Event types:
        - "TOKEN": Regular content token
        - "THINKING": Chain-of-thought / reasoning_content token
        - "DONE": Stream finished
    
    Args:
        messages: List of message dicts
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        top_p: Nucleus sampling
    
    Yields:
        tuple[str, str]: (event_type, content)
    """
    client = get_client()
    completion = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )
    
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        
        delta = chunk.choices[0].delta
        
        # Check for reasoning_content (chain-of-thought)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("THINKING", reasoning)
        
        # Check for regular content
        if delta.content is not None:
            yield ("TOKEN", delta.content)
        
        # Check for finish reason
        if chunk.choices[0].finish_reason is not None:
            yield ("DONE", "")


def build_messages(system_prompt: str, conversation_history: list[dict] = None, user_message: str = "") -> list[dict]:
    """
    Helper to build a properly formatted messages list.
    
    Args:
        system_prompt: The system instruction
        conversation_history: List of prior messages [{"role": "user/assistant", "content": "..."}]
        user_message: The current user message
    
    Returns:
        list[dict]: Formatted messages list
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    if conversation_history:
        messages.extend(conversation_history)
    
    if user_message:
        messages.append({"role": "user", "content": user_message})
    
    return messages
