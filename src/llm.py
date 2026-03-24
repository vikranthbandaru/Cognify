"""
llm.py - Single shared LLM call for all Cognify modules.

Both gemini_utils.py and doc_utils.py import `chat()` from here.
Model config lives in .env - change LLM_MODEL there to switch providers.
"""

import os
import litellm
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL    = os.getenv("LLM_MODEL",        "ollama/qwen2.5:3b")
API_KEY  = os.getenv("GEMINI_API_KEY",   "not-needed")
API_BASE = os.getenv("OLLAMA_API_BASE",  None)

litellm.set_verbose = False


def chat(messages: list[dict], temperature: float = 0, max_tokens: int = 500) -> str:
    """
    Call the configured LLM and return the response string.

    Args:
        messages:    OpenAI-style message list, e.g. [{"role": "user", "content": "..."}]
        temperature: 0 = deterministic, 0.7 = creative
        max_tokens:  Maximum response length in tokens

    Returns:
        Stripped response string from the model.
    """
    kwargs = dict(model=MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if API_BASE:
        kwargs["api_base"] = API_BASE
    else:
        kwargs["api_key"] = API_KEY

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip()
