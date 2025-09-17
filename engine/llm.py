from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

from loguru import logger
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Load env from common locations early to pick up OPENAI_API_KEY during import
try:
    # Try local .env inside ai-chat-simulation, then project root
    here = Path(__file__).resolve().parents[1]
    env_candidates = [here / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        if env_path.is_file():
            load_dotenv(dotenv_path=str(env_path), override=False)
            break
except Exception:
    # Fallback to default behavior if dotenv not available or path issues
    try:
        load_dotenv()
    except Exception:
        pass


@lru_cache(maxsize=8)
def get_openai_chat(model: Optional[str] = None, temperature: float = None) -> Optional[ChatOpenAI]:
    """Return a cached LangChain ChatOpenAI client using env configuration.

    Env vars:
      - OPENAI_API_KEY (required)
      - OPENAI_MODEL (optional; default: gpt-4o-mini)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set; cannot initialize OpenAI chat client")
        return None
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if temperature is None:
        try:
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "1"))
        except Exception:
            temperature = 1.0
    # Optional max tokens cap to speed up responses
    try:
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "160"))
    except Exception:
        max_tokens = None
    logger.debug(f"Initializing OpenAI chat model={mdl} temperature={temperature}")
    kwargs = {"model": mdl, "temperature": temperature, "api_key": api_key}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


# Eager singleton for convenience
openai_chat = get_openai_chat()
