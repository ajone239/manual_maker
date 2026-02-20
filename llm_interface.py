"""LLM interface supporting both Claude API and Ollama."""
from typing import Optional, Dict
from abc import ABC, abstractmethod
import os

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    LLM_PROVIDER
)


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM backend is available."""
        pass


class ClaudeLLM(BaseLLM):
    """Claude API interface."""

    def __init__(self, api_key: Optional[str] = None, model: str = CLAUDE_MODEL):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model
        self.client = None

        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return ANTHROPIC_AVAILABLE and self.api_key is not None and self.client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using Claude."""
        if not self.is_available():
            raise RuntimeError(
                "Claude API not available. Check API key and anthropic package.")

        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OllamaLLM(BaseLLM):
    """Ollama (local LLM) interface."""

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host
        self.client = None

        if OLLAMA_AVAILABLE:
            # Create client with explicit host
            os.environ['OLLAMA_HOST'] = host
            self.client = ollama.Client(host=host)

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if not OLLAMA_AVAILABLE or not self.client:
            return False

        try:
            # Try to list models to verify connection
            self.client.list()
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using Ollama."""
        if not self.is_available():
            raise RuntimeError(
                "Ollama not available. Make sure Ollama is running: "
                "https://ollama.ai/download"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        content = response['message']['content']

        # Log empty responses for debugging
        if not content or not content.strip():
            print(f"⚠️  Warning: {self.model} returned empty response")
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  Temperature: {temperature}")
            print(f"  Max tokens: {max_tokens}")

        return content


class LLMInterface:
    """
    Unified interface for multiple LLM backends.

    Automatically selects available backend or uses specified provider.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM interface.

        Args:
            provider: "anthropic", "ollama", or None for auto-detect
        """
        self.provider = provider or LLM_PROVIDER

        # Initialize backends
        self.claude = ClaudeLLM()
        self.ollama = OllamaLLM()

        # Select active backend
        self.llm = self._select_backend()

    def _select_backend(self) -> BaseLLM:
        """Select the appropriate LLM backend."""
        if self.provider == "anthropic":
            if not self.claude.is_available():
                raise RuntimeError(
                    "Claude requested but not available. "
                    "Set ANTHROPIC_API_KEY or install anthropic package."
                )
            print(f"Using Claude API ({self.claude.model})")
            return self.claude

        elif self.provider == "ollama":
            if not self.ollama.is_available():
                raise RuntimeError(
                    "Ollama requested but not available. "
                    "Make sure Ollama is running: https://ollama.ai/download"
                )
            print(f"Using Ollama ({self.ollama.model})")
            return self.ollama

        else:
            # Auto-detect: prefer Ollama (local, free) over Claude (API, paid)
            if self.ollama.is_available():
                print(f"Auto-detected: Using Ollama ({self.ollama.model})")
                return self.ollama
            elif self.claude.is_available():
                print(f"Auto-detected: Using Claude API ({self.claude.model})")
                return self.claude
            else:
                raise RuntimeError(
                    "No LLM backend available. Either:\n"
                    "1. Install and run Ollama: https://ollama.ai/download\n"
                    "2. Set ANTHROPIC_API_KEY for Claude API"
                )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using the active backend."""
        return self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the active provider."""
        if isinstance(self.llm, ClaudeLLM):
            return {
                "provider": "anthropic",
                "model": self.llm.model,
                "type": "API"
            }
        elif isinstance(self.llm, OllamaLLM):
            return {
                "provider": "ollama",
                "model": self.llm.model,
                "type": "Local"
            }
        return {"provider": "unknown"}


if __name__ == "__main__":
    # Test the LLM interface
    print("Testing LLM Interface...\n")

    llm = LLMInterface()
    info = llm.get_provider_info()
    print(f"Provider: {info['provider']}")
    print(f"Model: {info['model']}")
    print(f"Type: {info['type']}\n")

    # Test generation
    response = llm.generate(
        prompt="Tersely, what is RAG in the context of AI?",
    )
    print(f"Response: {response}")
