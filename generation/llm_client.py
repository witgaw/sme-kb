"""Unified LLM client for Ollama and OpenRouter."""

import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import httpx
import openai

from config import get_config


@dataclass
class LLMUsage:
    """Token usage statistics from LLM generation."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Unified client for Ollama and OpenRouter LLMs."""

    def __init__(
        self,
        provider: Literal["ollama", "openrouter"] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            provider: 'ollama' for local, 'openrouter' for hosted. Uses config if None.
            model: Model name. Uses config if None.
            base_url: Base URL for Ollama. Uses config if None.
            api_key: API key for OpenRouter. Uses config/env if None.
        """
        config = get_config()

        self.provider = provider or config.llm_provider
        self.model = model or config.llm_model

        if self.provider == "ollama":
            self.base_url = base_url or config.ollama_base_url
            self._http_client = httpx.Client(timeout=120.0)
        else:  # openrouter
            key = api_key or config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("OPENROUTER_API_KEY required for OpenRouter provider")
            self._openai_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, LLMUsage]:
        """Generate response from LLM.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature. Uses config if None.
            max_tokens: Max tokens to generate. Uses config if None.

        Returns:
            Tuple of (generated text, usage statistics).
        """
        config = get_config()
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens

        if self.provider == "ollama":
            return self._generate_ollama(prompt, temperature, max_tokens)
        else:
            return self._generate_openrouter(prompt, temperature, max_tokens)

    def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, LLMUsage]:
        """Generate using Ollama API."""
        response = self._http_client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract usage statistics
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return data["response"], usage

    def _generate_openrouter(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, LLMUsage]:
        """Generate using OpenRouter API."""
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract usage statistics
        usage_data = response.usage
        if usage_data:
            usage = LLMUsage(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )
        else:
            usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return response.choices[0].message.content or "", usage

    def generate_with_history(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, LLMUsage]:
        """Generate response from LLM with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Roles can be 'system', 'user', or 'assistant'.
            temperature: Sampling temperature. Uses config if None.
            max_tokens: Max tokens to generate. Uses config if None.

        Returns:
            Tuple of (generated text, usage statistics).
        """
        config = get_config()
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens

        if self.provider == "ollama":
            return self._generate_ollama_chat(messages, temperature, max_tokens)
        else:
            return self._generate_openrouter_chat(messages, temperature, max_tokens)

    def _generate_ollama_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, LLMUsage]:
        """Generate using Ollama chat API."""
        response = self._http_client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract usage statistics
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return data["message"]["content"], usage

    def _generate_openrouter_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, LLMUsage]:
        """Generate using OpenRouter chat API."""
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract usage statistics
        usage_data = response.usage
        if usage_data:
            usage = LLMUsage(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )
        else:
            usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return response.choices[0].message.content or "", usage

    def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Generate response with streaming.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Yields:
            Text chunks as they are generated.
        """
        config = get_config()
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens

        if self.provider == "ollama":
            yield from self._stream_ollama(prompt, temperature, max_tokens)
        else:
            yield from self._stream_openrouter(prompt, temperature, max_tokens)

    def _stream_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Stream from Ollama API."""
        with self._http_client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    def _stream_openrouter(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Stream from OpenRouter API."""
        stream = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def close(self) -> None:
        """Close HTTP client connections."""
        if self.provider == "ollama" and hasattr(self, "_http_client"):
            self._http_client.close()
