from __future__ import annotations

from typing import Any

import httpx2

from agent_obsidian.application.llm.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent_obsidian.application.llm.models import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


class OllamaLLMProvider:
    """Local Ollama implementation of the LLMProvider contract."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5:3b",
        timeout: float = 30.0,
        transport: Any = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama base URL cannot be empty")

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Ollama model name cannot be empty")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero")

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }

        try:
            async with httpx2.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

        except httpx2.TimeoutException as exc:
            raise LLMTimeoutError(
                "Ollama request timed out"
            ) from exc

        except httpx2.HTTPStatusError as exc:
            status_code = exc.response.status_code

            if status_code in (401, 403):
                raise LLMAuthenticationError(
                    f"Ollama authentication failed: HTTP {status_code}"
                ) from exc

            if status_code == 429:
                raise LLMRateLimitError(
                    "Ollama request was rate limited"
                ) from exc

            raise LLMProviderError(
                f"Ollama returned HTTP {status_code}"
            ) from exc

        except httpx2.RequestError as exc:
            raise LLMProviderError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(
                "Ollama returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise LLMInvalidResponseError(
                "Ollama response must be a JSON object"
            )

        message = data.get("message")

        if not isinstance(message, dict):
            raise LLMInvalidResponseError(
                "Ollama response is missing message object"
            )

        content = message.get("content")

        if not isinstance(content, str):
            raise LLMInvalidResponseError(
                "Ollama response message content must be a string"
            )

        model = data.get("model", self._model_name)

        if not isinstance(model, str) or not model.strip():
            raise LLMInvalidResponseError(
                "Ollama response contains an invalid model name"
            )

        finish_reason = data.get("done_reason")

        if finish_reason is not None and not isinstance(
            finish_reason, str
        ):
            raise LLMInvalidResponseError(
                "Ollama finish reason must be a string or None"
            )

        input_tokens = self._read_token_count(
            data,
            "prompt_eval_count",
        )
        output_tokens = self._read_token_count(
            data,
            "eval_count",
        )

        total_tokens = None

        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        try:
            usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except (TypeError, ValueError) as exc:
            raise LLMInvalidResponseError(
                "Ollama returned invalid token usage"
            ) from exc

        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _read_token_count(
        data: dict[str, Any],
        key: str,
    ) -> int | None:
        value = data.get(key)

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise LLMInvalidResponseError(
                f"Ollama returned invalid {key}"
            )

        return value