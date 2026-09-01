"""API-provider adapters with normalized response metadata."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from typing import Any

from benchmark_types import (
    ModelProvider,
    ModelText,
    resolve_api_key,
    serializable_metadata,
)

openai: Any
try:
    import openai as _openai

    openai = _openai
except ImportError:  # pragma: no cover - exercised by minimal installations
    openai = None

AsyncAnthropic: Any
try:
    from anthropic import AsyncAnthropic as _AsyncAnthropic

    AsyncAnthropic = _AsyncAnthropic
except ImportError:  # pragma: no cover - exercised by minimal installations
    AsyncAnthropic = None

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:  # pragma: no cover - exercised by minimal installations
    google_genai = None  # type: ignore[assignment]
    google_genai_types = None  # type: ignore[assignment]


def initialize_api_clients(config: dict[str, Any]) -> dict[str, Any]:
    """Create only the API clients for which a key is configured."""

    clients: dict[str, Any] = {}
    openai_key = resolve_api_key(config, ModelProvider.OPENAI)
    anthropic_key = resolve_api_key(config, ModelProvider.ANTHROPIC)
    gemini_key = resolve_api_key(config, ModelProvider.GEMINI)

    if openai_key:
        if openai is None:
            raise ImportError(
                "OpenAI SDK is required for provider=openai. Install requirements.txt."
            )
        clients["openai"] = openai.AsyncOpenAI(api_key=openai_key)

    if anthropic_key:
        if AsyncAnthropic is None:
            raise ImportError(
                "Anthropic SDK is required for provider=anthropic. "
                "Install requirements.txt."
            )
        clients["anthropic"] = AsyncAnthropic(api_key=anthropic_key)

    if gemini_key:
        if google_genai is None:
            raise ImportError(
                "Google Gen AI SDK is required for provider=gemini. "
                "Install requirements.txt."
            )
        clients["gemini"] = google_genai.Client(api_key=gemini_key)

    return clients


def huggingface_dependencies_available() -> bool:
    return bool(
        importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
    )


async def call_openai(
    clients: dict[str, Any],
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    role_config: dict[str, Any] | None = None,
) -> ModelText:
    if "openai" not in clients:
        raise ValueError(
            "OpenAI API key not configured. Set OPENAI_API_KEY or api_keys.openai."
        )

    role_config = role_config or {}
    if role_config.get("api_mode") == "responses":
        request: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "store": bool(role_config.get("store", False)),
        }
        if role_config.get("temperature") is not None:
            request["temperature"] = temperature
        if role_config.get("reasoning_effort"):
            request["reasoning"] = {"effort": role_config["reasoning_effort"]}
        response = await clients["openai"].responses.create(**request)
        return ModelText(
            response.output_text,
            {
                "provider": "openai",
                "requested_model": model,
                "resolved_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "interface": "responses",
                "status": getattr(response, "status", None),
                "incomplete_details": serializable_metadata(
                    getattr(response, "incomplete_details", None)
                ),
                "usage": serializable_metadata(getattr(response, "usage", None)),
            },
        )

    response = await clients["openai"].chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    return ModelText(
        choice.message.content,
        {
            "provider": "openai",
            "requested_model": model,
            "resolved_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "interface": "chat_completions",
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": serializable_metadata(getattr(response, "usage", None)),
        },
    )


async def call_anthropic(
    clients: dict[str, Any],
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float | None,
    role_config: dict[str, Any] | None = None,
) -> ModelText:
    if "anthropic" not in clients:
        raise ValueError(
            "Anthropic API key not configured. Set ANTHROPIC_API_KEY or "
            "api_keys.anthropic."
        )

    system_prompt = ""
    user_messages = []
    for message in messages:
        if message["role"] == "system":
            system_prompt = str(message["content"])
        else:
            user_messages.append(message)

    role_config = role_config or {}
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": user_messages,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if role_config.get("reasoning_effort"):
        request["thinking"] = {"type": "adaptive"}
        request["output_config"] = {"effort": role_config["reasoning_effort"]}

    response = await clients["anthropic"].messages.create(**request)
    text_parts = [
        str(block.text)
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not text_parts:
        raise RuntimeError("Anthropic response contained no text block")

    return ModelText(
        "\n".join(text_parts),
        {
            "provider": "anthropic",
            "requested_model": model,
            "resolved_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "interface": "messages_api",
            "usage": serializable_metadata(getattr(response, "usage", None)),
        },
    )


def _gemini_thinking_config(role_config: dict[str, Any]) -> Any | None:
    """Build a thinking config only when supported by the installed SDK."""

    thinking_level = role_config.get("thinking_level")
    thinking_budget = role_config.get("thinking_budget")
    if thinking_level is not None and thinking_budget is not None:
        raise ValueError(
            "Gemini config cannot set both thinking_level and thinking_budget"
        )
    if thinking_level is None and thinking_budget is None:
        return None
    if google_genai_types is None:
        raise RuntimeError("google-genai is not installed")

    model_fields = getattr(google_genai_types.ThinkingConfig, "model_fields", {})
    if thinking_level is not None:
        if "thinking_level" not in model_fields:
            raise ValueError(
                "Installed google-genai does not support thinking_level. "
                "Upgrade the SDK or omit this setting and use the model default."
            )
        return google_genai_types.ThinkingConfig.model_validate(
            {"thinking_level": thinking_level}
        )
    if thinking_budget is None:  # Narrow the dynamic configuration value for mypy.
        raise AssertionError("thinking_budget unexpectedly missing")
    return google_genai_types.ThinkingConfig(thinking_budget=int(thinking_budget))


async def call_gemini(
    clients: dict[str, Any],
    logger: logging.Logger,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    role_config: dict[str, Any] | None = None,
) -> ModelText:
    if "gemini" not in clients:
        raise ValueError(
            "Gemini API key not configured. Set GEMINI_API_KEY or api_keys.gemini."
        )
    if google_genai_types is None:
        raise RuntimeError("google-genai is not installed")

    role_config = role_config or {}
    prompt_parts = []
    for message in messages:
        role = str(message["role"]).title()
        prompt_parts.append(f"{role}: {message['content']}")
    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

    try:
        config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if not role_config.get("omit_sampling_parameters", False):
            config_kwargs["temperature"] = temperature
        thinking_config = _gemini_thinking_config(role_config)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        if role_config.get("response_mime_type"):
            config_kwargs["response_mime_type"] = str(role_config["response_mime_type"])

        response = await asyncio.to_thread(
            clients["gemini"].models.generate_content,
            model=model,
            contents=prompt,
            config=google_genai_types.GenerateContentConfig(**config_kwargs),
        )
        candidates = getattr(response, "candidates", None) or []
        finish_reasons = [
            str(getattr(candidate, "finish_reason", "unknown"))
            for candidate in candidates
        ]
        try:
            response_text = response.text
        except (AttributeError, ValueError) as error:
            logger.debug("Failed to access Gemini response.text: %s", error)
            response_text = None
        if not response_text:
            raise RuntimeError(
                "Gemini API returned no usable response content; finish reasons="
                f"{finish_reasons or ['none']}"
            )

        return ModelText(
            response_text,
            {
                "provider": "gemini",
                "requested_model": model,
                "resolved_model": getattr(response, "model_version", None),
                "interface": "google_genai_generate_content",
                "finish_reasons": finish_reasons,
                "usage": serializable_metadata(
                    getattr(response, "usage_metadata", None)
                ),
            },
        )
    except Exception as error:
        logger.error("Gemini API error: %s", error)
        logger.error(
            "Model: %s, Max tokens: %s, Temperature: %s, omit_sampling=%s",
            model,
            max_tokens,
            temperature,
            role_config.get("omit_sampling_parameters", False),
        )
        logger.error("Prompt length: %s", len(prompt))
        raise RuntimeError(f"Gemini API call failed: {error}") from error
