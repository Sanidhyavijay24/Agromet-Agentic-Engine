"""
@file extractor.py
@description Unstructured meteorological bulletin extraction pipeline with schema validation and self-correction retry loop.
@module services/llm_extractor
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Literal

from dotenv import load_dotenv
from loguru import logger
import openai
from pydantic import ValidationError

from services.api.schemas import ExtractedWeatherPayload, RawBulletin, SourceStatus
from services.llm_extractor.prompt_templates import (
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)

# Load environment variables from .env if present
load_dotenv()

DEFAULT_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_SITE_URL: str = "http://localhost:8000"
DEFAULT_APP_NAME: str = "SIH-Microclimate-Downscaler"


def extract_weather_payload(
    raw_input: str | RawBulletin,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int = 1,
    initial_max_tokens: int | None = None,
) -> ExtractedWeatherPayload:
    """
    Extract structured meteorological parameters and advisory from raw bulletin text.

    Supports both plain string text and validated RawBulletin models. Communicates with an
    OpenAI-compatible LLM endpoint (OpenRouter, local Ollama, vLLM, etc.) with zero temperature.
    Implements a 1-retry self-correction loop if the initial JSON response fails Pydantic validation.

    Args:
        raw_input: Raw bulletin text string or RawBulletin instance.
        model: Optional model identifier override (default: LLM_MODEL or OPENROUTER_MODEL).
        base_url: Optional API base URL override (default: LLM_BASE_URL or OPENROUTER_BASE_URL).
        api_key: Optional API key override (default: LLM_API_KEY or OPENROUTER_API_KEY).
        max_retries: Number of corrective retries on validation failure (default: 1).

    Returns:
        ExtractedWeatherPayload containing validated numerical and textual data.

    Raises:
        ValueError: If input is empty, whitespace, or invalid type.
        RuntimeError: If API credentials are not configured, if the API call fails,
                      or if validation fails after corrective retries.
    """
    # 1. Input Validation and Metadata Extraction
    if isinstance(raw_input, RawBulletin):
        raw_text = raw_input.raw_text
        source_status: SourceStatus = raw_input.source_status
    elif isinstance(raw_input, str):
        raw_text = raw_input
        source_status = "live"
    else:
        raise ValueError("Input raw_text must be non-empty string or RawBulletin instance")

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Input raw_text must be non-empty string")

    clean_text = raw_text.strip()
    source_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]

    if source_status == "mock":
        logger.warning(
            "[extractor] Processing bulletin flagged with source_status='mock' (hash: {})",
            source_hash,
        )

    # 2. Environment & Credential Check (Swappable Provider Hierarchy)
    resolved_api_key = (
        api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if not resolved_api_key or not resolved_api_key.strip():
        raise RuntimeError("LLM_API_KEY or OPENROUTER_API_KEY not configured")

    resolved_base_url = (
        base_url
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    )

    resolved_model = (
        model.strip()
        if (model and isinstance(model, str) and model.strip())
        else (os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL))
    )

    # 3. Client Initialization
    headers: dict[str, str] = {}
    if "openrouter.ai" in resolved_base_url:
        site_url = os.getenv("OPENROUTER_SITE_URL", DEFAULT_SITE_URL)
        app_name = os.getenv("OPENROUTER_APP_NAME", DEFAULT_APP_NAME)
        headers["HTTP-Referer"] = site_url
        headers["X-Title"] = app_name

    client = openai.OpenAI(
        base_url=resolved_base_url.strip(),
        api_key=resolved_api_key.strip(),
        default_headers=headers if headers else None,
    )

    user_prompt = build_user_prompt(clean_text)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]

    # 4. Execution Loop with Self-Correction Retry
    for attempt in range(max_retries + 1):
        tokens_to_request = initial_max_tokens if (attempt == 0 and initial_max_tokens is not None) else 2048
        logger.info(
            "Dispatching extraction request (attempt {}/{}, model: {}, text_length: {} chars, max_tokens: {})",
            attempt + 1,
            max_retries + 1,
            resolved_model,
            len(clean_text),
            tokens_to_request,
        )

        start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=tokens_to_request,
            )
        except Exception as exc:
            latency = time.perf_counter() - start_time
            logger.error("LLM API call failed after {:.2f}s: {}", latency, exc)
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

        latency = time.perf_counter() - start_time
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        logger.info(
            "LLM response received in {:.2f}s | Tokens: prompt={}, completion={}, total={}",
            latency,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        if not response.choices or not response.choices[0].message.content:
            if attempt < max_retries:
                logger.warning("LLM returned empty response. Retrying...")
                continue
            raise RuntimeError("LLM returned empty completion response")

        raw_content = response.choices[0].message.content.strip()

        # Clean markdown code fences if wrapped by model
        cleaned_content = raw_content
        if "```json" in cleaned_content:
            cleaned_content = cleaned_content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned_content:
            cleaned_content = cleaned_content.split("```", 1)[1].split("```", 1)[0].strip()

        # 5. Parsing & Pydantic Validation
        try:
            parsed_json: dict[str, Any] = json.loads(cleaned_content)
            # Inject provenance and hash metadata
            parsed_json["source_status"] = source_status
            parsed_json["source_hash"] = source_hash

            payload = ExtractedWeatherPayload.model_validate(parsed_json)
            logger.success(
                "Successfully extracted weather payload (status={}, hash={}): condition='{}', temp=[{}, {}]°C, rain=[{}, {}]mm, qualifier='{}'",
                payload.source_status,
                payload.source_hash,
                payload.weather_condition,
                payload.min_temp_c,
                payload.max_temp_c,
                payload.rainfall_min_mm,
                payload.rainfall_max_mm,
                payload.rainfall_qualifier,
            )
            return payload

        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            if attempt < max_retries:
                logger.warning(
                    "Extraction validation failed on attempt {}/{} ({}). Retrying with corrective message...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous JSON output failed validation with error:\n{exc}\n\n"
                            f"Please correct the JSON response so that it strictly adheres to the schema. "
                            f"Output ONLY valid JSON."
                        ),
                    }
                )
            else:
                logger.error(
                    "Pydantic validation failed for LLM response after {} attempts: {}\nRaw LLM Content:\n{}",
                    max_retries + 1,
                    exc,
                    raw_content,
                )
                raise RuntimeError(
                    f"Failed to parse LLM response into ExtractedWeatherPayload after {max_retries + 1} attempts: {exc}. Raw response: {raw_content}"
                ) from exc

    raise RuntimeError("Extraction failed: maximum retry attempts exceeded")

