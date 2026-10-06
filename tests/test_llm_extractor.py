"""
@file test_llm_extractor.py
@description Comprehensive unit tests for LLM extraction subsystem including synthetic fixtures,
             validation error retry recovery, and provider swappability.
@module tests
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from services.api.schemas import ExtractedWeatherPayload, RawBulletin
from services.llm_extractor.extractor import extract_weather_payload
from services.llm_extractor.prompt_templates import (
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic Test Fixtures (Clearly marked: Not real IMD bulletins)
# ─────────────────────────────────────────────────────────────────────────────

# SYNTHETIC FIXTURE (not a real IMD bulletin): Forecast + advisory with no warnings mentioned
SYNTHETIC_BULLETIN_NO_WARNINGS = """
SYNTHETIC TEST BULLETIN — JAIPUR DISTRICT
Weather: Sunny and clear throughout the period.
Temperatures: Max 36.0 C, Min 22.0 C.
Rainfall: Nil.
Crop Advisory: Regular weeding recommended for cotton crop at vegetative stage.
"""

# SYNTHETIC FIXTURE (not a real IMD bulletin): Severe weather bulletin without agricultural advisory
SYNTHETIC_BULLETIN_NO_ADVISORY = """
SYNTHETIC TEST BULLETIN — JAIPUR NOWCAST
Observed Sky: Thunderstorm with squall.
Max Temp: 33.0 C, Min Temp: 24.0 C.
Rainfall: 45 mm to 70 mm heavy downpour.
Warnings: Heavy Rainfall Warning, Thunderstorm with Lightning.
"""

# SYNTHETIC FIXTURE (not a real IMD bulletin): Pure temperature telemetry bulletin
SYNTHETIC_BULLETIN_TEMP_ONLY = """
SYNTHETIC TEST BULLETIN — TEMPERATURE OBSERVATION
District: Jaipur
Observed Maximum Temperature: 29.5 C
Observed Minimum Temperature: 18.0 C
"""

# SYNTHETIC FIXTURE (not a real IMD bulletin): Multi-hazard alert bulletin with 3 distinct warnings
SYNTHETIC_BULLETIN_MULTIPLE_WARNINGS = """
SYNTHETIC TEST BULLETIN — RED ALERT WARNING
District: Jaipur
Valid: 05-09-2026 to 06-09-2026
Warning 1: Extremely Heavy Rainfall
Warning 2: Gusty Winds 50-60 km/h
Warning 3: Thunderstorm with Lightning and Hailstorm
"""

# SYNTHETIC FIXTURE (not a real IMD bulletin): Qualitative rain description with no mm numbers
SYNTHETIC_BULLETIN_QUALITATIVE_RAIN = """
SYNTHETIC TEST BULLETIN — MONSOON OUTLOOK
District: Jaipur
Sky: Generally cloudy.
Rainfall: Light to moderate rainfall likely over isolated places during next 48 hours.
Wind: 15 km/h South-Westerly.
"""


def test_build_user_prompt() -> None:
    """Verify build_user_prompt injects both raw text and target JSON schema."""
    raw_sample = "Heavy rain expected in Jaipur tomorrow with temperatures 22 to 30 C."
    prompt = build_user_prompt(raw_sample)

    assert raw_sample in prompt
    assert "min_temp_c" in prompt
    assert "max_temp_c" in prompt
    assert "rainfall_min_mm" in prompt
    assert "rainfall_qualifier" in prompt
    assert "Target JSON Schema:" in prompt


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "\t\n  ",
        None,
        12345,
        ["list of text"],
        {"text": "bulletin"},
    ],
)
def test_input_validation_rejection(invalid_input: object) -> None:
    """Ensure invalid inputs raise ValueError immediately."""
    with pytest.raises(ValueError, match="Input raw_text must be non-empty string"):
        extract_weather_payload(invalid_input)  # type: ignore[arg-type]


def test_missing_api_key_raises_runtime_error() -> None:
    """Ensure RuntimeError is raised when neither LLM_API_KEY nor OPENROUTER_API_KEY is configured."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="LLM_API_KEY or OPENROUTER_API_KEY not configured"):
            extract_weather_payload("Some valid bulletin text")


def test_successful_extraction_with_raw_bulletin_and_hash() -> None:
    """Verify RawBulletin input propagates source_status='mock' and generates deterministic SHA-256 hash."""
    mock_payload_dict = {
        "district": "Jaipur",
        "valid_from": "04-09-2026",
        "valid_to": "08-09-2026",
        "min_temp_c": 24.0,
        "max_temp_c": 35.0,
        "rainfall_min_mm": 10.0,
        "rainfall_max_mm": 25.0,
        "rainfall_qualifier": None,
        "wind_speed_kmh": 15.5,
        "wind_direction": "SW",
        "weather_condition": "Partly Cloudy",
        "warnings": ["Moderate Rainfall"],
        "advisory_text": "Apply fertilizer before rainfall. Ensure good drainage for soybean.",
    }
    raw_json_str = json.dumps(mock_payload_dict)

    mock_choice = MagicMock()
    mock_choice.message.content = raw_json_str
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 150
    mock_completion.usage.completion_tokens = 60
    mock_completion.usage.total_tokens = 210

    raw_bulletin = RawBulletin(
        source="IMD_AGROMET_BULLETIN_MOCK",
        timestamp=json.loads(json.dumps("2026-09-04T15:00:00Z")),
        raw_text="Sample raw bulletin text for hashing test",
        source_status="mock",
    )
    expected_hash = hashlib.sha256(raw_bulletin.raw_text.encode("utf-8")).hexdigest()[:16]

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload(raw_bulletin, model="custom-model-id")

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.source_status == "mock"
            assert result.source_hash == expected_hash
            assert result.district == "Jaipur"
            assert result.min_temp_c == 24.0
            assert result.max_temp_c == 35.0
            assert result.rainfall_min_mm == 10.0
            assert result.rainfall_max_mm == 25.0
            assert result.wind_speed_kmh == 15.5
            assert result.wind_direction == "SW"
            assert result.weather_condition == "Partly Cloudy"
            assert result.warnings == ["Moderate Rainfall"]
            assert result.advisory_text is not None


def test_synthetic_no_warnings() -> None:
    """SYNTHETIC TEST: Bulletin with no warnings produces empty list [], discarding placeholder sentinel."""
    mock_payload_dict = {
        "district": "Jaipur",
        "min_temp_c": 22.0,
        "max_temp_c": 36.0,
        "rainfall_min_mm": 0.0,
        "rainfall_max_mm": 0.0,
        "rainfall_qualifier": "Nil",
        "weather_condition": "Sunny",
        "warnings": ["No warning issued"],  # Should be sanitized to []
        "advisory_text": "Regular weeding recommended for cotton crop.",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload(SYNTHETIC_BULLETIN_NO_WARNINGS)

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.warnings == []
            assert result.min_temp_c == 22.0
            assert result.max_temp_c == 36.0
            assert result.advisory_text == "Regular weeding recommended for cotton crop."


def test_synthetic_no_advisory() -> None:
    """SYNTHETIC TEST: Warning/nowcast bulletin without advisory leaves advisory_text as None."""
    mock_payload_dict = {
        "district": "Jaipur",
        "min_temp_c": 24.0,
        "max_temp_c": 33.0,
        "rainfall_min_mm": 45.0,
        "rainfall_max_mm": 70.0,
        "weather_condition": "Thunderstorm",
        "warnings": ["Heavy Rainfall Warning", "Thunderstorm with Lightning"],
        "advisory_text": None,
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload(SYNTHETIC_BULLETIN_NO_ADVISORY)

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.advisory_text is None
            assert len(result.warnings) == 2
            assert result.rainfall_min_mm == 45.0
            assert result.rainfall_max_mm == 70.0


def test_synthetic_temp_only() -> None:
    """SYNTHETIC TEST: Temperature-only bulletin leaves rainfall, wind, warnings, and advisory as null/empty."""
    mock_payload_dict = {
        "district": "Jaipur",
        "min_temp_c": 18.0,
        "max_temp_c": 29.5,
        "rainfall_min_mm": None,
        "rainfall_max_mm": None,
        "rainfall_qualifier": None,
        "wind_speed_kmh": None,
        "wind_direction": None,
        "weather_condition": None,
        "warnings": [],
        "advisory_text": None,
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload(SYNTHETIC_BULLETIN_TEMP_ONLY)

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.min_temp_c == 18.0
            assert result.max_temp_c == 29.5
            assert result.rainfall_min_mm is None
            assert result.rainfall_max_mm is None
            assert result.rainfall_qualifier is None
            assert result.wind_speed_kmh is None
            assert result.warnings == []
            assert result.advisory_text is None


def test_synthetic_multiple_warnings() -> None:
    """SYNTHETIC TEST: Bulletin with 3 distinct warnings extracts all cleanly."""
    mock_payload_dict = {
        "district": "Jaipur",
        "valid_from": "05-09-2026",
        "valid_to": "06-09-2026",
        "warnings": [
            "Extremely Heavy Rainfall",
            "Gusty Winds 50-60 km/h",
            "Thunderstorm with Lightning and Hailstorm",
        ],
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload(SYNTHETIC_BULLETIN_MULTIPLE_WARNINGS)

            assert isinstance(result, ExtractedWeatherPayload)
            assert len(result.warnings) == 3
            assert "Extremely Heavy Rainfall" in result.warnings
            assert "Gusty Winds 50-60 km/h" in result.warnings


def test_synthetic_qualitative_rainfall() -> None:
    """SYNTHETIC TEST: Qualitative rainfall phrasing extracted into rainfall_qualifier with numerical mm as None."""
    mock_payload_dict = {
        "district": "Jaipur",
        "rainfall_min_mm": None,
        "rainfall_max_mm": None,
        "rainfall_qualifier": "light to moderate rainfall",
        "wind_speed_kmh": 15.0,
        "wind_direction": "South-Westerly",
        "weather_condition": "Generally cloudy",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload(SYNTHETIC_BULLETIN_QUALITATIVE_RAIN)

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.rainfall_qualifier == "light to moderate rainfall"
            assert result.rainfall_min_mm is None
            assert result.rainfall_max_mm is None
            assert result.wind_speed_kmh == 15.0
            assert result.wind_direction == "South-Westerly"


def test_retry_self_correction_success() -> None:
    """Verify that a 1st attempt validation failure initiates a corrective retry that succeeds on 2nd attempt."""
    # Attempt 1: Invalid schema (min_temp > max_temp)
    invalid_payload = {
        "min_temp_c": 40.0,
        "max_temp_c": 20.0,
        "weather_condition": "Clear",
    }
    # Attempt 2: Corrected schema
    valid_payload = {
        "min_temp_c": 20.0,
        "max_temp_c": 40.0,
        "weather_condition": "Clear",
    }

    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = json.dumps(invalid_payload)
    mock_completion_1 = MagicMock()
    mock_completion_1.choices = [mock_choice_1]
    mock_completion_1.usage = None

    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = json.dumps(valid_payload)
    mock_completion_2 = MagicMock()
    mock_completion_2.choices = [mock_choice_2]
    mock_completion_2.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.side_effect = [
                mock_completion_1,
                mock_completion_2,
            ]

            result = extract_weather_payload("Raw sample bulletin text", max_retries=1)

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.min_temp_c == 20.0
            assert result.max_temp_c == 40.0
            assert mock_client_instance.chat.completions.create.call_count == 2

            # Check that second call included corrective message
            second_call_messages = mock_client_instance.chat.completions.create.call_args_list[1][1]["messages"]
            assert len(second_call_messages) == 4  # system, user, assistant (invalid), user (corrective)
            assert "Your previous JSON output failed validation" in second_call_messages[3]["content"]


def test_retry_self_correction_failure_raises_runtime_error() -> None:
    """Verify that if validation fails on all attempts (including retry), RuntimeError is raised loudly."""
    invalid_payload = {
        "min_temp_c": 40.0,
        "max_temp_c": 20.0,
    }

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(invalid_payload)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            with pytest.raises(RuntimeError, match="Failed to parse LLM response into ExtractedWeatherPayload after 2 attempts"):
                extract_weather_payload("Raw sample bulletin text", max_retries=1)


def test_swappable_provider_env_vars() -> None:
    """Verify that LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL env vars take priority and configure client cleanly."""
    mock_payload = {
        "min_temp_c": 15.0,
        "max_temp_c": 25.0,
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    env_vars = {
        "LLM_BASE_URL": "https://custom-llm-host.internal/v1",
        "LLM_API_KEY": "custom-secret-token",
        "LLM_MODEL": "custom-mistral-large",
    }

    with patch.dict("os.environ", env_vars, clear=True):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            result = extract_weather_payload("Raw test bulletin")

            assert isinstance(result, ExtractedWeatherPayload)
            mock_openai_cls.assert_called_once_with(
                base_url="https://custom-llm-host.internal/v1",
                api_key="custom-secret-token",
                default_headers=None,
            )
            call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "custom-mistral-large"


def test_retry_self_correction_on_malformed_json_syntax() -> None:
    """Verify that malformed JSON syntax on 1st attempt triggers corrective retry that succeeds on 2nd attempt."""
    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = '{"min_temp_c": 20.0, "max_temp_c": '  # Corrupted syntax
    mock_completion_1 = MagicMock()
    mock_completion_1.choices = [mock_choice_1]
    mock_completion_1.usage = None

    valid_payload = {
        "min_temp_c": 20.0,
        "max_temp_c": 35.0,
        "weather_condition": "Partly Cloudy",
    }
    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = json.dumps(valid_payload)
    mock_completion_2 = MagicMock()
    mock_completion_2.choices = [mock_choice_2]
    mock_completion_2.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.side_effect = [
                mock_completion_1,
                mock_completion_2,
            ]

            result = extract_weather_payload("Raw sample bulletin text", max_retries=1)

            assert isinstance(result, ExtractedWeatherPayload)
            assert result.min_temp_c == 20.0
            assert result.max_temp_c == 35.0
            assert mock_client_instance.chat.completions.create.call_count == 2


def test_source_status_and_hash_fidelity_preservation() -> None:
    """Verify source_status ('mock', 'cached', 'live') and 16-char source_hash are strictly preserved end-to-end."""
    mock_payload = {"min_temp_c": 22.0, "max_temp_c": 34.0}
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value = mock_completion

            # 1. Mock status preservation
            b_mock = RawBulletin(
                source="IMD_MOCK",
                timestamp=json.loads(json.dumps("2026-09-04T12:00:00Z")),
                raw_text="Mock bulletin text for fidelity testing",
                source_status="mock",
            )
            res_mock = extract_weather_payload(b_mock)
            assert res_mock.source_status == "mock"
            assert res_mock.source_hash == hashlib.sha256(b_mock.raw_text.encode("utf-8")).hexdigest()[:16]

            # 2. Cached status preservation
            b_cached = RawBulletin(
                source="IMD_CACHED",
                timestamp=json.loads(json.dumps("2026-09-04T12:00:00Z")),
                raw_text="Cached bulletin text for fidelity testing",
                source_status="cached",
            )
            res_cached = extract_weather_payload(b_cached)
            assert res_cached.source_status == "cached"
            assert res_cached.source_hash == hashlib.sha256(b_cached.raw_text.encode("utf-8")).hexdigest()[:16]

            # 3. String input defaults to live status and computes non-empty hash
            raw_str = "Direct string bulletin text"
            res_str = extract_weather_payload(raw_str)
            assert res_str.source_status == "live"
            assert res_str.source_hash == hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]
            assert len(res_str.source_hash) == 16


def test_api_network_failure_raises_runtime_error() -> None:
    """Verify that upstream API errors raise RuntimeError and do NOT return mock fallback data."""
    with patch.dict("os.environ", {"LLM_API_KEY": "test-mock-key"}):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client_instance = MagicMock()
            mock_openai_cls.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.side_effect = ConnectionError("Connection refused")

            with pytest.raises(RuntimeError, match="LLM API call failed"):
                extract_weather_payload("Raw sample bulletin text")

