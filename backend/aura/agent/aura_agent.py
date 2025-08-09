"""Aura AI Mastering Agent implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import openai
import google.generativeai as genai
from pydantic import ValidationError

from aura.config import AppConfig
from aura.agent.google_agent import generate_with_google
from aura.schemas import AISummaryModel, MasteringParams
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, PROMPT_VERSION

logger = logging.getLogger(__name__)


def pre_process_ai_constraints(ai_summary: AISummaryModel) -> dict:
    """Apply deterministic rules based on analysis summary."""
    constraints: Dict[str, Any] = {}
    summary_text = " ".join(ai_summary.global_metric_highlights).lower()
    if "true peak" in summary_text and "-0.5" in summary_text:
        constraints["clipper_settings.threshold_db"] = -1.5
    if "very quiet" in ai_summary.overall_assessment.lower():
        constraints["lufs_normalization_settings.max_gain_db"] = 8.0
    return constraints


def _call_openai(system_prompt: str, user_prompt: str, app_config: AppConfig) -> str:
    client = openai.OpenAI(api_key=app_config.OPENAI_API_KEY.get_secret_value())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, user_prompt: str, app_config: AppConfig) -> str:
    genai.configure(api_key=app_config.GEMINI_API_KEY.get_secret_value())
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        [
            {"role": "system", "parts": [system_prompt]},
            {"role": "user", "parts": [user_prompt]},
        ]
    )
    return response.text


def get_ai_mastering_plan(
    ai_summary: AISummaryModel,
    user_intent: str,
    app_config: AppConfig,
) -> tuple[MasteringParams, str]:
    """Generate a mastering plan using an LLM with validation and retries."""

    constraints = pre_process_ai_constraints(ai_summary)
    schema_json = MasteringParams.model_json_schema()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        ai_summary=ai_summary.model_dump(),
        user_intent=user_intent,
        constraints=json.dumps(constraints),
        schema=json.dumps(schema_json),
    )
    system_prompt = SYSTEM_PROMPT + f"\nPrompt version: {PROMPT_VERSION}"

    last_error = None
    if app_config.llm_provider == "google":
        def call_llm(system_prompt: str, user_prompt: str, cfg: AppConfig) -> str:
            return generate_with_google(f"{system_prompt}\n{user_prompt}")
    elif app_config.llm_provider == "gemini":
        call_llm = _call_gemini
    else:
        call_llm = _call_openai
    for attempt in range(3):
        try:
            content = call_llm(system_prompt, user_prompt, app_config)
            json_str, _, explanation = content.partition("\n")
            plan_dict = json.loads(json_str)
            plan = MasteringParams(**plan_dict)
            return plan, explanation.strip()
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            last_error = e
            logger.error("AI plan parse/validation error on attempt %s: %s", attempt + 1, e)
            user_prompt += f"\nThe previous output failed validation: {e}"
        except Exception as e:  # network or API errors
            last_error = e
            logger.error("LLM API error on attempt %s: %s", attempt + 1, e)
    logger.warning("Falling back to safe default plan due to error: %s", last_error)
    return MasteringParams(), "I encountered difficulty generating a detailed plan. Here's a safe master."


__all__ = ["get_ai_mastering_plan", "pre_process_ai_constraints"]
