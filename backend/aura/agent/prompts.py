"""Prompt templates and versioning for Aura AI agent."""

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = (
    "You are Aura, an expert AI mastering engineer. Your goal is to enhance audio "
    "for clarity, balance, appropriate loudness, and to match the user's artistic "
    "intent. You output a precise JSON mastering plan and a human-readable explanation."
)

USER_PROMPT_TEMPLATE = (
    "Analysis summary:\n{ai_summary}\n\n"
    "User intent:\n{user_intent}\n\n"
    "Deterministic constraints:\n{constraints}\n\n"
    "Please produce a mastering plan JSON that conforms exactly to the following schema:\n{schema}\n"
    "After the JSON object, include a plain-text explanation of your decisions."
)

