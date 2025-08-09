"""
Wrapper for Google’s PaLM / Gemini Generative AI REST API.
The function signature matches the one expected by aura_agent.py.
"""
from __future__ import annotations

import os, requests, json

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GOOGLE_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/text-bison-001:generateText"
)

def generate_with_google(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY environment variable not set")

    body = {
        "prompt": {"text": prompt},
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    resp = requests.post(
        f"{GOOGLE_API_URL}?key={GOOGLE_API_KEY}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["output"]

__all__ = ["generate_with_google"]
