"""Simple end-to-end sanity check using the local worker."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aura.config import AppConfig
from aura.worker.local_worker import process_audio_locally

logging.basicConfig(level=logging.INFO)


def main() -> None:
    cfg = AppConfig()

    # Locate a test file
    possible = [Path("tests/fixtures/full_track.wav")]
    audio_path = next((p for p in possible if p.exists()), None)
    if audio_path is None:
        raise FileNotFoundError("No test audio file found")

    user_intent = {"text_intent": "Make it loud for streaming."}

    result = process_audio_locally(str(audio_path), user_intent, cfg)

    print("AI explanation:\n", result["ai_explanation"])
    print("Target LUFS:", result["ai_mastering_plan"]["target_lufs"])
    print("Post-analysis LUFS:", result["post_analysis"]["global_metrics"]["integrated_lufs"])


if __name__ == "__main__":
    main()
