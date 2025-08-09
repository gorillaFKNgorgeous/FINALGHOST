"""Local worker orchestration for Aura pipeline."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

from aura.config import AppConfig
from aura.analysis.orchestrator import analyze_audio_file, sanitize_audio_input
from aura.agent.aura_agent import get_ai_mastering_plan
from aura.processing.chain import run_full_chain

logger = logging.getLogger(__name__)


def process_audio_locally(
    input_audio_path: str,
    user_intent_data: Dict[str, Any],
    app_config: AppConfig,
) -> Dict[str, Any]:
    """Run the entire analysis->AI planning->processing pipeline locally."""

    # 1. Initial analysis
    analysis_result = analyze_audio_file(input_audio_path, app_config)

    # 2. AI mastering plan
    mastering_plan, ai_explanation = get_ai_mastering_plan(
        analysis_result.ai_summary_input,
        user_intent_data.get("text_intent", ""),
        app_config,
    )

    # 3. Load audio for processing
    audio, sr = sf.read(input_audio_path, always_2d=False)
    audio = sanitize_audio_input(audio, sr)

    # 4. Processing chain
    processed_audio = run_full_chain(audio, sr, mastering_plan, logger=logger)

    # 5. Save to temporary path
    tmp_dir = Path(app_config.TEMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".wav", delete=False) as f:
        sf.write(f.name, processed_audio.T, sr)
        temp_path = f.name

    try:
        # 6. Post analysis
        post_analysis = analyze_audio_file(temp_path, app_config)

        # 7. Return structured result
        return {
            "original_analysis": analysis_result.model_dump(mode="python"),
            "ai_mastering_plan": mastering_plan.model_dump(mode="python"),
            "ai_explanation": ai_explanation,
            "processed_audio_data": processed_audio,
            "post_analysis": post_analysis.model_dump(mode="python"),
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)
