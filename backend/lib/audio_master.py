"""
Single helper used by analyzer_main.py to combine analysis + AI planning.
"""
from aura.analysis.orchestrator import analyze_audio_file
from aura.agent.aura_agent import get_ai_mastering_plan

def analyze_and_plan(local_path: str, cfg) -> tuple[dict, dict, str]:
    analysis = analyze_audio_file(local_path, cfg)
    plan, explanation = get_ai_mastering_plan(
        analysis.ai_summary_input, user_intent="", cfg=cfg
    )
    return analysis.model_dump(), plan.model_dump(), explanation
