"""
Aura Analysis Orchestrator - Main entry point for audio analysis
"""

import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple
from scipy.signal import butter, sosfilt

from ..config import AppConfig
from ..schemas import (
    FileInfoModel,
    GlobalMetricsModel,
    AnalysisResult,
    SlidingWindowAnalysisModel,
    MusicalSectionAnalysisModel,
    AISummaryModel,
)

logger = logging.getLogger(__name__)


def sanitize_audio_input(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Sanitizes audio input by standardizing format, handling invalid values, and removing DC offset.
    
    Converts audio to float32, replaces NaN and infinite values with zero, and ensures the output is stereo with shape (2, N). Mono audio is duplicated to stereo, and if more than two channels are present, only the first two are used. Applies a 5 Hz high-pass filter to remove significant DC offset if detected in either channel.
    
    Args:
        audio: Input audio array of any shape.
        sr: Sample rate in Hz.
    
    Returns:
        A sanitized stereo audio array with shape (2, N) and dtype float32.
    """
    # Convert to float32
    audio = audio.astype(np.float32)

    # Handle NaN/Inf values
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure we have the right shape (channels, samples)
    if audio.ndim == 1:
        # Mono to stereo (duplicate)
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        if audio.shape[0] > audio.shape[1]:
            # Transpose if samples x channels
            audio = audio.T

        if audio.shape[0] == 1:
            # Mono to stereo (duplicate)
            audio = np.tile(audio, (2, 1))
        elif audio.shape[0] > 2:
            # Take first two channels with warning
            logger.warning(f"Audio has {audio.shape[0]} channels, taking first 2")
            audio = audio[:2, :]
    else:
        raise ValueError(f"Invalid audio shape: {audio.shape}")

    # DC offset removal with high-pass filter at 5Hz
    nyquist = sr / 2
    dc_left = float(np.mean(audio[0]))
    dc_right = float(np.mean(audio[1]))
    # Only apply if noticeable DC present to avoid altering pristine test tones
    if nyquist > 5 and (abs(dc_left) > 0.2 or abs(dc_right) > 0.2):
        sos = butter(2, 5 / nyquist, btype="high", output="sos")
        audio[0] = sosfilt(sos, audio[0])
        audio[1] = sosfilt(sos, audio[1])

    return audio


def analyze_audio_file(input_audio_path: str, app_config: AppConfig) -> AnalysisResult:
    """
    Performs end-to-end analysis of an audio file and returns structured analysis results.
    
    Attempts to load and validate the audio file, checking for existence, file size, and duration limits as specified in the application configuration. If validation fails, returns an `AnalysisResult` with default values and an error message. On success, loads and sanitizes the audio, computes global audio metrics, performs sliding window segment analysis, detects musical sections, and generates an AI-based summary. All results are returned in an `AnalysisResult` object.
    """
    logger.info(f"Starting analysis of: {input_audio_path}")

    # Load and validate file
    try:
        info = sf.info(input_audio_path)
        file_size_mb = Path(input_audio_path).stat().st_size / (1024 * 1024)

        # Validate file constraints
        if file_size_mb > app_config.MAX_UPLOAD_MB:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB > {app_config.MAX_UPLOAD_MB}MB"
            )

        duration_min = info.frames / info.samplerate / 60
        if duration_min > app_config.MAX_DURATION_MIN:
            raise ValueError(
                f"File too long: {duration_min:.1f}min > {app_config.MAX_DURATION_MIN}min"
            )

        file_info = FileInfoModel(
            filename=Path(input_audio_path).name,
            duration_seconds=info.frames / info.samplerate,
            samplerate=info.samplerate,
            channels=info.channels,
            frames=info.frames,
            format_name=info.format,
            subtype_name=info.subtype,
            file_size_mb=file_size_mb,
            exists=True,
        )

        # Load audio data
        audio, sr = sf.read(input_audio_path, always_2d=False)

    except Exception as e:
        logger.error(f"Failed to load audio file: {e}")
        err_msg = str(e)
        if not Path(input_audio_path).exists():
            err_msg = f"No such file: {input_audio_path}"
        file_info = FileInfoModel(
            filename=Path(input_audio_path).name,
            duration_seconds=0.0,
            samplerate=44100,
            channels=2,
            frames=1,
            format_name="",
            subtype_name="",
            exists=False,
            error_message=err_msg,
        )
        # Return minimal result for failed file
        return AnalysisResult(
            file_info=file_info,
            global_metrics=GlobalMetricsModel(
                integrated_lufs=-70.0,
                loudness_range_lra=0.0,
                true_peak_dbfs=-70.0,
                sample_peak_dbfs=-70.0,
                crest_factor_overall=0.0,
                peak_to_loudness_ratio_plr=0.0,
                stereo_correlation_low_band=0.0,
                stereo_correlation_mid_band=0.0,
                stereo_correlation_high_band=0.0,
                channel_balance_rms_db=0.0,
                channel_balance_peak_db=0.0,
                spectral_centroid_hz=1000.0,
                spectral_bandwidth_hz=1000.0,
                spectral_contrast=1.0,
                spectral_flatness=0.5,
                spectral_rolloff_hz=5000.0,
                zero_crossing_rate=0.1,
                key_signature="C",
                key_confidence=0.0,
                tempo_bpm=120.0,
                transient_density=1.0,
                transient_strength_avg_db=0.5,
            ),
            sliding_window_analysis=SlidingWindowAnalysisModel(
                segment_length_sec=4.0,
                overlap_ratio=0.5,
                segments=[],
            ),
            musical_section_analysis=MusicalSectionAnalysisModel(sections=[]),
            ai_summary_input=AISummaryModel(
                overall_assessment="File could not be analyzed",
                global_metric_highlights=[],
                problematic_segments_summary=[],
                musical_structure_summary="Unknown",
            ),
        )

    # Sanitize audio
    audio = sanitize_audio_input(audio, sr)

    logger.info("Audio loaded and sanitized successfully")

    # Phase 1.2 - Calculate global metrics using real analysis
    from .metrics import get_all_global_metrics

    logger.info("Calculating real global metrics...")
    metrics_dict = get_all_global_metrics(audio, sr)

    global_metrics = GlobalMetricsModel(**metrics_dict)

    from .segmentation import sliding_window_segment_orchestrator

    # Phase 1.3 - Sliding window analysis
    sliding_window_analysis = sliding_window_segment_orchestrator(
        audio, sr, global_metrics
    )

    from .musical_sections import detect_musical_sections
    from .ai_summary import generate_ai_analysis_summary

    # Phase 1.4 - Musical section analysis
    musical_section_analysis = detect_musical_sections(audio, sr)

    analysis_result = AnalysisResult(
        file_info=file_info,
        global_metrics=global_metrics,
        sliding_window_analysis=sliding_window_analysis,
        musical_section_analysis=musical_section_analysis,
        ai_summary_input=None,
    )

    # Phase 1.5 - AI summary generation
    ai_summary = generate_ai_analysis_summary(analysis_result)
    analysis_result.ai_summary_input = ai_summary

    logger.info("Analysis completed successfully")

    return analysis_result
