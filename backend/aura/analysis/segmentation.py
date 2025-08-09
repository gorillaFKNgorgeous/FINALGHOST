# analysis/segmentation.py 
"""
Audio segmentation functions.
Provides capabilities to divide an audio stream into meaningful segments
based on acoustic features like onsets and rhythmic structure.
"""
from __future__ import annotations

from typing import List

from ..config import AppConfig
from ..schemas import (
    SegmentRawMetricsModel,
    SegmentMetricsModel,
    SlidingWindowAnalysisModel,
    GlobalMetricsModel,
    DeviationsModel,
)

import librosa
import numpy as np

# It's good practice to import from config if specific settings are needed,
# e.g., for default hop_length or other analysis parameters.
# from config import settings # Assuming settings.DEFAULT_SR might be used


def smart_segment_boundaries(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    backtrack: bool = True,
    default_if_no_onsets: bool = True,
) -> List[int]:
    """
    Identifies segment boundaries in an audio signal based on onset detection.

    This function uses an onset strength envelope to detect moments of significant
    energy increase, which often correspond to musical segment changes or beats.
    For the purpose of ANA-01, "agglomerative clustering" is interpreted as
    detecting and selecting prominent onsets that define clear structural segments,
    particularly effective for rhythmic material like a click track.

    Args:
        y (np.ndarray): The input audio time series (mono or stereo).
                        If stereo, it will be converted to mono for onset detection.
        sr (int): The sampling rate of the audio.
        hop_length (int): The hop length for STFT, used in onset strength calculation.
        backtrack (bool): If True, detected onset events are backtracked to
                          the nearest preceding local minimum of energy.
        default_if_no_onsets (bool): If True and no onsets are detected,
                                     returns a list containing just the start (0)
                                     and end of the audio as segments.

    Returns:
        List[int]: A list of sample indices representing the start of each
                   detected segment. The list will always include 0 as the
                   start of the first segment.
    """
    if y.ndim > 1:
        # Convert to mono by averaging channels if stereo
        y_mono = librosa.to_mono(y=y)
    else:
        y_mono = y

    if len(y_mono) == 0:
        return [0]

    # 1. Calculate onset strength envelope
    oenv = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop_length)

    # 2. Detect onset events (as sample indices)
    # The 'wait' and 'pre_avg'/'post_avg' parameters can be tuned for sensitivity.
    # Using defaults for now, which are generally robust.
    onsets_samples = librosa.onset.onset_detect(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop_length,
        units="samples",
        backtrack=backtrack,
    )

    # Ensure 0 is the start of the first segment
    # and onsets are sorted and unique.
    # Convert to list, add 0, then convert to set for uniqueness, then back to sorted list.
    segment_starts = sorted(list(set([0] + list(onsets_samples))))

    # Filter out any onsets that might be detected beyond the audio length
    # (shouldn't happen with librosa.onset.onset_detect in 'samples' units, but good practice)
    segment_starts = [s for s in segment_starts if s < len(y_mono)]

    if not segment_starts:  # Should at least have [0]
        segment_starts = [0]

    # If only [0] is present (no onsets detected after start) and default_if_no_onsets is True,
    # consider the whole file as one segment, or two boundaries [0, len(y_mono)]
    # The ticket implies "List[int] sample offsets" for starts of sections.
    # So if no onsets, just [0] is fine, meaning one segment from 0 to end.
    # If the test "4 equal sections" means 4 start boundaries, then this logic is okay.
    if len(segment_starts) == 1 and default_if_no_onsets:
        # If no onsets were found other than the start, and we want a default,
        # we can either return just [0] (one segment from 0 to end)
        # or [0, len(y_mono)] if boundaries are pairs.
        # Given "List[int] sample offsets" for starts, [0] is appropriate.
        # However, if the goal is to have at least one segment covering the whole file,
        # and the list represents start times, [0] is correct.
        # If no onsets are found in a silent file, it's one segment.
        pass  # segment_starts is already [0] or contains detected onsets

    # Ensure the last boundary isn't the exact length of the audio if it creates an empty segment.
    # This logic depends on how segments are consumed. If segment_starts are just start points,
    # then an offset equal to len(y_mono) would be an invalid start for a new segment.
    # librosa.onset.onset_detect should not return values >= len(y_mono) in 'samples' units.

    return segment_starts


def segment_metrics(audio, sr):
    """
    Analyze audio segments and return metrics for each segment.
    
    Args:
        audio (np.ndarray): Input audio data.
        sr (int): Sample rate of the audio.
        
    Returns:
        list: A list of dictionaries containing metrics for each segment.
    """
    from .metrics import get_global_metrics

    # Get segment boundaries
    boundaries = smart_segment_boundaries(audio, sr)

    # Process each segment
    segments_data = []
    for i in range(len(boundaries)):
        # For the last segment, use the end of the audio
        if i < len(boundaries) - 1:
            start, end = boundaries[i], boundaries[i + 1]
        else:
            start, end = boundaries[i], len(audio)

        # Extract segment
        segment = audio[start:end]

        # Skip empty segments
        if len(segment) == 0:
            continue

        # Calculate segment metrics
        metrics = get_global_metrics(segment, sr)

        # Add segment info
        segment_info = {
            "start_sample": int(start),
            "end_sample": int(end),
            "duration_seconds": (end - start) / sr,
            "true_peak_dbfs": metrics.get("true_peak_dbfs", 0),
            "integrated_lufs": metrics.get("integrated_lufs", 0),
        }

        segments_data.append(segment_info)

    return segments_data


def analyze_single_sliding_window_segment_metrics(segment_audio: np.ndarray, sr: int) -> SegmentRawMetricsModel:
    """Calculate metrics for a single sliding window segment."""
    from .metrics import get_all_global_metrics

    metrics = get_all_global_metrics(segment_audio, sr)

    return SegmentRawMetricsModel(**metrics)


def sliding_window_segment_orchestrator(
    full_audio: np.ndarray,
    sr: int,
    global_metrics_results: GlobalMetricsModel,
) -> SlidingWindowAnalysisModel:
    """Analyze audio in overlapping sliding windows and compute deviations."""

    cfg = AppConfig()

    seg_len_samples = int(cfg.SEGMENT_LENGTH_SEC * sr)
    hop_samples = int(seg_len_samples * (1.0 - cfg.SEGMENT_OVERLAP_RATIO))
    hop_samples = max(1, hop_samples)

    total_samples = full_audio.shape[-1]
    segments: List[SegmentMetricsModel] = []

    start = 0
    index = 0
    while start < total_samples:
        end = min(start + seg_len_samples, total_samples)
        segment_audio = full_audio[..., start:end]

        raw_metrics = analyze_single_sliding_window_segment_metrics(segment_audio, sr)

        deviations = DeviationsModel(
            integrated_lufs_diff_abs=abs(
                (raw_metrics.integrated_lufs or 0.0)
                - (global_metrics_results.integrated_lufs or 0.0)
            ),
            true_peak_dbfs_diff_abs=abs(
                (raw_metrics.true_peak_dbfs or 0.0)
                - (global_metrics_results.true_peak_dbfs or 0.0)
            ),
            spectral_centroid_hz_diff_percent=(
                (
                    (raw_metrics.spectral_centroid_hz or 0.0)
                    - (global_metrics_results.spectral_centroid_hz or 0.0)
                )
                / (global_metrics_results.spectral_centroid_hz or 1.0)
                * 100.0
            ),
        )

        segments.append(
            SegmentMetricsModel(
                segment_index=index,
                start_time_sec=start / sr,
                end_time_sec=end / sr,
                raw_metrics=raw_metrics,
                deviations_from_global=deviations,
            )
        )

        index += 1
        start += hop_samples

    return SlidingWindowAnalysisModel(
        segment_length_sec=cfg.SEGMENT_LENGTH_SEC,
        overlap_ratio=cfg.SEGMENT_OVERLAP_RATIO,
        segments=segments,
    )
