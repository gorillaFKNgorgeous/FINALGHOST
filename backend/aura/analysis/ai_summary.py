from __future__ import annotations

"""Generate structured AI summaries of audio analysis."""

from typing import List, Tuple

from ..schemas import (
    AnalysisResult,
    AISummaryModel,
    GlobalMetricsModel,
    SlidingWindowAnalysisModel,
    MusicalSectionAnalysisModel,
)

LUFS_LOW_THRESH = -16.0
LUFS_HIGH_THRESH = -8.0
TRUE_PEAK_CLIP_DBFS = 0.0
CORRELATION_BAD = 0.10
LUFS_SEGMENT_DIFF = 3.0


def _describe_spectral_tilt(freq_hz: float | None) -> str:
    """
    Returns a textual description of spectral balance based on the provided frequency.
    
    Args:
        freq_hz: The average frequency in Hz representing spectral tilt, or None if unavailable.
    
    Returns:
        A string describing the spectral balance as unknown, slightly dark, balanced, or bright.
    """
    if freq_hz is None:
        return "unknown spectral balance"
    if freq_hz < 2500:
        return "slightly dark spectral balance"
    if freq_hz > 4000:
        return "bright spectral balance"
    return "balanced spectral tilt"


def _describe_stereo_width(corr: float | None) -> str:
    """
    Returns a textual description of stereo width based on the given stereo correlation value.
    
    Args:
        corr: Stereo correlation value, where lower values indicate wider stereo image and higher values indicate narrow or mono.
    
    Returns:
        A string describing the stereo width category, such as "wide", "good stereo width", or "narrow / near-mono". Returns "unknown stereo width" if the input is None.
    """
    if corr is None:
        return "unknown stereo width"
    if corr < 0.10:
        return "very wide / potentially phase-inverted"
    if corr < 0.40:
        return "wide"
    if corr < 0.75:
        return "good stereo width"
    return "narrow / near-mono"


def _find_section_for_range(
    ms: MusicalSectionAnalysisModel | None, start: float, end: float
) -> Tuple[str, str]:
    """
    Finds the musical section containing a given time range and returns its name and formatted time range.
    
    If no section contains the specified range, returns "Unlabelled section" with the formatted start and end times.
    
    Args:
        ms: Musical section analysis model, or None.
        start: Start time in seconds.
        end: End time in seconds.
    
    Returns:
        A tuple of (section name, formatted time range as "M:SS-M:SS").
    """
    def fmt(t: float) -> str:
        m, s = divmod(int(t + 0.5), 60)
        return f"{m}:{s:02d}"

    if ms and ms.sections:
        for sec in ms.sections:
            if sec.start_time_sec <= start and sec.end_time_sec >= end:
                return sec.name, f"{fmt(sec.start_time_sec)}-{fmt(sec.end_time_sec)}"
    return "Unlabelled section", f"{fmt(start)}-{fmt(end)}"


def generate_ai_analysis_summary(ar: AnalysisResult) -> AISummaryModel:
    """
    Generates a structured AI summary of audio analysis results, highlighting global metrics, problematic segments, and musical structure.
    
    Args:
        ar: The complete audio analysis result containing global metrics, sliding window analysis, and musical section analysis.
    
    Returns:
        An AISummaryModel with schema version 1.0, including an overall assessment, global metric highlights, a summary of problematic segments, and a musical structure summary.
    """

    gm: GlobalMetricsModel = ar.global_metrics
    sw: SlidingWindowAnalysisModel | None = ar.sliding_window_analysis
    ms: MusicalSectionAnalysisModel | None = ar.musical_section_analysis

    key_text = gm.key_signature or "unknown key"
    tempo_text = f"~{gm.tempo_bpm:.0f} BPM" if gm.tempo_bpm else "unknown tempo"
    spectral = _describe_spectral_tilt(gm.spectral_centroid_hz)
    stereo = _describe_stereo_width(gm.stereo_correlation_overall)
    overall = f"Track in {key_text}, {tempo_text}, {spectral}, {stereo}."

    highlights: List[str] = []
    if gm.integrated_lufs is not None:
        if gm.integrated_lufs < LUFS_LOW_THRESH:
            highlights.append(
                f"Integrated LUFS very low at {gm.integrated_lufs:.1f} LUFS."
            )
        elif gm.integrated_lufs > LUFS_HIGH_THRESH:
            highlights.append(
                f"Integrated LUFS very high at {gm.integrated_lufs:.1f} LUFS."
            )
    if gm.true_peak_dbfs is not None and gm.true_peak_dbfs > TRUE_PEAK_CLIP_DBFS:
        highlights.append(f"True Peak at +{gm.true_peak_dbfs:.2f} dBFS – clipping!")
    if (
        gm.stereo_correlation_overall is not None
        and abs(gm.stereo_correlation_overall) < CORRELATION_BAD
    ):
        highlights.append(
            f"Overall stereo correlation {gm.stereo_correlation_overall:.2f} (possible phase issues)."
        )

    problems: List[str] = []
    if sw and sw.segments:
        for seg in sw.segments:
            dev = seg.deviations_from_global
            if (
                dev.integrated_lufs_diff_abs
                and dev.integrated_lufs_diff_abs > LUFS_SEGMENT_DIFF
            ):
                sec_label, sec_range = _find_section_for_range(
                    ms, seg.start_time_sec, seg.end_time_sec
                )
                problems.append(
                    f"{sec_label} ({sec_range}) includes {seg.start_time_sec:.2f}-{seg.end_time_sec:.2f}s segment {seg.segment_index} "
                    f"with loudness diff {dev.integrated_lufs_diff_abs:+.1f} LU."
                )
            if (
                seg.raw_metrics.stereo_correlation_overall is not None
                and abs(seg.raw_metrics.stereo_correlation_overall) < CORRELATION_BAD
            ):
                sec_label, sec_range = _find_section_for_range(
                    ms, seg.start_time_sec, seg.end_time_sec
                )
                problems.append(
                    f"{sec_label} ({sec_range}) segment {seg.segment_index} stereo correlation {seg.raw_metrics.stereo_correlation_overall:.2f} (phase risk)."
                )

    if ms and ms.sections:
        names = [s.name.split()[0].capitalize() for s in ms.sections]
        structure = "–".join(names)
    else:
        structure = "Unknown"

    return AISummaryModel(
        schema_version="1.0",
        overall_assessment=overall,
        global_metric_highlights=highlights,
        problematic_segments_summary=problems,
        musical_structure_summary=structure,
    )
