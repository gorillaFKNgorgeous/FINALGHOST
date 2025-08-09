"""Musical section detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import librosa

from .segmentation import smart_segment_boundaries
from ..schemas import MusicalSectionAnalysisModel, SectionModel


def _make_section(
    name: str, start_sample: int, end_sample: int, sr: int, conf: float = 1.0
) -> SectionModel:
    """Helper to create a SectionModel with time conversion."""
    return SectionModel(
        name=name,
        start_time_sec=start_sample / sr,
        end_time_sec=end_sample / sr,
        confidence=conf,
    )


@dataclass
class HeuristicSection:
    """Internal representation of a detected musical section in samples."""

    name: str
    start_sample: int
    end_sample: int



def detect_musical_sections(audio: np.ndarray, sample_rate: int) -> MusicalSectionAnalysisModel:
    """
    Detects musical sections in an audio signal using onset-derived boundaries.

    Divides the input audio into sequentially numbered sections based on detected onset boundaries, assigning each a generic name and start/end times in seconds. Returns a model containing the list of detected sections, primarily for use in unit testing.

    Args:
    	audio: Audio signal as a NumPy array.
    	sample_rate: Sampling rate of the audio in Hz.

    Returns:
    	A MusicalSectionAnalysisModel containing the detected sections.
    """

    boundaries = smart_segment_boundaries(audio, sample_rate)

    heuristic_sections: list[HeuristicSection] = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else audio.shape[-1]
        heuristic_sections.append(
            HeuristicSection(name=f"Section {idx + 1}", start_sample=start, end_sample=end)
        )

    sections = [
        _make_section(sec.name, sec.start_sample, sec.end_sample, sample_rate)
        for sec in heuristic_sections
    ]

    return MusicalSectionAnalysisModel(sections=sections)


def _simple_boundaries(y: np.ndarray, sr: int, hop_length: int) -> List[int]:
    """Fallback: evenly spaced 6-second boundaries."""

    step = 6 * sr
    bounds = list(range(0, len(y), step))
    if bounds[-1] != len(y):
        bounds.append(len(y))
    return bounds


def _determine_boundaries(y: np.ndarray, sr: int, hop_length: int) -> List[int]:
    """Return cleaned, snapped segment boundaries."""

    boundaries = smart_segment_boundaries(y, sr, hop_length)
    if len(boundaries) <= 1:
        boundaries = _simple_boundaries(y, sr, hop_length)

    boundaries = sorted(set(boundaries))
    step = 6 * sr
    snapped = []
    last = None
    for b in boundaries:
        b_snap = int(round(b / step) * step)
        b_snap = max(0, min(b_snap, len(y)))
        if last is None or b_snap - last >= sr:
            snapped.append(b_snap)
            last = b_snap

    if snapped[0] != 0:
        snapped.insert(0, 0)
    if snapped[-1] != len(y):
        snapped.append(len(y))

    return sorted(set(snapped))


def _extract_segment_features(
    y_mono: np.ndarray,
    sr: int,
    hop_length: int,
    frame_boundaries: List[int],
) -> np.ndarray:
    """Compute aggregated chroma and tempo features for each segment."""

    chroma = librosa.feature.chroma_cens(y=y_mono, sr=sr, hop_length=hop_length)
    temp = librosa.feature.tempogram(y=y_mono, sr=sr, hop_length=hop_length)
    feat = np.vstack([chroma, temp])

    agg = []
    for i in range(len(frame_boundaries) - 1):
        start = frame_boundaries[i]
        end = frame_boundaries[i + 1]
        sl = feat[:, start:end]
        if sl.size == 0:
            agg.append(np.zeros(feat.shape[0]))
        else:
            agg.append(np.mean(sl, axis=1))
    return np.array(agg)


def _cluster_segments(
    segment_features: np.ndarray,
    n_segments: int,
    max_clusters: int = 8,
) -> np.ndarray:
    """Cluster segments using agglomerative clustering."""

    from sklearn.cluster import AgglomerativeClustering

    n_clusters = min(max_clusters, max(2, n_segments // 2))
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return clustering.fit_predict(segment_features)

def _calculate_segment_properties(
    y_mono: np.ndarray, sr: int, boundaries: List[int]
) -> Tuple[List[float], List[float], List[float]]:
    """Return duration, RMS (dB), and spectral centroid for each segment."""

    dur = []
    rms_db = []
    centroid = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg = y_mono[start:end]
        dur.append((end - start) / sr)
        if len(seg) == 0:
            rms_db.append(-np.inf)
            centroid.append(0.0)
            continue
        rms = np.sqrt(np.mean(seg ** 2))
        rms_db.append(20 * np.log10(rms + 1e-10))
        cent = librosa.feature.spectral_centroid(y=seg, sr=sr)
        centroid.append(float(np.mean(cent)))

    return dur, rms_db, centroid


def _assign_heuristic_labels(
    cluster_labels: np.ndarray,
    segment_properties: Tuple[List[float], List[float], List[float]],
    chorus_cluster: int,
    repeat_counts: Dict[int, int],
) -> List[str]:
    """Assign heuristic section labels based on clusters and properties."""

    dur, rms_db, centroid = segment_properties
    labels = []
    for i, clust in enumerate(cluster_labels):
        label = "Verse"
        if clust == chorus_cluster and repeat_counts.get(clust, 0) > 1:
            label = "Chorus"
        elif i == 0 and dur[i] < 15 and rms_db[i] < -25:
            label = "Intro"
        elif i == len(cluster_labels) - 1 and rms_db[i] < -20:
            label = "Outro"
        elif repeat_counts.get(clust, 0) >= 2 and clust != chorus_cluster:
            label = "Verse"
        elif dur[i] < 8 and centroid[i] > 3000:
            label = "Bridge"
        labels.append(label)
    return labels


def _build_sections(
    boundaries: List[int],
    labels: List[str],
    cluster_labels: np.ndarray,
    repeat_counts: Dict[int, int],
    max_repeat: int,
    sr: int,
) -> List[SectionModel]:
    """Create dataclass sections with confidence values."""

    sections: List[SectionModel] = []
    for i, start in enumerate(boundaries[:-1]):
        end = boundaries[i + 1]
        repeat_count = repeat_counts.get(cluster_labels[i], 0)
        confidence = min(1.0, repeat_count / max_repeat) if max_repeat > 0 else 0.0
        sections.append(
            _make_section(labels[i], start, end, sr, float(confidence))
        )
    return sections


def analyze_musical_sections(y: np.ndarray, sr: int, hop_length: int = 512) -> List[SectionModel]:
    """Orchestrate musical section analysis using internal helpers.

    Args:
        y: Audio time series (mono or stereo).
        sr: Sampling rate of ``y``.
        hop_length: Hop length for STFT operations.

    Returns:
        List[SectionModel]: Detected sections sorted by start sample.
    """

    boundaries = _determine_boundaries(y, sr, hop_length)
    if len(boundaries) <= 2:
        return [_make_section("Full Track", 0, len(y), sr)]

    y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
    frame_boundaries = librosa.samples_to_frames(boundaries, hop_length=hop_length)

    segment_features = _extract_segment_features(y_mono, sr, hop_length, frame_boundaries)
    cluster_labels = _cluster_segments(segment_features, len(boundaries) - 1)

    dur, rms_db, centroid = _calculate_segment_properties(y_mono, sr, boundaries)

    from collections import Counter

    repeat_counts = Counter(cluster_labels)
    chorus_cluster = max(repeat_counts, key=repeat_counts.get)
    max_repeat = max(repeat_counts.values())

    labels = _assign_heuristic_labels(
        cluster_labels,
        (dur, rms_db, centroid),
        chorus_cluster,
        repeat_counts,
    )

    return _build_sections(
        boundaries,
        labels,
        cluster_labels,
        repeat_counts,
        max_repeat,
        sr,
    )
