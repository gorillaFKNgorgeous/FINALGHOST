"""
Aura Analysis Metrics - Core audio measurement functions
"""

import logging
import numpy as np
import librosa
import pyloudnorm as pyln
import scipy.signal as signal
from typing import Dict, Any, Tuple, Optional

# torch removed - using scipy for resampling instead

logger = logging.getLogger(__name__)


def _integrated_lufs_array(audio: np.ndarray, sr: int) -> float:
    """Return integrated loudness in LUFS for a numpy array."""
    meter = pyln.Meter(sr)
    if audio.ndim == 1:
        # pyloudnorm accepts 1-D mono – keep it mono to avoid a +3 dB bias
        audio_formatted = audio
    elif audio.shape[0] in (1, 2):
        audio_formatted = audio.T
    else:
        audio_formatted = audio
    loudness = meter.integrated_loudness(audio_formatted)
    if np.isnan(loudness) or np.isinf(loudness):
        raise ValueError("Invalid loudness")
    return float(loudness)


def measure_integrated_lufs(audio: np.ndarray, sr: int) -> float:
    """
    Measure integrated LUFS using pyloudnorm

    Args:
        audio: Stereo audio (2, N) or mono (N,)
        sr: Sample rate

    Returns:
        Integrated LUFS value
    """
    try:
        return _integrated_lufs_array(audio, sr)

    except Exception as e:
        logger.error(f"LUFS measurement failed: {e}")
        return -70.0


def measure_lra(audio: np.ndarray, sr: int) -> float:
    """
    Approximate Loudness Range (LRA) as in EBU-R128:
    1. Convert to mono RMS in LUFS over 3-second windows with 0.1-s hop.
    2. Discard windows below –70 LUFS (silence gate).
    3. Return difference between 95-th and 10-th percentiles.
    Falls back to 0.0 if track <3 s or <5 valid windows.
    """
    try:
        if audio.shape[-1] < 3 * sr:
            return 0.0
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
        win = int(3 * sr)
        hop = int(0.1 * sr)
        lufs_vals = []
        for start in range(0, len(mono) - win + 1, hop):
            seg = mono[start : start + win]
            if np.max(np.abs(seg)) < 1e-7:
                continue
            lufs_vals.append(_integrated_lufs_array(seg, sr))
        lufs_vals = np.array(lufs_vals)
        lufs_vals = lufs_vals[lufs_vals > -70]
        if len(lufs_vals) < 5:
            return 0.0
        return float(np.percentile(lufs_vals, 95) - np.percentile(lufs_vals, 10))
    except Exception as e:
        logger.warning("LRA measurement failed: %s", e)
        return 0.0


def measure_true_peak(audio: np.ndarray, sr: int) -> float:
    """
    Measure true peak using 4x oversampling

    Args:
        audio: Stereo audio (2, N) or mono (N,)
        sr: Sample rate

    Returns:
        True peak in dBFS
    """
    try:
        # Ensure correct audio format
        if audio.ndim == 1:
            audio_work = audio.reshape(1, -1)
        else:
            audio_work = audio

        # Simple 4x oversampling using scipy
        upsampled_channels = []
        for ch in range(audio_work.shape[0]):
            # Use scipy.signal.resample for upsampling
            upsampled = signal.resample(audio_work[ch], len(audio_work[ch]) * 4)
            upsampled_channels.append(upsampled)

        # Find maximum absolute value across all channels
        if upsampled_channels:
            all_peaks = [np.max(np.abs(ch)) for ch in upsampled_channels]
            true_peak_linear = max(all_peaks)
        else:
            true_peak_linear = 0.0

        # Convert to dBFS
        if true_peak_linear > 0:
            true_peak_db = 20 * np.log10(true_peak_linear)
        else:
            true_peak_db = -70.0

        return float(true_peak_db)

    except Exception as e:
        logger.error(f"True peak measurement failed: {e}")
        return -70.0


def measure_sample_peak(audio: np.ndarray) -> float:
    """
    Measure sample peak in dBFS

    Args:
        audio: Audio array

    Returns:
        Sample peak in dBFS
    """
    try:
        peak_linear = np.max(np.abs(audio))

        if peak_linear > 0:
            peak_db = 20 * np.log10(peak_linear)
        else:
            peak_db = -70.0

        return float(peak_db)

    except Exception as e:
        logger.error(f"Sample peak measurement failed: {e}")
        return -70.0


def measure_crest_factor(audio: np.ndarray) -> float:
    """
    Measure crest factor (peak to RMS ratio) in dB

    Args:
        audio: Audio array

    Returns:
        Crest factor in dB
    """
    try:
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))

        # Calculate peak
        peak = np.max(np.abs(audio))

        if rms > 0 and peak > 0:
            crest_factor = 20 * np.log10(peak / rms)
        else:
            crest_factor = 0.0

        return float(crest_factor)

    except Exception as e:
        logger.error(f"Crest factor measurement failed: {e}")
        return 0.0


def measure_plr(audio: np.ndarray, sr: int) -> float:
    """
    Measure Peak to Loudness Ratio (PLR)

    Args:
        audio: Stereo audio (2, N)
        sr: Sample rate

    Returns:
        PLR in dB
    """
    try:
        # Get true peak
        true_peak = measure_true_peak(audio, sr)

        # Get integrated loudness
        integrated_lufs = measure_integrated_lufs(audio, sr)

        # Calculate PLR
        plr = true_peak - integrated_lufs

        return float(plr)

    except Exception as e:
        logger.error(f"PLR measurement failed: {e}")
        return 0.0


def measure_multiband_stereo_correlation(
    audio: np.ndarray, sr: int
) -> Tuple[float, float, float]:
    """
    Measure stereo correlation in low, mid, and high frequency bands

    Args:
        audio: Stereo audio (2, N)
        sr: Sample rate

    Returns:
        Tuple of (low_corr, mid_corr, high_corr)
    """
    try:
        if audio.shape[0] != 2:
            return (0.0, 0.0, 0.0)

        left, right = audio[0], audio[1]

        # Define frequency bands
        nyquist = sr / 2
        low_cutoff = 250 / nyquist
        high_cutoff = 4000 / nyquist

        # Design filters
        sos_low = signal.butter(4, low_cutoff, btype="low", output="sos")
        sos_band = signal.butter(
            4, [low_cutoff, high_cutoff], btype="band", output="sos"
        )
        sos_high = signal.butter(4, high_cutoff, btype="high", output="sos")

        # Filter into bands
        left_low = signal.sosfilt(sos_low, left)
        right_low = signal.sosfilt(sos_low, right)

        left_mid = signal.sosfilt(sos_band, left)
        right_mid = signal.sosfilt(sos_band, right)

        left_high = signal.sosfilt(sos_high, left)
        right_high = signal.sosfilt(sos_high, right)

        # Calculate correlations
        def calc_correlation(l, r):
            if len(l) < 2 or len(r) < 2:
                return 0.0
            corr_matrix = np.corrcoef(l, r)
            return float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        low_corr = calc_correlation(left_low, right_low)
        mid_corr = calc_correlation(left_mid, right_mid)
        high_corr = calc_correlation(left_high, right_high)

        return (low_corr, mid_corr, high_corr)

    except Exception as e:
        logger.error(f"Multiband stereo correlation measurement failed: {e}")
        return (0.0, 0.0, 0.0)


def measure_stereo_correlation_overall(audio: np.ndarray) -> float:
    """Measure overall stereo correlation for a two-channel signal."""
    try:
        if audio.ndim == 1:
            return 1.0
        if audio.shape[0] != 2:
            return 0.0

        left, right = audio[0], audio[1]
        if len(left) < 2 or len(right) < 2:
            return 0.0

        corr = np.corrcoef(left, right)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)

    except Exception as e:
        logger.error(f"Stereo correlation measurement failed: {e}")
        return 0.0


def measure_channel_balance(audio: np.ndarray) -> Tuple[float, float]:
    """
    Measure L/R channel balance for RMS and peak

    Args:
        audio: Stereo audio (2, N)

    Returns:
        Tuple of (rms_balance_db, peak_balance_db)
    """
    try:
        if audio.shape[0] != 2:
            return (0.0, 0.0)

        left, right = audio[0], audio[1]

        # RMS balance
        left_rms = np.sqrt(np.mean(left**2))
        right_rms = np.sqrt(np.mean(right**2))

        if left_rms > 0 and right_rms > 0:
            rms_balance = 20 * np.log10(left_rms / right_rms)
        else:
            rms_balance = 0.0

        # Peak balance
        left_peak = np.max(np.abs(left))
        right_peak = np.max(np.abs(right))

        if left_peak > 0 and right_peak > 0:
            peak_balance = 20 * np.log10(left_peak / right_peak)
        else:
            peak_balance = 0.0

        return (float(rms_balance), float(peak_balance))

    except Exception as e:
        logger.error(f"Channel balance measurement failed: {e}")
        return (0.0, 0.0)


def measure_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """
    Measure spectral centroid using librosa

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Spectral centroid in Hz
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=sr)[0]
        return float(np.mean(centroid))

    except Exception as e:
        logger.error(f"Spectral centroid measurement failed: {e}")
        return 1000.0


def measure_spectral_bandwidth(audio: np.ndarray, sr: int) -> float:
    """
    Measure spectral bandwidth using librosa

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Spectral bandwidth in Hz
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sr)[0]
        return float(np.mean(bandwidth))

    except Exception as e:
        logger.error(f"Spectral bandwidth measurement failed: {e}")
        return 1000.0


def measure_spectral_contrast(audio: np.ndarray, sr: int) -> float:
    """
    Measure spectral contrast using librosa

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Mean spectral contrast
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        contrast = librosa.feature.spectral_contrast(y=audio_mono, sr=sr)
        return float(np.mean(contrast))

    except Exception as e:
        logger.error(f"Spectral contrast measurement failed: {e}")
        return 1.0


def measure_spectral_flatness(audio: np.ndarray, sr: int) -> float:
    """
    Measure spectral flatness (Wiener entropy)

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Spectral flatness (0-1)
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        flatness = librosa.feature.spectral_flatness(y=audio_mono)[0]
        return float(np.mean(flatness))

    except Exception as e:
        logger.error(f"Spectral flatness measurement failed: {e}")
        return 0.5


def measure_spectral_rolloff(audio: np.ndarray, sr: int) -> float:
    """
    Measure spectral rolloff frequency

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Spectral rolloff frequency in Hz
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=sr)[0]
        return float(np.mean(rolloff))

    except Exception as e:
        logger.error(f"Spectral rolloff measurement failed: {e}")
        return 5000.0


def measure_zero_crossing_rate(audio: np.ndarray) -> float:
    """
    Measure zero crossing rate

    Args:
        audio: Audio array (mono or average of stereo)

    Returns:
        Zero crossing rate
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        zcr = librosa.feature.zero_crossing_rate(y=audio_mono)[0]
        return float(np.mean(zcr))

    except Exception as e:
        logger.error(f"Zero crossing rate measurement failed: {e}")
        return 0.1


def detect_key_and_confidence(audio: np.ndarray, sr: int) -> Tuple[str, float]:
    """
    Detect musical key and confidence using librosa

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Tuple of (key_name, confidence)
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=audio_mono, sr=sr)

        # Average over time
        chroma_mean = np.mean(chroma, axis=1)

        # Find the most prominent pitch class
        key_idx = np.argmax(chroma_mean)

        # Map to key names
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        detected_key = key_names[key_idx]

        # Calculate confidence as the ratio of max to mean
        confidence = float(chroma_mean[key_idx] / (np.mean(chroma_mean) + 1e-8))
        confidence = min(1.0, confidence / 3.0)  # Normalize roughly to 0-1

        return (detected_key, confidence)

    except Exception as e:
        logger.error(f"Key detection failed: {e}")
        return ("C", 0.0)


def detect_tempo_bpm(audio: np.ndarray, sr: int) -> float:
    """
    Detect tempo using librosa

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Tempo in BPM
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Check if signal is too short or simple for tempo detection
        if len(audio_mono) < sr * 2:  # Less than 2 seconds
            logger.warning("Audio too short for reliable tempo detection")
            return 120.0

        # Use more conservative parameters for simple signals
        hop_length = 512

        # Suppress warnings about n_fft by setting explicit hop_length
        with np.errstate(all="ignore"):
            tempo, _ = librosa.beat.beat_track(
                y=audio_mono,
                sr=sr,
                hop_length=hop_length,
                start_bpm=120.0,
                tightness=100,
            )

        # Validate tempo range
        if tempo < 30.0 or tempo > 300.0:
            logger.warning(f"Detected tempo {tempo} outside valid range, using default")
            return 120.0

        return float(tempo)

    except Exception as e:
        logger.error(f"Tempo detection failed: {e}")
        return 120.0


def calculate_onset_density_and_strength(
    audio: np.ndarray, sr: int
) -> Tuple[float, float]:
    """
    Calculate onset density and transient strength

    Args:
        audio: Audio array (mono or average of stereo)
        sr: Sample rate

    Returns:
        Tuple of (onset_density_per_sec, transient_strength)
    """
    try:
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=audio_mono, sr=sr)

        # Calculate onset density (onsets per second)
        duration = len(audio_mono) / sr
        onset_density = len(onset_frames) / duration if duration > 0 else 0.0

        # Calculate onset strength envelope
        onset_envelope = librosa.onset.onset_strength(y=audio_mono, sr=sr)

        # Transient strength as mean of onset envelope
        transient_strength = float(np.mean(onset_envelope))

        return (float(onset_density), transient_strength)

    except Exception as e:
        logger.error(f"Onset analysis failed: {e}")
        return (1.0, 0.5)


def get_all_global_metrics(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Calculate all global metrics for an audio file

    Args:
        audio: Stereo audio (2, N) or mono (N,)
        sr: Sample rate

    Returns:
        Dictionary containing all global metrics
    """
    logger.info("Calculating comprehensive global metrics...")

    # Ensure stereo format (2, N)
    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio], axis=0)
    else:
        audio_stereo = audio

    # Loudness and peak metrics
    integrated_lufs = measure_integrated_lufs(audio_stereo, sr)
    lra_lu = measure_lra(audio_stereo, sr)
    true_peak_db = measure_true_peak(audio_stereo, sr)
    sample_peak_db = measure_sample_peak(audio_stereo)

    # Dynamic range metrics
    crest_factor_db = measure_crest_factor(audio_stereo)
    plr_db = measure_plr(audio_stereo, sr)

    # Stereo image metrics
    stereo_corr_low, stereo_corr_mid, stereo_corr_high = (
        measure_multiband_stereo_correlation(audio_stereo, sr)
    )
    stereo_corr_overall = measure_stereo_correlation_overall(audio_stereo)
    channel_balance_rms_db, channel_balance_peak_db = measure_channel_balance(
        audio_stereo
    )

    # Spectral metrics
    spectral_centroid_hz = measure_spectral_centroid(audio_stereo, sr)
    spectral_bandwidth_hz = measure_spectral_bandwidth(audio_stereo, sr)
    spectral_contrast = measure_spectral_contrast(audio_stereo, sr)
    spectral_flatness = measure_spectral_flatness(audio_stereo, sr)
    spectral_rolloff_hz = measure_spectral_rolloff(audio_stereo, sr)
    zero_crossing_rate = measure_zero_crossing_rate(audio_stereo)

    # Musical feature metrics
    detected_key, key_confidence = detect_key_and_confidence(audio_stereo, sr)
    tempo_bpm = detect_tempo_bpm(audio_stereo, sr)
    onset_density, transient_strength = calculate_onset_density_and_strength(
        audio_stereo, sr
    )

    return {
        "integrated_lufs": integrated_lufs,
        "loudness_range_lra": lra_lu,
        "true_peak_dbfs": true_peak_db,
        "sample_peak_dbfs": sample_peak_db,
        "crest_factor_overall": crest_factor_db,
        "peak_to_loudness_ratio_plr": plr_db,
        "stereo_correlation_overall": stereo_corr_overall,
        "stereo_correlation_low_band": stereo_corr_low,
        "stereo_correlation_mid_band": stereo_corr_mid,
        "stereo_correlation_high_band": stereo_corr_high,
        "channel_balance_rms_db": channel_balance_rms_db,
        "channel_balance_peak_db": channel_balance_peak_db,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
        "spectral_contrast": spectral_contrast,
        "spectral_flatness": spectral_flatness,
        "spectral_rolloff_hz": spectral_rolloff_hz,
        "zero_crossing_rate": zero_crossing_rate,
        "key_signature": detected_key,
        "key_confidence": key_confidence,
        "tempo_bpm": tempo_bpm,
        "transient_density": onset_density,
        "transient_strength_avg_db": transient_strength,
    }


def get_global_metrics(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Compatibility helper returning a minimal metric set."""
    metrics = get_all_global_metrics(audio, sr)
    return {
        "integrated_lufs": metrics.get("integrated_lufs", -70.0),
        "true_peak_dbfs": metrics.get("true_peak_dbfs", -70.0),
    }
