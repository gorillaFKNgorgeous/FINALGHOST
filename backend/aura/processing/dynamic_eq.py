import numpy as np
import scipy.signal as signal

from aura.schemas import DynamicEQSettingsModel


def _peak_envelope(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Peak envelope follower with attack/release coefficients."""
    env = np.zeros_like(x)
    for i, sample in enumerate(x):
        prev = env[i - 1] if i else 0.0
        coeff = attack if abs(sample) > prev else release
        env[i] = coeff * prev + (1 - coeff) * abs(sample)
    return env


def _process_band(channel: np.ndarray, band, sr: int) -> np.ndarray:
    """Process a single dynamic EQ band on one channel."""
    b, a = signal.iirpeak(band.freq_hz / (sr / 2), Q=band.q)
    attack = np.exp(-1.0 / (band.attack_ms / 1000 * sr))
    release = np.exp(-1.0 / (band.release_ms / 1000 * sr))
    thr = 10 ** (band.threshold_db / 20)
    rat = band.ratio

    filtered = signal.lfilter(b, a, channel)
    env = _peak_envelope(filtered, attack, release)
    gain = np.ones_like(env)
    over = env > thr
    knee = getattr(band, "knee_db", None)
    if knee is not None:
        knee = 10 ** (knee / 20)
        t = np.clip((env - thr) / (knee - thr + 1e-9), 0, 1)
        comp = (thr + (env - thr) / rat) / env
        gain[over] = (1 - t[over]) + t[over] * comp[over]
    else:
        gain[over] = (thr + (env[over] - thr) / rat) / env[over]
    processed = channel * gain
    if band.gain_db:
        processed *= 10 ** (band.gain_db / 20)
    return processed


def apply_dynamic_eq(audio: np.ndarray, sr: int, settings: DynamicEQSettingsModel) -> np.ndarray:
    """Multi-band dynamic EQ with per-band threshold and ratio."""
    if not settings.bands:
        return audio

    work = audio.astype(np.float64)
    if work.ndim == 1:
        work = work[np.newaxis, :]

    for band in settings.bands:
        for ch in range(work.shape[0]):
            work[ch] = _process_band(work[ch], band, sr)

    out = work.astype(np.float32)
    return out if audio.ndim > 1 else out[0]
