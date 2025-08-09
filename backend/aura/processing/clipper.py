import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter1d

from aura.schemas import LookaheadClipperSettingsModel


def apply_lookahead_clipper(audio: np.ndarray, sr: int, settings: LookaheadClipperSettingsModel) -> np.ndarray:
    """True peak clipper with lookahead buffer and waveshaping modes."""
    work = audio.astype(np.float64)
    if work.ndim == 1:
        work = work[np.newaxis, :]

    os_factor = getattr(settings, "oversample_factor", 2)
    look = int(sr * settings.lookahead_ms / 1000 * os_factor)
    thr = 10 ** (settings.threshold_db / 20)
    mode = getattr(settings, "mode", "hard")

    def shape(x: np.ndarray) -> np.ndarray:
        if mode == "soft":
            return np.where(np.abs(x) < thr, x, np.sign(x) * thr + 0.5 * (x - np.sign(x) * thr))
        if mode == "tanh":
            return thr * np.tanh(x / thr)
        if mode == "tape":
            return thr * (2 / np.pi) * np.arctan(x / thr)
        return np.clip(x, -thr, thr)

    for ch in range(work.shape[0]):
        up = signal.resample(work[ch], len(work[ch]) * os_factor)
        abs_up = np.abs(up)
        env = maximum_filter1d(abs_up, size=look, mode="nearest")
        gain = np.minimum(1.0, thr / (env + 1e-12))
        processed = shape(up * gain)
        down = signal.resample(processed, work.shape[1])
        work[ch] = down

    out = work.astype(np.float32)
    return out if audio.ndim > 1 else out[0]
