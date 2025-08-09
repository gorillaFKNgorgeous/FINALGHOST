import numpy as np
from scipy import signal

from aura.schemas import DeesserSettingsModel
from .compressor import _envelope


def _design_deesser_filters(sr: int, settings: DeesserSettingsModel) -> tuple:
    """Return filter coefficients and detector parameters."""
    split_norm = settings.split_freq_hz / (sr / 2)
    sos_high = signal.butter(2, split_norm, btype="high", output="sos")
    sos_low = signal.butter(2, split_norm, btype="low", output="sos")
    atk = np.exp(-1.0 / (sr * settings.attack_ms / 1000))
    rel = np.exp(-1.0 / (sr * settings.release_ms / 1000))
    thr = 10 ** (settings.threshold_db / 20)
    reduction_gain = 10 ** (-settings.reduction_db / 20)
    return sos_high, sos_low, atk, rel, thr, reduction_gain


def _process_channel(
    channel: np.ndarray,
    sos_high: np.ndarray,
    sos_low: np.ndarray,
    atk: float,
    rel: float,
    thr: float,
    reduction_gain: float,
) -> np.ndarray:
    """Apply de-essing to a single channel."""
    hf = signal.sosfilt(sos_high, channel)
    lf = signal.sosfilt(sos_low, channel)
    env = _envelope(hf, atk, rel)
    gain = np.ones_like(env)
    over = env > thr
    gain[over] = reduction_gain
    hf *= gain
    return lf + hf


def apply_deesser(audio: np.ndarray, sr: int, settings: DeesserSettingsModel) -> np.ndarray:
    """Wideband de-esser with envelope follower."""
    work = audio.astype(np.float64)
    if work.ndim == 1:
        work = work[np.newaxis, :]

    sos_high, sos_low, atk, rel, thr, reduction_gain = _design_deesser_filters(
        sr, settings
    )

    for ch in range(work.shape[0]):
        work[ch] = _process_channel(
            work[ch], sos_high, sos_low, atk, rel, thr, reduction_gain
        )

    out = work.astype(np.float32)
    return out if audio.ndim > 1 else out[0]
