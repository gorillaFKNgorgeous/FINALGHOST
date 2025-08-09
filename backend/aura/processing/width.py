import numpy as np
from scipy import signal

from aura.schemas import StereoWidthSettingsModel


def apply_advanced_stereo_width(audio: np.ndarray, sr: int, settings: StereoWidthSettingsModel) -> np.ndarray:
    """Mid/Side width processing with optional side EQ shelf."""
    if audio.ndim == 1:
        return audio

    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2

    side_gain = settings.side_gain_db
    mid_gain = settings.mid_gain_db
    width = settings.width_factor

    if settings.side_shelf_hz:
        shelf_norm = settings.side_shelf_hz / (sr / 2)
        sos = signal.iirfilter(2, shelf_norm, btype="low", ftype="butter", output="sos")
        side = signal.sosfilt(sos, side)

    mid *= 10 ** (mid_gain / 20)
    side *= width * 10 ** (side_gain / 20)

    left = mid + side
    right = mid - side
    return np.vstack([left, right]).astype(np.float32)
