import numpy as np

from aura.schemas import SaturationSettingsModel


def _waveshape(x: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("tape", "arctan"):
        return (2 / np.pi) * np.arctan(x)
    if mode == "tube":
        return np.tanh(x * 1.5 + 0.2)
    if mode == "transformer":
        return 2 / (1 + np.exp(-x)) - 1
    if mode == "tanh":
        return np.tanh(x)
    raise ValueError(
        f"Unsupported saturation algorithm '{mode}'. "
        "Expected one of {'tape', 'tube', 'transformer', 'arctan', 'tanh'}."
    )

def apply_saturation(audio: np.ndarray, sr: int, settings: SaturationSettingsModel) -> np.ndarray:
    """Apply waveshaping saturation."""
    drive = 10 ** (settings.drive_db / 20)
    mix = settings.mix_percent / 100.0
    mode = settings.algorithm

    work = audio.astype(np.float64)
    shaped = _waveshape(work * drive, mode)
    out = (1 - mix) * work + mix * shaped
    out *= 10 ** (settings.output_gain_db / 20)
    return out.astype(np.float32)
