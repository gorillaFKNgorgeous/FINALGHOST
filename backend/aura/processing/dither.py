import numpy as np

from aura.schemas import DitherSettingsModel


def apply_dithering(audio: np.ndarray, settings: DitherSettingsModel) -> np.ndarray:
    """Apply TPDF dithering to audio."""
    bit_depth = getattr(settings, "output_bit_depth", 32)  # default no-dither
    if bit_depth >= 32:
        return audio
    scale = np.sqrt(2) / (2 ** (bit_depth - 1))
    noise = (np.random.rand(*audio.shape) - 0.5
             + np.random.rand(*audio.shape) - 0.5) * scale
    high_pass = getattr(settings, "high_pass", False)
    if high_pass and audio.ndim == 2:
        noise = np.concatenate([noise[:, :1],
                                np.diff(noise, axis=1)], axis=1)
    return (audio + noise).astype(np.float32)
