
# aura/processing/transient_shaper.py

import numpy as np
from aura.schemas import TransientShaperSettingsModel


def _envelope(x: np.ndarray, coeff: float) -> np.ndarray:
    """
    Generate a peak-hold style envelope follower with the given coefficient.
    Reacts quickly to peaks and decays according to the time constant.
    """
    env = np.zeros_like(x)
    abs_x = np.abs(x)
    if not np.any(abs_x):
        return env
    env[0] = abs_x[0]
    for i in range(1, len(x)):
        env[i] = max(abs_x[i], coeff * env[i - 1])
    return env


def apply_transient_shaper(audio: np.ndarray, sr: int, settings: TransientShaperSettingsModel) -> np.ndarray:
    """
    Apply transient shaping using a dual-envelope peak-hold strategy.
    Parameters are taken from TransientShaperSettingsModel.
    """
    work = np.atleast_2d(audio.astype(np.float64))

    atk_ms = max(settings.attack_ms, 1.0)  # Ensure attack is at least 1ms
    sus_ms = max(settings.sustain_ms, 1.0)  # Also respect user sustain setting

    atk_coeff = np.exp(-1.0 / (atk_ms / 1000.0 * sr))
    sus_coeff = np.exp(-1.0 / (sus_ms / 1000.0 * sr))
    atk_gain = 10 ** (settings.attack_gain_db / 20.0)
    sus_gain = 10 ** (settings.sustain_gain_db / 20.0)
    mix = settings.mix

    for ch in range(work.shape[0]):
        channel_data = work[ch]

        # Generate fast (attack) and slow (sustain) envelopes
        attack_env = _envelope(np.abs(channel_data), atk_coeff)
        sustain_env = _envelope(np.abs(channel_data), sus_coeff)

        # Calculate transient strength, normalized safely
        transient_strength = (attack_env - sustain_env) / (attack_env + 1e-9)
        transient_strength = np.clip(transient_strength, 0.0, 1.0)

        # Interpolate gain based on transient strength
        gain_multiplier = (
            1.0
            + (sus_gain - 1.0) * (1 - transient_strength)
            + (atk_gain - 1.0) * transient_strength
        )

        processed = channel_data * gain_multiplier
        work[ch] = (1 - mix) * channel_data + mix * processed

        # Optional: Clamp peak to prevent clipping
        peak = np.max(np.abs(work[ch]))
        if peak > 1.0:
            work[ch] /= peak * 1.02  # Normalize just below full scale

    out = work if audio.ndim > 1 else work[0]
    return out.astype(np.float32)
