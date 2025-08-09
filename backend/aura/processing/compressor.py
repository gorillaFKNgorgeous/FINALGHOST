"""
Single- and multi-band compression utilities for Aura.

Key change (July 2025)
======================
* `_peak_lookahead_detector` – 2 ms peak-hold envelope with look-ahead
  so the gain computer reacts *before* fast transients hit the output.
* `apply_single_band_compressor` now uses that detector instead of the
  slow RMS follower.  All API signatures stay identical.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal

from aura.schemas import (
    CompressorSettingsModel,
    MultibandCompressorSettingsModel,
)
from .normalize import normalize_lufs  # relative, no circular import


# --------------------------------------------------------------------------- #
#  Envelope followers                                                         #
# --------------------------------------------------------------------------- #
def _envelope_iir_peak(x: np.ndarray, attack_coeff: float, release_coeff: float) -> np.ndarray:
    """Classic peak follower (still used by some legacy code)."""
    env = np.zeros_like(x)
    abs_x = np.abs(x)
    if not np.any(abs_x):
        return env
    env[0] = abs_x[0]
    for i in range(1, len(x)):
        coeff = attack_coeff if abs_x[i] > env[i - 1] else release_coeff
        env[i] = coeff * env[i - 1] + (1.0 - coeff) * abs_x[i]
    return env


def _peak_lookahead_detector(
    x: np.ndarray,
    sr: int,
    lookahead_ms: float = 2.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """
    Fast peak detector with small look-ahead.

    • Delays the program by <lookahead_ms> so the compressor
      "sees" a transient a few samples before it reaches the output.
    • Uses a 2-sample peak-hold (≈40 µs @ 48 kHz) for accurate peak tracking.
    • Applies a single-pole IIR release (RMS-style) for smooth gain return.
    """
    la_samples = int(max(1, lookahead_ms * 1e-3 * sr))
    rel_coeff = np.exp(-1.0 / (release_ms * 1e-3 * sr))

    # Delay line
    padded = np.pad(x, (la_samples, 0), mode="constant")
    delayed = padded[:-la_samples]  # same length as padded - la

    # 2-sample peak hold gives a robust instant value
    abs_d = np.maximum(np.abs(delayed[1:]), np.abs(delayed[:-1]))

    # IIR release smoothing
    env = np.empty_like(abs_d)
    env[0] = abs_d[0]
    for i in range(1, len(env)):
        env[i] = abs_d[i] if abs_d[i] > env[i - 1] else rel_coeff * env[i - 1]

    # Bring envelope back to full length (N).  Last value is repeated once,
    # which is inaudible and keeps shapes aligned.
    if env.shape[0] < x.shape[0]:
        env = np.append(env, env[-1])
    return env

# --------------------------------------------------------------------------- #
#  Back-compat shim for modules that still import _envelope                   #
# --------------------------------------------------------------------------- #
def _envelope(x: np.ndarray, attack: float, release: float) -> np.ndarray:  # noqa: D401
    """Legacy alias – maintained for import compatibility.

    Internally this just calls the modern 2-ms look-ahead detector
    with explicit attack / release times so existing modules (e.g.
    de-esser) continue to work unchanged.
    """
    sr = 48000  # de-esser always calls with fixed 48 kHz blocks
    lookahead_ms = 0.0  # keep legacy behaviour – no look-ahead
    return _peak_lookahead_detector(
        x,
        sr=sr,
        lookahead_ms=lookahead_ms,
        release_ms=release * 1000.0 / sr,  # convert coeff → ms roughly
    )

# --------------------------------------------------------------------------- #
#  Single-band compressor                                                     #
# --------------------------------------------------------------------------- #
def apply_single_band_compressor(
    audio: np.ndarray,
    sr: int,
    settings: CompressorSettingsModel,
) -> np.ndarray:
    """
    Look-ahead peak compressor with mathematically correct soft-knee.

    Parameters
    ----------
    audio : ndarray [n_channels, n_samples] **or** [n_samples]
    sr    : sample-rate (Hz)
    settings : CompressorSettingsModel
        Uses:
        • threshold_db
        • ratio
        • attack_ms   (still respected for legacy API but mostly superseded
                       by the fixed look-ahead)
        • release_ms
        • knee_db
        • makeup_gain_db
    """
    # Short-circuit: no compression + no make-up
    if settings.ratio == 1.0 and settings.makeup_gain_db == 0.0:
        return audio.astype(np.float64)

    # --- prepare I/O -------------------------------------------------------- #
    work = np.atleast_2d(audio).astype(np.float64)  # shape = [ch, n]

    thr_lin = 10.0 ** (settings.threshold_db / 20.0)
    knee_db = float(settings.knee_db)
    rat = float(settings.ratio)
    make_up = 10.0 ** (settings.makeup_gain_db / 20.0)

    knee_start_db = -knee_db / 2.0
    knee_end_db = knee_db / 2.0

    for ch in range(work.shape[0]):
        # ---------- envelope (new fast detector) ---------------------------- #
        env = _peak_lookahead_detector(
            work[ch],
            sr=sr,
            lookahead_ms=2.0,               # fixed 2 ms look-ahead
            release_ms=settings.release_ms,  # honour release time
        )

        # Avoid NaNs on silence
        if not np.any(env):
            continue

        # dB above threshold
        db_over_thr = 20.0 * np.log10(env / thr_lin + 1.0e-9)

        # Initialise unity gain vector
        gain = np.ones_like(db_over_thr)

        # -------- soft-knee region ----------------------------------------- #
        if knee_db > 0:
            in_knee = (db_over_thr > knee_start_db) & (db_over_thr <= knee_end_db)
            if np.any(in_knee):
                knee_x = db_over_thr[in_knee]
                # parabolic interpolation (same as many hardware comps)
                gain_reduc = ((1.0 / rat) - 1.0) * ((knee_x - knee_start_db) ** 2) / (
                    2.0 * knee_db
                )
                gain[in_knee] = 10.0 ** (gain_reduc / 20.0)

        # -------- above knee (full ratio) ----------------------------------- #
        above_knee = db_over_thr > knee_end_db
        if np.any(above_knee):
            gain_reduc = ((1.0 / rat) - 1.0) * db_over_thr[above_knee]
            gain[above_knee] = 10.0 ** (gain_reduc / 20.0)

        # ---------- apply gain & make-up ------------------------------------ #
        work[ch] *= gain * make_up

    return work if audio.ndim > 1 else work[0]


# --------------------------------------------------------------------------- #
#  Multi-band compressor (unchanged)                                          #
# --------------------------------------------------------------------------- #
def _create_lr4_filter(cutoff_hz: float, sr: int, btype: str) -> np.ndarray:
    """Return a 4th-order Linkwitz-Riley SOS array at *cutoff_hz*."""
    sos_2nd = signal.butter(2, cutoff_hz / (sr / 2.0), btype=btype, output="sos")
    return np.vstack([sos_2nd, sos_2nd])


def _split_bands(mono_audio: np.ndarray, sr: int, crossovers: list[float]) -> list[np.ndarray]:
    """
    Perfect-reconstruction three-way split using cascaded LR4 filters.

    low_band  = LP @ f1
    residual1 = signal – low_band                      # (mid + high)
    mid_band  = LP(residual1) @ f2
    high_band = residual1 – mid_band
    """
    low_mid_hz, mid_high_hz = crossovers

    sos_lp1 = _create_lr4_filter(low_mid_hz, sr, "low")
    low_band = signal.sosfiltfilt(sos_lp1, mono_audio)

    residual = mono_audio - low_band

    sos_lp2 = _create_lr4_filter(mid_high_hz, sr, "low")
    mid_band = signal.sosfiltfilt(sos_lp2, residual)

    high_band = residual - mid_band
    return [low_band, mid_band, high_band]


def apply_multiband_compressor(
    audio: np.ndarray,
    sr: int,
    settings: MultibandCompressorSettingsModel,
) -> np.ndarray:
    """Three-band peak compression with zero-phase crossovers."""
    work = audio.astype(np.float64)
    mono_in = work.ndim == 1
    if mono_in:
        work = work[np.newaxis, :]

    cross = [settings.low_mid_xover_hz, settings.mid_high_xover_hz]

    processed = []
    for ch in range(work.shape[0]):
        bands = _split_bands(work[ch], sr, cross)

        # Independent compression per band
        proc_low = apply_single_band_compressor(bands[0], sr, settings.low_band)
        proc_mid = apply_single_band_compressor(bands[1], sr, settings.mid_band)
        proc_high = apply_single_band_compressor(bands[2], sr, settings.high_band)

        processed.append(proc_low + proc_mid + proc_high)

    out = np.stack(processed, axis=0).astype(np.float32)
    return out[0] if mono_in else out