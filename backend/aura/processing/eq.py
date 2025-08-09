"""Aura – Parametric EQ module (fixed syntax; aligns with unit‑tests)

* Handles field names: ``global_gain_db``, ``overall_gain_db``, ``gain_db``,
  ``output_gain_db``.
* Respects per‑settings ``linear_phase`` flag when function arg is *None*.
* RBJ peak & shelf, Butterworth HP/LP with variable order (6 dB/oct per pole).
* Minimum‑phase via ``sosfilt``; Linear‑phase via ``sosfiltfilt``.
"""
from __future__ import annotations

from math import sin, cos, pi, sqrt
from typing import List

import numpy as np
import scipy.signal as signal

from aura.schemas import EQBandModel, ParametricEQSettingsModel

__all__ = ["apply_parametric_eq"]

# -----------------------------------------------------------------------------
# RBJ biquad helpers
# -----------------------------------------------------------------------------

def _rbj_peak_sos(freq_hz: float, sr: int, q: float, gain_db: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * pi * freq_hz / sr
    alpha = sin(w0) / (2 * q)
    cosw0 = cos(w0)

    b0 = 1 + alpha * a
    b1 = -2 * cosw0
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cosw0
    a2 = 1 - alpha / a

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.tf2sos(b, a)


def _rbj_shelf_sos(freq_hz: float, sr: int, gain_db: float, shelf_type: str) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * pi * freq_hz / sr
    sinw0 = sin(w0)
    cosw0 = cos(w0)
    sqrt_a = sqrt(a)
    alpha = sinw0 / 2 * sqrt((a + 1 / a))

    if shelf_type == "low":
        b0 = a * ((a + 1) - (a - 1) * cosw0 + 2 * sqrt_a * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cosw0)
        b2 = a * ((a + 1) - (a - 1) * cosw0 - 2 * sqrt_a * alpha)
        a0 = (a + 1) + (a - 1) * cosw0 + 2 * sqrt_a * alpha
        a1 = -2 * ((a - 1) + (a + 1) * cosw0)
        a2 = (a + 1) + (a - 1) * cosw0 - 2 * sqrt_a * alpha
    elif shelf_type == "high":
        b0 = a * ((a + 1) + (a - 1) * cosw0 + 2 * sqrt_a * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cosw0)
        b2 = a * ((a + 1) + (a - 1) * cosw0 - 2 * sqrt_a * alpha)
        a0 = (a + 1) - (a - 1) * cosw0 + 2 * sqrt_a * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cosw0)
        a2 = (a + 1) - (a - 1) * cosw0 - 2 * sqrt_a * alpha
    else:
        raise ValueError("shelf_type must be 'low' or 'high'")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.tf2sos(b, a)

# -----------------------------------------------------------------------------
# Filter designer per band
# -----------------------------------------------------------------------------

def _design_sos(band: EQBandModel, sr: int) -> np.ndarray:
    t = band.type.lower()
    f = band.freq_hz
    if t == "peak":
        return _rbj_peak_sos(f, sr, band.q, band.gain_db)
    if t == "highpass":
        order = max(2, int(round(((band.slope_db_oct or 12) / 6))))
        return signal.butter(order, f / (sr / 2), btype="highpass", output="sos")
    if t == "lowpass":
        order = max(2, int(round(((band.slope_db_oct or 12) / 6))))
        return signal.butter(order, f / (sr / 2), btype="lowpass", output="sos")
    if t == "lowshelf":
        return _rbj_shelf_sos(f, sr, band.gain_db, "low")
    if t == "highshelf":
        return _rbj_shelf_sos(f, sr, band.gain_db, "high")
    raise ValueError(f"Unsupported EQ band type: {band.type}")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_parametric_eq(
    audio: np.ndarray,
    sr: int,
    settings: ParametricEQSettingsModel,
    *,
    linear_phase: bool | None = None,
) -> np.ndarray:
    """Apply parametric EQ to ``audio``.

    If ``linear_phase`` is *None* (default) the function checks if *settings*
    itself has a ``linear_phase`` attribute (which the unit tests rely on).
    """

    # detect overall gain field used by schema/tests
    overall_gain_db = next(
        (
            getattr(settings, fld)
            for fld in (
                "global_gain_db",
                "overall_gain_db",
                "gain_db",
                "output_gain_db",
            )
            if hasattr(settings, fld)
        ),
        0.0,
    )

    if not settings.bands and overall_gain_db == 0.0:
        return audio.astype(np.float64)

    if linear_phase is None:
        linear_phase = bool(getattr(settings, "linear_phase", False))

    # Numpy <1.26 has no ``copy`` kw for asanyarray – use ``np.asarray`` instead
    work = np.asarray(audio, dtype=np.float64)
    if work.ndim == 1:
        work = work[np.newaxis, :]

    for ch in range(work.shape[0]):
        sig = work[ch]
        for band in settings.bands:
            sos = _design_sos(band, sr)
            sig = signal.sosfiltfilt(sos, sig) if linear_phase else signal.sosfilt(sos, sig)
        sig *= 10 ** (overall_gain_db / 20)
        work[ch] = sig

    return work if audio.ndim > 1 else work[0]
