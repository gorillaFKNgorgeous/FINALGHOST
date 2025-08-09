
"""Aura – full mastering‐chain orchestrator (channel‐order safe, circular‐import free)

* Accepts (samples × channels) or (channels × samples); converts to
  channel‐first internally so every DSP module sees the expected layout.
* Restores original orientation before returning so external code/tests see
  the same shape they passed in.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from aura.analysis.metrics import measure_integrated_lufs
from aura.processing.eq import apply_parametric_eq
from aura.processing.compressor import apply_single_band_compressor, apply_multiband_compressor
from aura.processing.width import apply_advanced_stereo_width
from aura.processing.normalize import normalize_lufs
from aura.processing.clipper import apply_lookahead_clipper
from aura.processing.dither import apply_dithering
from aura.processing.saturation import apply_saturation
from aura.processing.deesser import apply_deesser
from aura.processing.transient_shaper import apply_transient_shaper

from aura.schemas import MasteringParams


def _log(logger: Optional[logging.Logger], msg: str):
    if logger:
        logger.info(msg)


def _log_levels(
    logger: Optional[logging.Logger],
    stage: str,
    audio: np.ndarray,
    sr: int,
    prev: float
) -> float:
    """Log LUFS/peak at each stage; return new LUFS."""
    if logger is None:
        return measure_integrated_lufs(audio, sr)

    lufs = measure_integrated_lufs(audio, sr)
    delta = lufs - prev
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(max(peak, 1e-12))
    logger.info(f"{stage} -> {lufs:.2f} LUFS ({delta:+.2f}), peak {peak_db:.2f} dBFS")
    return lufs


def run_full_chain(
    audio: np.ndarray,
    sr: int,
    params: MasteringParams,
    *,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Master audio according to *params*.
    The function is agnostic to the input channel orientation (Nx2 or 2xN) and
    returns audio in the same orientation.
    """
    if audio.ndim != 2:
        raise ValueError("Audio must be a 2‐D array (mono/stereo)")

    ch_first_in = audio.shape[0] <= 8
    work = audio if ch_first_in else audio.T

    prev_lufs = _log_levels(logger, "Input", work, sr, -120.0)
    _log(logger, "Start mastering chain")

    if params.eq_on and params.eq_settings is not None:
        _log(logger, "EQ")
        work = apply_parametric_eq(work, sr, params.eq_settings)
        prev_lufs = _log_levels(logger, "After EQ", work, sr, prev_lufs)

    # ── Imaging BEFORE dynamics ──
    if params.stereo_on and params.stereo_settings is not None:
        _log(logger, "Stereo Width")
        work = apply_advanced_stereo_width(work, sr, params.stereo_settings)
        prev_lufs = _log_levels(logger, "After Stereo Width", work, sr, prev_lufs)

    if params.compressor_on and params.compressor_settings is not None:
        _log(logger, "Compressor")
        work = apply_single_band_compressor(work, sr, params.compressor_settings)
        prev_lufs = _log_levels(logger, "After Compressor", work, sr, prev_lufs)

    if params.mb_comp_on and params.mb_comp_settings is not None:
        _log(logger, "Multiband Compressor")
        work = apply_multiband_compressor(work, sr, params.mb_comp_settings)
        prev_lufs = _log_levels(logger, "After Multiband", work, sr, prev_lufs)

    if params.transient_on and params.transient_settings is not None:
        _log(logger, "Transient Shaper")
        work = apply_transient_shaper(work, sr, params.transient_settings)
        prev_lufs = _log_levels(logger, "After Transients", work, sr, prev_lufs)

    if params.saturation_on and params.saturation_settings is not None:
        _log(logger, "Saturation")
        work = apply_saturation(work, sr, params.saturation_settings)
        prev_lufs = _log_levels(logger, "After Saturation", work, sr, prev_lufs)

    if params.deesser_on and params.deesser_settings is not None:
        _log(logger, "De-esser")
        work = apply_deesser(work, sr, params.deesser_settings)
        prev_lufs = _log_levels(logger, "After De-esser", work, sr, prev_lufs)

    # ---------------- Loudness match first ----------------
    if params.lufs_normalize_on:
        _log(logger, f"Normalizing to {params.target_lufs:.2f} LUFS")
        work = normalize_lufs(work, sr, params.target_lufs)
        prev_lufs = _log_levels(logger, "After Normalization", work, sr, prev_lufs)

    # ---------------- Peak-taming second ------------------
    if params.clipper_on and params.clipper_settings is not None:
        _log(logger, "Lookahead Clipper")
        work = apply_lookahead_clipper(work, sr, params.clipper_settings)
        prev_lufs = _log_levels(logger, "After Clipper", work, sr, prev_lufs)

    if params.dither_on and params.dither_settings is not None:
        _log(logger, "Dithering")
        work = apply_dithering(work, params.dither_settings)

    _log(logger, "Finished processing chain")

    return work if ch_first_in else work.T
