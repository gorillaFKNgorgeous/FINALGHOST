import numpy as np
import pyloudnorm as pyln
import logging
from aura.processing.clipper import apply_lookahead_clipper
from aura.schemas import LookaheadClipperSettingsModel

logger = logging.getLogger(__name__)

def normalize_lufs(
        audio: np.ndarray,
        sr: int,
        target_lufs: float,
        peak_ceil_db: float = -1.0,
        clipper_settings: LookaheadClipperSettingsModel = None,
    ) -> np.ndarray:
        """
        True-peak-safe LUFS normalization for mastering. Applies gain, then true peak clipper.
        """
        work = audio.astype(np.float64)
        if np.max(np.abs(work)) < 1e-7:
            logger.warning("Input audio is silent, skipping LUFS normalization.")
            return work.astype(np.float32)
        audio_for_meter = work.T if work.ndim > 1 else work
        meter = pyln.Meter(sr)
        try:
            loudness = meter.integrated_loudness(audio_for_meter)
        except Exception as e:
            logger.error(f"pyloudnorm failed: {e}")
            return work.astype(np.float32)

        if not np.isfinite(loudness):
            logger.warning(f"Invalid loudness value ({loudness}) measured. Skipping normalization.")
            return work.astype(np.float32)

        gain_db = target_lufs - loudness
        gain_linear = 10 ** (gain_db / 20)
        normalized_audio = work * gain_linear

        # --- True Peak Clipper as Final Safety ---
        if clipper_settings is None:
            # Reasonable defaults for mastering
            clipper_settings = LookaheadClipperSettingsModel(
                threshold_db=peak_ceil_db,
                lookahead_ms=1.5,
                mode='hard',
                oversample_factor=2,
            )
        clipped_audio = apply_lookahead_clipper(
            normalized_audio, sr, clipper_settings
        )

        # Optional: Log post-processing metrics for QA
        audio_post = clipped_audio.T if clipped_audio.ndim > 1 else clipped_audio
        final_loudness = meter.integrated_loudness(audio_post)
        final_peak = np.max(np.abs(clipped_audio))

        logger.info(
            f"Target LUFS: {target_lufs:.2f}, Out LUFS: {final_loudness:.2f}, "
            f"Peak (linear): {final_peak:.4f} ({20*np.log10(final_peak):.2f} dBFS)"
        )

        return clipped_audio.astype(np.float32)