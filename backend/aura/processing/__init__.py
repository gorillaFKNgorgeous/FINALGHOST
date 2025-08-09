"""Aura processing public interface – no circular imports (THIS IS __init__.py)

Import order is critical: load *normalize* first (no dependencies), then
*compressor* (which needs normalize), then the rest. This avoids the circular
initialisation that stopped tests from collecting.
"""

from __future__ import annotations

# 1.  Modules with **no** internal Aura‑processing deps -----------------------
from .normalize import normalize_lufs  # noqa: F401

# 2.  Modules that depend on *normalize* --------------------------------------
from .compressor import apply_single_band_compressor, apply_multiband_compressor  # noqa: F401

# 3.  Everything else ---------------------------------------------------------
from .eq import apply_parametric_eq  # noqa: F401
from .dynamic_eq import apply_dynamic_eq  # noqa: F401
from .saturation import apply_saturation  # noqa: F401
from .width import apply_advanced_stereo_width  # noqa: F401
from .deesser import apply_deesser  # noqa: F401
from .transient_shaper import apply_transient_shaper  # noqa: F401
from .clipper import apply_lookahead_clipper  # noqa: F401
from .dither import apply_dithering  # noqa: F401
from .chain import run_full_chain  # noqa: F401

__all__ = [
    "apply_parametric_eq",
    "apply_dynamic_eq",
    "apply_single_band_compressor",
    "apply_multiband_compressor",
    "apply_saturation",
    "apply_advanced_stereo_width",
    "apply_deesser",
    "apply_transient_shaper",
    "apply_lookahead_clipper",
    "apply_dithering",
    "normalize_lufs",
    "run_full_chain",
]
