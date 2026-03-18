"""
Shared contracts for detection and synthesis modules.
"""

from .waveform import (
    WaveformAnalysisMode,
    WaveformSegment,
    WaveformSegmentPayload,
    normalize_waveform_segment,
    normalize_waveform_segments,
)

__all__ = [
    "WaveformAnalysisMode",
    "WaveformSegment",
    "WaveformSegmentPayload",
    "normalize_waveform_segment",
    "normalize_waveform_segments",
]
