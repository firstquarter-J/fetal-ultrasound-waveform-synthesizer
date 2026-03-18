"""
Stable public API for waveform detection.
"""

from __future__ import annotations

from pathlib import Path

from src.contracts.waveform import (
    WaveformAnalysisMode,
    WaveformSegmentPayload,
    normalize_waveform_segments,
)

from .analyzer import (
    analyze_video as _analyze_video_impl,
    analyze_video_cv as _analyze_video_cv_impl,
    analyze_video_timeseries as _analyze_video_timeseries_impl,
)


def analyze_video(
    video_path: str | Path,
    verbose: bool = True,
    mode: WaveformAnalysisMode = "timeseries",
) -> list[WaveformSegmentPayload]:
    segments = _analyze_video_impl(str(video_path), verbose=verbose, mode=mode)
    return normalize_waveform_segments(segments)


def analyze_video_timeseries(
    video_path: str | Path,
    verbose: bool = True,
) -> list[WaveformSegmentPayload]:
    segments = _analyze_video_timeseries_impl(str(video_path), verbose=verbose)
    return normalize_waveform_segments(segments)


def analyze_video_cv(
    video_path: str | Path,
    verbose: bool = True,
) -> list[WaveformSegmentPayload]:
    segments = _analyze_video_cv_impl(str(video_path), verbose=verbose)
    return normalize_waveform_segments(segments)


__all__ = [
    "analyze_video",
    "analyze_video_cv",
    "analyze_video_timeseries",
]
