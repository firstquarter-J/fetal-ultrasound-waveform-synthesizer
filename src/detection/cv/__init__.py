"""
OpenCV-based waveform detection modules.
"""

from .frame_detector import detect_waveform_region
from .motion import (
    analyze_segment_motion,
    build_combined_region,
    detect_motion_in_waveform,
    get_waveform_region_bounds,
)
from .postprocess import (
    analyze_video_cv,
    calculate_avg_wide_rows,
    find_continuous_segments,
    merge_adjacent_segments,
    normalize_waveform_type,
)

__all__ = [
    "analyze_segment_motion",
    "analyze_video_cv",
    "build_combined_region",
    "calculate_avg_wide_rows",
    "detect_motion_in_waveform",
    "detect_waveform_region",
    "find_continuous_segments",
    "get_waveform_region_bounds",
    "merge_adjacent_segments",
    "normalize_waveform_type",
]
