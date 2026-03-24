#!/usr/bin/env python3
"""
Waveform analysis compatibility layer.

The internal implementation is split across:
  - src.detection.timeseries for the default timeseries path
  - src.detection.cv for the OpenCV path
"""

from __future__ import annotations

import sys

from .cv import (
    analyze_segment_motion,
    analyze_video_cv,
    build_combined_region,
    calculate_avg_wide_rows,
    detect_motion_in_waveform,
    detect_waveform_region,
    find_continuous_segments,
    get_waveform_region_bounds,
    merge_adjacent_segments,
    normalize_waveform_type,
)
from .timeseries.segmenter import analyze_video_timeseries


def analyze_video(video_path, verbose=True, mode="timeseries"):
    """
    영상 분석 및 파형 구간 검출

    Args:
        video_path: 분석할 영상 파일 경로
        verbose: 로그 출력 여부
        mode: 'timeseries'(기본) 또는 'cv'
    """
    if mode in ("cv", "vision", "opencv"):
        return analyze_video_cv(video_path, verbose=verbose)
    return analyze_video_timeseries(video_path, verbose=verbose)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "assets/heartbeat-samples/28w-126bpm.mp4"

    analyze_video(video_path)
