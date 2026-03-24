"""
ROI helpers for the timeseries waveform detection path.
"""

from __future__ import annotations

from . import config


def extract_fixed_roi(frame):
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    x1 = int(width * config.TS_FIXED_ROI_LEFT_RATIO)
    x2 = int(width * config.TS_FIXED_ROI_RIGHT_RATIO)
    y1 = int(height * config.TS_FIXED_ROI_TOP_RATIO)
    y2 = int(height * config.TS_FIXED_ROI_BOTTOM_RATIO)

    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


__all__ = ["extract_fixed_roi"]
