"""
Frame-level waveform detection for the OpenCV path.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_waveform_region(frame):
    """
    단일 프레임에서 파형 검출

    3가지 타입의 파형을 검출합니다:
      1. Orange large contour: HSV 색상 기반 큰 컨투어
      2. Gray Doppler: 회색 도플러 파형 (thresholding 기반)
      3. Orange fragmented: 단편화된 오렌지 파형 (픽셀 기반)

    Args:
        frame: OpenCV 이미지 프레임

    Returns:
        (has_waveform, regions, pixel_count)
        - has_waveform: 파형 발견 여부
        - regions: [('type', x, y, w, h), ...]
        - pixel_count: 픽셀 수
    """
    height, width = frame.shape[:2]

    # ========================================
    # Type 1: Orange Large Contour
    # ========================================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([40, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((3, 3), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0

        is_horizontal = aspect_ratio > 5
        is_large = area > 10000
        is_bottom_center = y > height * 0.5 and x > 50

        if is_horizontal and is_large and is_bottom_center:
            return True, [("orange", x, y, w, h)], area

    # ========================================
    # Type 2: Gray Doppler
    # ========================================
    default_y = int(height * 0.5)
    default_h = int(height * 0.5)
    bottom_region = frame[default_y:, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(thresh)

    # 기존 코드와 동일하게 계산은 남겨 둡니다.
    row_counts = np.count_nonzero(thresh, axis=1)
    wide_rows = np.sum(row_counts > width * 0.3)
    left_region = thresh[:, : int(width * 0.6)]
    left_pixels = np.count_nonzero(left_region)
    left_ratio = (left_pixels / white_pixel_count * 100) if white_pixel_count > 0 else 0
    _ = (wide_rows, left_ratio)

    if white_pixel_count > 500:
        return True, [("gray", 0, default_y, width, default_h)], white_pixel_count

    if white_pixel_count > 1000:
        roi_x2 = int(width * 0.6)
        gray_left = gray[:, :roi_x2]
        _, thresh_left = cv2.threshold(gray_left, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh_left = cv2.morphologyEx(thresh_left, cv2.MORPH_CLOSE, kernel)
        thresh_left = cv2.morphologyEx(thresh_left, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh_left,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            y_global = default_y + y
            return True, [("gray_left", x, y_global, w_box, h_box)], white_pixel_count

    # ========================================
    # Type 3: Orange Fragmented
    # ========================================
    bottom_offset = int(height * 0.5)
    bottom_region_orange = orange_mask[bottom_offset:, :]
    orange_pixel_count = cv2.countNonZero(bottom_region_orange)

    if 5000 < orange_pixel_count < 80000:
        row_counts_orange = np.count_nonzero(bottom_region_orange, axis=1)
        rows_with_pixels = np.sum(row_counts_orange > 10)

        left_region_orange = bottom_region_orange[:, : int(width * 0.3)]
        left_orange_pixels = cv2.countNonZero(left_region_orange)
        left_orange_ratio = (
            left_orange_pixels / orange_pixel_count * 100
            if orange_pixel_count > 0
            else 0
        )

        if rows_with_pixels > 20 and left_orange_pixels > 1000 and 10 < left_orange_ratio < 50:
            coords = cv2.findNonZero(bottom_region_orange)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                y += bottom_offset
            else:
                x, y, w, h = 0, bottom_offset, width, int(height * 0.5)
            return True, [("orange_fragmented", x, y, w, h)], orange_pixel_count

    return False, [], 0


__all__ = ["detect_waveform_region"]
