"""
Feature extraction helpers for timeseries waveform detection.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .roi import extract_fixed_roi


def smooth_1d(values, window):
    if window <= 1 or len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="same")


def _mask_column_score(mask, min_col_count):
    if mask.size == 0:
        return 0.0, 0.0
    col_counts = mask.sum(axis=0)
    coverage = float(np.mean(col_counts >= min_col_count))
    ratio = float(mask.mean())
    return ratio, coverage


def extract_timeseries_features_fixed_roi(frame):
    """고정 ROI에서 파형 후보 점수(orange/white) 및 검은 배경 비율을 추출."""
    roi_full = extract_fixed_roi(frame)
    if roi_full is None:
        return 0.0, 0.0, 0.0, 0.0

    roi_h_full, roi_w_full = roi_full.shape[:2]
    if roi_h_full <= 0 or roi_w_full <= 0:
        return 0.0, 0.0, 0.0, 0.0

    # 하단 썸네일/버튼 UI(파형이 아닌 영역)가 ROI에 포함되는 경우가 있어
    # 내부 ROI로 다시 잘라 특징을 계산합니다.
    y_inner1 = int(roi_h_full * config.TS_FIXED_ROI_INNER_TOP_RATIO)
    y_inner2 = int(roi_h_full * config.TS_FIXED_ROI_INNER_BOTTOM_RATIO)
    y_inner1 = max(0, min(y_inner1, roi_h_full - 1))
    y_inner2 = max(y_inner1 + 1, min(y_inner2, roi_h_full))

    roi = roi_full[y_inner1:y_inner2, :]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    roi_gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    black_ratio_full = float(np.mean(roi_gray_full <= config.TS_BLACK_THRESHOLD))
    top_h = max(1, int(roi_gray_full.shape[0] * config.TS_TOP_BLACK_HEIGHT_RATIO))
    top_black_ratio = float(np.mean(roi_gray_full[:top_h, :] <= config.TS_BLACK_THRESHOLD))
    if black_ratio_full < config.TS_BLACK_RATIO_MIN_ANY:
        return black_ratio_full, 0.0, 0.0, top_black_ratio

    roi_h, roi_w = roi.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return 0.0, 0.0, 0.0, top_black_ratio

    min_col_count = max(config.TS_MIN_COL_COUNT_PX, int(roi_h * config.TS_MIN_COL_COUNT_RATIO))

    hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_orange = 0.0
    best_orange_black = 0.0

    for split_ratio in config.TS_FIXED_ROI_SPLITS_ORANGE:
        w = int(roi_w * float(split_ratio))
        if w <= 0:
            continue
        hsv = hsv_full[:, :w]
        gray = roi_gray_full[:, :w]

        orange_mask = cv2.inRange(
            hsv,
            np.array([0, config.TS_COLOR_ORANGE_S_MIN, config.TS_COLOR_ORANGE_V_MIN]),
            np.array([config.TS_COLOR_ORANGE_H_MAX, 255, 255]),
        )
        orange_mask = (orange_mask > 0).astype(np.uint8)
        orange_ratio, orange_coverage = _mask_column_score(orange_mask, min_col_count)
        orange_score = orange_ratio * orange_coverage

        if orange_score > best_orange:
            best_orange = orange_score
            best_orange_black = float(np.mean(gray <= config.TS_BLACK_THRESHOLD))

    best_white = 0.0
    best_white_black = 0.0
    best_white_row_max = 0.0

    for split_ratio in config.TS_FIXED_ROI_SPLITS_WHITE:
        w = int(roi_w * float(split_ratio))
        if w <= 0:
            continue
        gray = roi_gray_full[:, :w]

        white_mask = (gray >= config.TS_WHITE_THRESHOLD).astype(np.uint8)
        white_ratio, white_coverage = _mask_column_score(white_mask, min_col_count)
        if white_coverage < config.TS_WHITE_COVERAGE_MIN:
            continue
        white_score = white_ratio * white_coverage

        split_black_ratio = float(np.mean(gray <= config.TS_BLACK_THRESHOLD))
        row_ratios = white_mask.mean(axis=1)
        row_max = float(row_ratios.max()) if row_ratios.size else 0.0

        if row_max < config.TS_WHITE_ROW_MAX_MIN:
            continue
        if split_black_ratio < config.TS_WHITE_SPLIT_BLACK_MIN:
            continue

        if white_score > best_white:
            best_white = white_score
            best_white_black = split_black_ratio
            best_white_row_max = row_max

    orange_ok = (
        black_ratio_full >= config.TS_ORANGE_BLACK_RATIO_MIN
        and best_orange_black >= config.TS_BLACK_GATE_ORANGE_MIN
        and top_black_ratio >= config.TS_TOP_BLACK_GATE_MIN
    )
    orange_score_final = best_orange if orange_ok else 0.0

    white_ok = (
        best_white_row_max >= config.TS_WHITE_ROW_MAX_MIN
        and top_black_ratio >= config.TS_TOP_BLACK_GATE_MIN
        and best_white_black >= config.TS_WHITE_SPLIT_BLACK_MIN
    )
    white_score_final = best_white if white_ok else 0.0

    return black_ratio_full, orange_score_final, white_score_final, top_black_ratio


def detect_baseline_in_frame(gray):
    """하단 UI 영역에서 기준선(수평선) y좌표를 추정."""
    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return None, 0.0

    x1 = int(width * config.TS_ROI_X_MARGIN)
    x2 = int(width * (1.0 - config.TS_ROI_X_MARGIN))
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))

    def baseline_quality(baseline_y, bright_threshold):
        if baseline_y is None:
            return None
        baseline_y = int(baseline_y)
        if baseline_y <= 0 or baseline_y >= height:
            return None

        band = int(config.TS_BASELINE_BAND_PX)
        above = gray[max(0, baseline_y - band):baseline_y, x1:x2]
        below = gray[baseline_y:min(height, baseline_y + band), x1:x2]
        if above.size == 0 or below.size == 0:
            return None

        row = gray[baseline_y:baseline_y + 1, x1:x2]
        bright_ratio = float(np.mean(row >= bright_threshold))

        black_above = float(np.mean(above <= config.TS_BASELINE_BLACK_THRESHOLD))
        black_below = float(np.mean(below <= config.TS_BASELINE_BLACK_THRESHOLD))
        mean_diff = abs(float(above.mean()) - float(below.mean()))

        if min(black_above, black_below) < config.TS_BASELINE_BLACK_MIN_RATIO:
            return None
        if mean_diff > config.TS_BASELINE_MAX_MEAN_DIFF:
            return None

        score = bright_ratio * min(black_above, black_below)
        return {
            "y": baseline_y,
            "bright_ratio": bright_ratio,
            "score": score,
        }

    candidates = []

    # 1) row 밝기 피크 기반(엄격)
    y0 = int(height * config.TS_BASELINE_SEARCH_TOP_RATIO)
    y1 = int(height * config.TS_BASELINE_SEARCH_BOTTOM_RATIO)
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    roi = gray[y0:y1, x1:x2]
    if roi.size > 0:
        # 기준선은 보통 1~2px 두께의 수평선이라 세로 블러(3x3)는
        # 라인 밝기를 크게 깎아먹습니다. 가로 방향만 살짝 스무딩합니다.
        roi_blur = cv2.GaussianBlur(roi, (5, 1), 0)
        bright_mask = (roi_blur >= config.TS_BASELINE_BRIGHT_ROW_THRESHOLD).astype(np.uint8)
        row_ratios = bright_mask.mean(axis=1)
        row_ratios_s = smooth_1d(row_ratios.tolist(), config.TS_BASELINE_ROW_SMOOTH_WINDOW)
        idx_sorted = np.argsort(row_ratios_s)[::-1]
        for i in idx_sorted[:50]:
            if float(row_ratios_s[int(i)]) <= 0.01:
                break
            cand_y = y0 + int(i)
            q = baseline_quality(cand_y, config.TS_BASELINE_BRIGHT_ROW_THRESHOLD)
            if q is not None:
                candidates.append(q)
                break

    # 2) Hough 기반(완화) - row 방식이 실패할 때를 위한 백업
    if not candidates:
        y0_relaxed = int(height * 0.45)
        y1_relaxed = int(height * 0.95)
        y0_relaxed = max(0, min(y0_relaxed, height - 1))
        y1_relaxed = max(y0_relaxed + 1, min(y1_relaxed, height))
        roi_relaxed = gray[y0_relaxed:y1_relaxed, :]
        if roi_relaxed.size > 0:
            roi_blur = cv2.GaussianBlur(roi_relaxed, (5, 5), 0)
            roi_eq = cv2.equalizeHist(roi_blur)
            edges = cv2.Canny(roi_eq, 40, 140)
            min_len = int(width * 0.30)
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=70,
                minLineLength=min_len,
                maxLineGap=30,
            )
            if lines is not None:
                for x1_line, y1_line, x2_line, y2_line in lines[:, 0]:
                    if abs(y2_line - y1_line) > 2:
                        continue
                    length = abs(x2_line - x1_line)
                    if length < min_len:
                        continue
                    cand_y = y0_relaxed + int(y1_line)
                    if (
                        cand_y < int(height * config.TS_BASELINE_SEARCH_TOP_RATIO)
                        or cand_y > int(height * config.TS_BASELINE_SEARCH_BOTTOM_RATIO)
                    ):
                        continue
                    q = baseline_quality(cand_y, config.TS_BASELINE_BRIGHT_ROW_THRESHOLD_RELAXED)
                    if q is not None:
                        candidates.append(q)

    if not candidates:
        return None, 0.0

    best = max(candidates, key=lambda c: c["score"])
    baseline_y = int(best["y"])
    baseline_ratio = float(best["bright_ratio"])

    if baseline_y > int(height * 0.97):
        return None, 0.0

    return baseline_y, baseline_ratio


def extract_timeseries_features(frame):
    """프레임에서 파형 관련 특징을 추출(고정 ROI + 1D 시계열용)."""
    return extract_timeseries_features_fixed_roi(frame)


__all__ = [
    "detect_baseline_in_frame",
    "extract_timeseries_features",
    "extract_timeseries_features_fixed_roi",
    "smooth_1d",
]
