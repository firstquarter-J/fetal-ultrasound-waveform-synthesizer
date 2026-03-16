#!/usr/bin/env python3
"""
초음파 영상 심박 파형 자동 검출

OpenCV 기반 컴퓨터 비전으로 태아 초음파 영상에서
심박 파형 구간을 자동으로 검출합니다.

지원 파형 타입:
  - orange: 오렌지색 큰 컨투어 파형
  - gray: 회색/흰색 도플러 파형
  - orange_fragmented: 오렌지색 단편화 파형
"""

import cv2
import numpy as np
import sys


# ============================================================================
# 헬퍼 함수
# ============================================================================
MIN_SEG_DURATION = 3.0
SHORT_CONFIDENT_MIN_DURATION = 1.0
SHORT_STRONG_PIXEL_COUNT = 50000
SHORT_MIN_WIDE_ROWS = 30
EDGE_PAD_BEFORE = 1.2
EDGE_PAD_AFTER = 1.0
MOTION_GAP_THRESHOLD = 1.5
MOTION_MIN_DURATION = 1.0
MOTION_EDGE_PADDING = 0.2
MOTION_REFINEMENT_MIN_SOURCE = 12.0
MOTION_KEEP_MIN_DURATION = 0.2
MOTION_MERGE_GAP = 0.3
MOTION_PAD_BEFORE = 0.4
MOTION_PAD_AFTER = 0.4
MIN_REFINED_SEGMENT = 1.0
MIN_FINAL_SEG_DURATION = 1.0
MAX_SEG_DURATION_BEFORE_MOTION = 4.0
MAX_FINAL_SEG_DURATION = 12.0
MOTION_KEEP_MIN_DURATION = 0.1

# =========================================================================
# 시계열(1D) 기반 파형 구간 검출 파라미터
# =========================================================================
TS_STRIDE_FRAMES = 3
TS_FIXED_ROI_LEFT_RATIO = 0.08
TS_FIXED_ROI_RIGHT_RATIO = 0.88
TS_FIXED_ROI_TOP_RATIO = 0.55
TS_FIXED_ROI_BOTTOM_RATIO = 0.92
TS_FIXED_ROI_INNER_TOP_RATIO = 0.02
TS_FIXED_ROI_INNER_BOTTOM_RATIO = 0.82
TS_FIXED_ROI_SPLITS_ORANGE = (1.0, 0.6, 0.4)
TS_FIXED_ROI_SPLITS_WHITE = (1.0, 0.6)
TS_BLACK_THRESHOLD = 35
TS_WHITE_THRESHOLD = 200
TS_COLOR_ORANGE_H_MAX = 40
TS_COLOR_ORANGE_S_MIN = 70
TS_COLOR_ORANGE_V_MIN = 80
TS_MIN_COL_COUNT_PX = 3
TS_MIN_COL_COUNT_RATIO = 0.02
TS_BLACK_GATE_ORANGE_MIN = 0.70
TS_BLACK_GATE_WHITE_MIN = 0.55
TS_WHITE_ROW_MAX_MIN = 0.35
TS_WHITE_ROW_MAX_STRONG = 0.70
TS_ORANGE_BLACK_RATIO_MIN = 0.70
TS_WHITE_BLACK_RATIO_MIN = 0.50
TS_TOP_BLACK_HEIGHT_RATIO = 0.25
TS_TOP_BLACK_GATE_MIN = 0.80
TS_BLACK_RATIO_MIN_ANY = 0.20
TS_WHITE_SPLIT_BLACK_MIN = 0.25
TS_WHITE_COVERAGE_MIN = 0.10
TS_BASELINE_MIN_RATIO = 0.08
TS_BASELINE_BRIGHT_ROW_THRESHOLD = 150
TS_BASELINE_BRIGHT_ROW_THRESHOLD_RELAXED = 130
TS_BASELINE_BLACK_THRESHOLD = 50
TS_BASELINE_BLACK_MIN_RATIO = 0.08
TS_BASELINE_MAX_MEAN_DIFF = 70
TS_BASELINE_BAND_PX = 18
TS_BASELINE_ROW_SMOOTH_WINDOW = 9
TS_BASELINE_SEARCH_TOP_RATIO = 0.52
TS_BASELINE_SEARCH_BOTTOM_RATIO = 0.92
TS_SMOOTH_WINDOW = 9
TS_ORANGE_S_MIN = 150
TS_ORANGE_V_MIN = 150
TS_ORANGE_MIN_ON = 0.035
TS_ORANGE_MIN_OFF = 0.022
TS_WHITE_MIN_ON = 0.050
TS_WHITE_MIN_OFF = 0.030
TS_ORANGE_MIN_SEGMENT_SEC = 0.8
TS_WHITE_MIN_SEGMENT_SEC = 0.6
TS_MERGE_GAP_SEC = 0.5
TS_MAX_SEGMENT_SEC = 30.0
TS_SHORT_SEGMENT_SEC = 2.0
TS_SHORT_BLACK_RATIO_MIN = 0.42
TS_SHORT_ORANGE_PEAK_MIN = 0.0025
TS_SHORT_ORANGE_MEAN_MIN = 0.08
TS_SHORT_WHITE_PEAK_MIN = 0.006
TS_ORANGE_SUPPRESS_WHITE_THRESHOLD = 0.010
TS_HYSTERESIS_HIGH_Q = 0.995
TS_HYSTERESIS_ON_SCALE = 0.55
TS_HYSTERESIS_OFF_SCALE = 0.40


def _smooth_1d(values, window):
    if window <= 1 or len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode='same')


def _find_binary_segments(times, mask, min_duration_sec, merge_gap_sec):
    segments = []
    active = False
    start = None
    for ts, val in zip(times, mask):
        ts = float(ts)
        if not active and val:
            active = True
            start = ts
        elif active and not val:
            end = ts
            if start is not None and end - start >= min_duration_sec:
                segments.append((start, end))
            active = False
            start = None

    if active and start is not None:
        end = float(times[-1])
        if end - start >= min_duration_sec:
            segments.append((start, end))

    if not segments:
        return []

    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]

def _extract_fixed_roi(frame):
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    x1 = int(width * TS_FIXED_ROI_LEFT_RATIO)
    x2 = int(width * TS_FIXED_ROI_RIGHT_RATIO)
    y1 = int(height * TS_FIXED_ROI_TOP_RATIO)
    y2 = int(height * TS_FIXED_ROI_BOTTOM_RATIO)

    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def _mask_column_score(mask, min_col_count):
    if mask.size == 0:
        return 0.0, 0.0
    col_counts = mask.sum(axis=0)
    coverage = float(np.mean(col_counts >= min_col_count))
    ratio = float(mask.mean())
    return ratio, coverage


def _extract_timeseries_features_fixed_roi(frame):
    """고정 ROI에서 파형 후보 점수(orange/white) 및 검은 배경 비율을 추출."""
    roi_full = _extract_fixed_roi(frame)
    if roi_full is None:
        return 0.0, 0.0, 0.0, 0.0

    roi_h_full, roi_w_full = roi_full.shape[:2]
    if roi_h_full <= 0 or roi_w_full <= 0:
        return 0.0, 0.0, 0.0, 0.0

    # 하단 썸네일/버튼 UI(파형이 아닌 영역)가 ROI에 포함되는 경우가 있어
    # 내부 ROI로 다시 잘라 특징을 계산합니다.
    y_inner1 = int(roi_h_full * TS_FIXED_ROI_INNER_TOP_RATIO)
    y_inner2 = int(roi_h_full * TS_FIXED_ROI_INNER_BOTTOM_RATIO)
    y_inner1 = max(0, min(y_inner1, roi_h_full - 1))
    y_inner2 = max(y_inner1 + 1, min(y_inner2, roi_h_full))

    roi = roi_full[y_inner1:y_inner2, :]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    roi_gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    black_ratio_full = float(np.mean(roi_gray_full <= TS_BLACK_THRESHOLD))
    top_h = max(1, int(roi_gray_full.shape[0] * TS_TOP_BLACK_HEIGHT_RATIO))
    top_black_ratio = float(np.mean(roi_gray_full[:top_h, :] <= TS_BLACK_THRESHOLD))
    if black_ratio_full < TS_BLACK_RATIO_MIN_ANY:
        return black_ratio_full, 0.0, 0.0, top_black_ratio

    roi_h, roi_w = roi.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return 0.0, 0.0, 0.0, top_black_ratio

    min_col_count = max(TS_MIN_COL_COUNT_PX, int(roi_h * TS_MIN_COL_COUNT_RATIO))

    hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_orange = 0.0
    best_orange_black = 0.0

    for split_ratio in TS_FIXED_ROI_SPLITS_ORANGE:
        w = int(roi_w * float(split_ratio))
        if w <= 0:
            continue
        hsv = hsv_full[:, :w]
        gray = roi_gray_full[:, :w]

        orange_mask = cv2.inRange(
            hsv,
            np.array([0, TS_COLOR_ORANGE_S_MIN, TS_COLOR_ORANGE_V_MIN]),
            np.array([TS_COLOR_ORANGE_H_MAX, 255, 255]),
        )
        orange_mask = (orange_mask > 0).astype(np.uint8)
        orange_ratio, orange_coverage = _mask_column_score(orange_mask, min_col_count)
        orange_score = orange_ratio * orange_coverage

        if orange_score > best_orange:
            best_orange = orange_score
            best_orange_black = float(np.mean(gray <= TS_BLACK_THRESHOLD))

    best_white = 0.0
    best_white_black = 0.0
    best_white_row_max = 0.0

    for split_ratio in TS_FIXED_ROI_SPLITS_WHITE:
        w = int(roi_w * float(split_ratio))
        if w <= 0:
            continue
        gray = roi_gray_full[:, :w]

        white_mask = (gray >= TS_WHITE_THRESHOLD).astype(np.uint8)
        white_ratio, white_coverage = _mask_column_score(white_mask, min_col_count)
        if white_coverage < TS_WHITE_COVERAGE_MIN:
            continue
        white_score = white_ratio * white_coverage

        split_black_ratio = float(np.mean(gray <= TS_BLACK_THRESHOLD))
        row_ratios = white_mask.mean(axis=1)
        row_max = float(row_ratios.max()) if row_ratios.size else 0.0

        if row_max < TS_WHITE_ROW_MAX_MIN:
            continue
        if split_black_ratio < TS_WHITE_SPLIT_BLACK_MIN:
            continue

        if white_score > best_white:
            best_white = white_score
            best_white_black = split_black_ratio
            best_white_row_max = row_max

    orange_ok = (
        black_ratio_full >= TS_ORANGE_BLACK_RATIO_MIN
        and best_orange_black >= TS_BLACK_GATE_ORANGE_MIN
        and top_black_ratio >= TS_TOP_BLACK_GATE_MIN
    )
    orange_score_final = best_orange if orange_ok else 0.0

    white_ok = (
        best_white_row_max >= TS_WHITE_ROW_MAX_MIN
        and top_black_ratio >= TS_TOP_BLACK_GATE_MIN
        and best_white_black >= TS_WHITE_SPLIT_BLACK_MIN
    )
    white_score_final = best_white if white_ok else 0.0

    return black_ratio_full, orange_score_final, white_score_final, top_black_ratio


def _adaptive_hysteresis_threshold(series, min_on, min_off):
    series = np.asarray(series, dtype=np.float64)
    if series.size == 0:
        return float(min_on), float(min_off)

    high = float(np.quantile(series, TS_HYSTERESIS_HIGH_Q))
    on = max(float(min_on), high * float(TS_HYSTERESIS_ON_SCALE))
    off = max(float(min_off), high * float(TS_HYSTERESIS_OFF_SCALE))
    if off >= on:
        off = on * 0.8
    return float(on), float(off)


def _refine_segment_bounds(times_arr, raw_series, start, end, threshold):
    idx = np.where((times_arr >= start) & (times_arr <= end) & (raw_series >= threshold))[0]
    if idx.size == 0:
        return float(start), float(end)
    return float(times_arr[int(idx[0])]), float(times_arr[int(idx[-1])])


def _detect_baseline_in_frame(gray):
    """하단 UI 영역에서 기준선(수평선) y좌표를 추정.

    기존 Hough 기반은 대비/노이즈 조건에 민감해 B-mode에서도
    "가짜 수평선"을 기준선으로 오검출하는 경우가 있어,
    (1) row 밝기 피크 기반(엄격) → (2) Hough 기반(완화) 순으로
    후보를 만들고, 공통 품질 조건(검은 배경/상하 평균차)을 통과한
    최적 후보만 채택합니다.

    Returns:
        (baseline_y, baseline_ratio)
        baseline_ratio는 "기준선 row에서 밝은 픽셀 비율"(0~1).
    """

    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return None, 0.0

    x1 = int(width * TS_ROI_X_MARGIN)
    x2 = int(width * (1.0 - TS_ROI_X_MARGIN))
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))

    def baseline_quality(baseline_y, bright_threshold):
        if baseline_y is None:
            return None
        baseline_y = int(baseline_y)
        if baseline_y <= 0 or baseline_y >= height:
            return None

        band = int(TS_BASELINE_BAND_PX)
        above = gray[max(0, baseline_y - band):baseline_y, x1:x2]
        below = gray[baseline_y:min(height, baseline_y + band), x1:x2]
        if above.size == 0 or below.size == 0:
            return None

        row = gray[baseline_y:baseline_y + 1, x1:x2]
        bright_ratio = float(np.mean(row >= bright_threshold))

        black_above = float(np.mean(above <= TS_BASELINE_BLACK_THRESHOLD))
        black_below = float(np.mean(below <= TS_BASELINE_BLACK_THRESHOLD))
        mean_diff = abs(float(above.mean()) - float(below.mean()))

        if min(black_above, black_below) < TS_BASELINE_BLACK_MIN_RATIO:
            return None
        if mean_diff > TS_BASELINE_MAX_MEAN_DIFF:
            return None

        score = bright_ratio * min(black_above, black_below)
        return {
            "y": baseline_y,
            "bright_ratio": bright_ratio,
            "score": score,
        }

    candidates = []

    # 1) row 밝기 피크 기반(엄격)
    y0 = int(height * TS_BASELINE_SEARCH_TOP_RATIO)
    y1 = int(height * TS_BASELINE_SEARCH_BOTTOM_RATIO)
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    roi = gray[y0:y1, x1:x2]
    if roi.size > 0:
        # 기준선은 보통 1~2px 두께의 수평선이라 세로 블러(3x3)는
        # 라인 밝기를 크게 깎아먹습니다. 가로 방향만 살짝 스무딩합니다.
        roi_blur = cv2.GaussianBlur(roi, (5, 1), 0)
        bright_mask = (roi_blur >= TS_BASELINE_BRIGHT_ROW_THRESHOLD).astype(np.uint8)
        row_ratios = bright_mask.mean(axis=1)
        row_ratios_s = _smooth_1d(row_ratios.tolist(), TS_BASELINE_ROW_SMOOTH_WINDOW)
        # 상위 후보들을 순회하며 품질 조건을 통과하는 첫 후보를 채택합니다.
        idx_sorted = np.argsort(row_ratios_s)[::-1]
        for i in idx_sorted[:50]:
            if float(row_ratios_s[int(i)]) <= 0.01:
                break
            cand_y = y0 + int(i)
            q = baseline_quality(cand_y, TS_BASELINE_BRIGHT_ROW_THRESHOLD)
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
                    if cand_y < int(height * TS_BASELINE_SEARCH_TOP_RATIO) or cand_y > int(height * TS_BASELINE_SEARCH_BOTTOM_RATIO):
                        continue
                    q = baseline_quality(cand_y, TS_BASELINE_BRIGHT_ROW_THRESHOLD_RELAXED)
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


def _extract_timeseries_features(frame):
    """프레임에서 파형 관련 특징을 추출(고정 ROI + 1D 시계열용)."""
    return _extract_timeseries_features_fixed_roi(frame)


def analyze_video_timeseries(video_path, verbose=True):
    """ROI 고정/시계열 기반으로 파형 구간을 검출.

    프레임별로 파형 박스를 찾지 않고,
    (1) 기준선 존재 여부를 확인한 뒤
    (2) 기준선 주변 ROI에서 orange/white 비율 시계열을 만들고
    (3) 임계값 히스테리시스로 구간을 산출합니다.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상 열기 실패: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or np.isnan(fps) or fps <= 1e-6:
        if verbose:
            print("경고: FPS 미확인 → 30.0으로 대체")
        fps = 30.0

    def log(message):
        if verbose:
            print(message)

    video_duration = (total_frames / fps) if fps else 0.0
    log(f"\n[TS] 영상: {video_path}")
    log(f"  FPS: {fps}, 총 프레임: {total_frames}, 길이: {video_duration:.1f}초")

    frame_idx = 0
    times = []
    black_ratios = []
    orange_series = []
    white_series = []
    top_black_series = []

    while True:
        if frame_idx % TS_STRIDE_FRAMES != 0:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        black_ratio, orange_score, white_score, top_black_ratio = _extract_timeseries_features(frame)
        times.append(frame_idx / fps)
        black_ratios.append(black_ratio)
        orange_series.append(orange_score)
        white_series.append(white_score)
        top_black_series.append(top_black_ratio)

        frame_idx += 1
        if total_frames and frame_idx % 300 == 0:
            pct = frame_idx / total_frames * 100
            log(f"  진행: {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()

    if not times:
        return []

    orange_raw = np.array(orange_series, dtype=np.float64)
    white_raw = np.array(white_series, dtype=np.float64)
    orange_s = _smooth_1d(orange_raw, TS_SMOOTH_WINDOW)
    white_s = _smooth_1d(white_raw, TS_SMOOTH_WINDOW)
    times_arr = np.array(times, dtype=np.float64)
    black_arr = np.array(black_ratios, dtype=np.float64)
    top_black_arr = np.array(top_black_series, dtype=np.float64)
    orange_arr = np.array(orange_s, dtype=np.float64)
    white_arr = np.array(white_s, dtype=np.float64)

    def hysteresis_mask(series, on_threshold, off_threshold):
        active = False
        out = []
        for v in series:
            v = float(v)
            if not active:
                if v >= on_threshold:
                    active = True
            else:
                if v <= off_threshold:
                    active = False
            out.append(active)
        return out

    orange_on, orange_off = _adaptive_hysteresis_threshold(orange_arr, TS_ORANGE_MIN_ON, TS_ORANGE_MIN_OFF)
    white_on, white_off = _adaptive_hysteresis_threshold(white_arr, TS_WHITE_MIN_ON, TS_WHITE_MIN_OFF)

    # 구간 밖 오탐을 막기 위해, 동영상별 적응형 임계값 대신
    # 고정 임계값(최소치)을 사용합니다.
    orange_on, orange_off = TS_ORANGE_MIN_ON, TS_ORANGE_MIN_OFF
    white_on, white_off = TS_WHITE_MIN_ON, TS_WHITE_MIN_OFF

    orange_mask = hysteresis_mask(orange_arr, orange_on, orange_off)

    # 오렌지 파형이 충분히 존재하는 프레임에서는 white 채널을 비활성화하여
    # (오렌지 파형이 white 임계값까지 끌어올리는 경우) 중복/오탐을 줄입니다.
    white_for_detect = np.where(orange_arr >= TS_ORANGE_SUPPRESS_WHITE_THRESHOLD, 0.0, white_arr)
    white_mask = hysteresis_mask(white_for_detect, white_on, white_off)

    orange_segments = _find_binary_segments(times, orange_mask, TS_ORANGE_MIN_SEGMENT_SEC, TS_MERGE_GAP_SEC)
    white_segments = _find_binary_segments(times, white_mask, TS_WHITE_MIN_SEGMENT_SEC, TS_MERGE_GAP_SEC)

    # 채널별 구간을 합친 뒤, 겹치거나 가까운 구간을 병합
    segments = sorted(orange_segments + white_segments)
    merged_segments = []
    for s, e in segments:
        if not merged_segments:
            merged_segments.append([s, e])
            continue
        if s - merged_segments[-1][1] <= TS_MERGE_GAP_SEC:
            merged_segments[-1][1] = max(merged_segments[-1][1], e)
        else:
            merged_segments.append([s, e])
    segments = [(float(s), float(e)) for s, e in merged_segments]

    segments_to_report = []
    for start, end in segments:
        start = max(0.0, start)
        end = min(video_duration, end)
        duration = end - start
        if duration < TS_WHITE_MIN_SEGMENT_SEC or duration > TS_MAX_SEGMENT_SEC:
            continue

        idx = (times_arr >= start) & (times_arr <= end)
        if not np.any(idx):
            continue

        mean_black_ratio = float(black_arr[idx].mean())
        peak_top_black = float(top_black_arr[idx].max())
        peak_orange = float(orange_arr[idx].max())
        peak_white = float(white_arr[idx].max())
        mean_orange = float(orange_arr[idx].mean())
        mean_white = float(white_arr[idx].mean())

        if peak_top_black < TS_TOP_BLACK_GATE_MIN:
            continue

        if duration <= TS_SHORT_SEGMENT_SEC:
            if mean_black_ratio < TS_SHORT_BLACK_RATIO_MIN:
                continue
            if peak_orange < TS_SHORT_ORANGE_PEAK_MIN and peak_white < TS_SHORT_WHITE_PEAK_MIN:
                continue
            if peak_orange >= peak_white and mean_orange < TS_SHORT_ORANGE_MEAN_MIN:
                continue

        # 스무딩으로 인해 구간이 과도하게 늘어나는 것을 방지하기 위해
        # 원본 시계열에서 off threshold 기준으로 경계를 재조정합니다.
        refined_start, refined_end = start, end
        refined_start, refined_end = _refine_segment_bounds(
            times_arr,
            np.maximum(orange_raw, white_raw),
            start,
            end,
            min(orange_off, white_off),
        )
        start, end = refined_start, refined_end
        if end - start < TS_WHITE_MIN_SEGMENT_SEC:
            continue

        segments_to_report.append({
            'start_time': start,
            'end_time': end,
            'type': 'timeseries',
        })

    if fps:
        for seg in segments_to_report:
            seg['start_frame'] = max(0, int(seg['start_time'] * fps))
            seg['end_frame'] = max(seg['start_frame'], int(seg['end_time'] * fps))

    log(f"\n  [TS] 파형 감지 구간: {len(segments_to_report)}개")
    for i, seg in enumerate(segments_to_report, 1):
        dur = seg['end_time'] - seg['start_time']
        log(f"    구간 {i}: {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 ({dur:.2f}초)")

    return segments_to_report

def normalize_waveform_type(waveform_type):
    """
    파형 타입 정규화

    orange와 orange_fragmented를 같은 타입으로 취급하여
    단일 프레임 타입 변경으로 인한 구간 분리를 방지합니다.

    Args:
        waveform_type: 'orange', 'orange_fragmented', 'gray' 등

    Returns:
        정규화된 타입 ('orange' 또는 원본)
    """
    if waveform_type in ['orange', 'orange_fragmented']:
        return 'orange'
    if waveform_type in ['gray_left']:
        return 'gray'
    return waveform_type


def calculate_avg_wide_rows(video_path, segment, fps):
    """
    구간의 평균 wide_rows 계산

    Wide rows: 화면 너비의 50% 이상을 차지하는 행의 개수
    실제 파형과 false positive를 구분하는 핵심 지표입니다.

    Args:
        video_path: 영상 파일 경로
        segment: 분석할 구간 {'start_time', 'end_time'}
        fps: 영상 FPS

    Returns:
        평균 wide_rows 값
    """
    cap = cv2.VideoCapture(video_path)
    start_frame = int(segment['start_time'] * fps)
    end_frame = int(segment['end_time'] * fps)
    wide_rows_list = []

    region = segment.get('region')

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        if region and len(region) == 5:
            _, rx, ry, rw, rh = region
            x1 = max(0, min(int(rx), width - 1))
            y1 = max(0, min(int(ry), height - 1))
            x2 = max(x1 + 1, min(int(rx + rw), width))
            y2 = max(y1 + 1, min(int(ry + rh), height))
            roi = frame[y1:y2, x1:x2]
        else:
            y1 = int(height * 0.7)
            roi = frame[y1:, :]

        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        row_counts = np.count_nonzero(thresh, axis=1)
        roi_width = roi.shape[1]
        if roi_width == 0:
            continue
        wide_rows = np.sum(row_counts > roi_width * 0.5)
        wide_rows_list.append(wide_rows)

    cap.release()
    return np.mean(wide_rows_list) if wide_rows_list else 0


def get_waveform_region_bounds(region, frame_shape):
    """
    파형 ROI 좌표 계산

    region 정보가 없으면 화면 하단 절반을 기본 영역으로 사용합니다.
    """
    height, width = frame_shape[:2]

    if region and len(region) == 5:
        _, x, y, w, h = region
    else:
        x, y = 0, int(height * 0.5)
        w, h = width, int(height * 0.5)

    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    x2 = min(x + int(w), width)
    y2 = min(y + int(h), height)

    if x2 <= x:
        x2 = min(width, x + 1)
    if y2 <= y:
        y2 = min(height, y + 1)

    return x, y, x2, y2


def build_combined_region(region_a, region_b, frame_shape):
    """
    두 ROI를 모두 포함하는 최소 사각형을 생성
    """
    coords = []
    for region in (region_a, region_b):
        if region and len(region) == 5:
            _, x, y, w, h = region
            coords.append((int(x), int(y), int(x + w), int(y + h)))

    if not coords:
        return None

    width = frame_shape[1]
    height = frame_shape[0]

    x1 = max(0, min(coord[0] for coord in coords))
    y1 = max(0, min(coord[1] for coord in coords))
    x2 = min(width, max(coord[2] for coord in coords))
    y2 = min(height, max(coord[3] for coord in coords))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return ('combined', x1, y1, x2 - x1, y2 - y1)


def detect_motion_in_waveform(prev_frame, curr_frame, region):
    """
    연속 프레임 간 파형 ROI 움직임 점수를 계산 (0~1 사이)
    """
    if prev_frame is None or curr_frame is None:
        return 0.0

    x1, y1, x2, y2 = get_waveform_region_bounds(region, prev_frame.shape)
    prev_roi = prev_frame[y1:y2, x1:x2]
    curr_roi = curr_frame[y1:y2, x1:x2]

    if prev_roi.size == 0 or curr_roi.size == 0:
        return 0.0

    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)

    motion_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    if total_pixels == 0:
        return 0.0

    return motion_pixels / total_pixels


def analyze_segment_motion(video_path, segment, fps,
                           motion_threshold=0.008,
                           min_pause_duration=0.5,
                           sample_stride=3):
    """
    세그먼트 내 프레임 움직임 분석
    """

    def default_result():
        duration = max(segment['end_time'] - segment['start_time'], 0.0)
        result = {
            'has_paused_sections': False,
            'paused_intervals': [],
            'moving_intervals': [],
            'moving_duration': duration,
            'paused_duration': 0.0,
            'avg_motion_score': 0.0,
            'max_motion_score': 0.0,
            'status': 'moving'
        }
        if duration > 0:
            result['moving_intervals'].append({'start': segment['start_time'], 'end': segment['end_time']})
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default_result()

    start_frame = max(int(segment['start_time'] * fps), 0)
    end_frame = max(int(segment['end_time'] * fps), start_frame + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return default_result()

    segment_region = segment.get('region')
    has_prev_waveform, prev_regions, _ = detect_waveform_region(prev_frame)
    active_region = prev_regions[0] if has_prev_waveform and prev_regions else segment_region

    frame_idx = start_frame + 1
    motion_curve = []
    prev_sample_frame = prev_frame
    frames_since_sample = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frames_since_sample += 1
        if frames_since_sample < sample_stride:
            frame_idx += 1
            continue

        curr_has_waveform, curr_regions, _ = detect_waveform_region(frame)
        current_region = curr_regions[0] if curr_has_waveform and curr_regions else None
        combined_region = build_combined_region(active_region, current_region, frame.shape)
        region_to_use = combined_region or current_region or active_region or segment_region
        score = detect_motion_in_waveform(prev_sample_frame, frame, region_to_use)
        timestamp = frame_idx / fps if fps else segment['start_time']
        motion_curve.append((timestamp, score))
        prev_sample_frame = frame
        active_region = current_region or active_region or region_to_use
        frames_since_sample = 0
        frame_idx += 1

    cap.release()

    if not motion_curve:
        return default_result()

    motion_scores = [score for _, score in motion_curve]
    paused_intervals = []
    moving_intervals = []
    paused_duration = 0.0
    moving_duration = 0.0
    current_state = None
    state_start = segment['start_time']

    def finalize_state(state, start, end):
        nonlocal paused_duration, moving_duration
        duration = max(end - start, 0.0)
        if duration <= 0 or state is None:
            return

        if state == 'paused':
            if duration >= min_pause_duration:
                paused_intervals.append({'start': start, 'end': end})
                paused_duration += duration
            else:
                moving_intervals.append({'start': start, 'end': end})
                moving_duration += duration
        else:
            moving_intervals.append({'start': start, 'end': end})
            moving_duration += duration

    for timestamp, score in motion_curve:
        state = 'paused' if score < motion_threshold else 'moving'
        if current_state is None:
            current_state = state
            state_start = segment['start_time']

        if state != current_state:
            finalize_state(current_state, state_start, timestamp)
            state_start = timestamp
            current_state = state

    finalize_state(current_state, state_start, segment['end_time'])

    result = {
        'has_paused_sections': bool(paused_intervals),
        'paused_intervals': paused_intervals,
        'moving_intervals': moving_intervals,
        'moving_duration': moving_duration,
        'paused_duration': paused_duration,
        'avg_motion_score': float(np.mean(motion_scores)) if motion_scores else 0.0,
        'max_motion_score': float(np.max(motion_scores)) if motion_scores else 0.0,
    }

    if paused_duration > 0 and paused_duration >= moving_duration:
        result['status'] = 'paused'
    else:
        result['status'] = 'moving'

    return result


# ============================================================================
# 파형 검출
# ============================================================================

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

    # Morphological operations로 노이즈 제거
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
            return True, [('orange', x, y, w, h)], area

    # ========================================
    # Type 2: Gray Doppler
    # ========================================
    default_y = int(height * 0.5)  # 하단 50% 영역
    default_h = int(height * 0.5)
    bottom_region = frame[default_y:, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(thresh)
    row_counts = np.count_nonzero(thresh, axis=1)
    wide_rows = np.sum(row_counts > width * 0.3)  # 완화
    left_region = thresh[:, :int(width * 0.6)]
    left_pixels = np.count_nonzero(left_region)
    left_ratio = (left_pixels / white_pixel_count * 100) if white_pixel_count > 0 else 0

    # 완화된 Gray 조건 (Fallback)
    if white_pixel_count > 500:
        return True, [('gray', 0, default_y, width, default_h)], white_pixel_count

    # Gray Doppler (좌측 집중형) 완화
    if white_pixel_count > 1000:
        roi_x2 = int(width * 0.6)
        gray_left = gray[:, :roi_x2]
        _, thresh_left = cv2.threshold(gray_left, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh_left = cv2.morphologyEx(thresh_left, cv2.MORPH_CLOSE, kernel)
        thresh_left = cv2.morphologyEx(thresh_left, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            y_global = default_y + y
            return True, [('gray_left', x, y_global, w_box, h_box)], white_pixel_count

    # ========================================
    # Type 3: Orange Fragmented
    # ========================================
    bottom_offset = int(height * 0.5)
    bottom_region_orange = orange_mask[bottom_offset:, :]
    orange_pixel_count = cv2.countNonZero(bottom_region_orange)

    if 5000 < orange_pixel_count < 80000:
        # 검증 1: 수평 분포
        row_counts_orange = np.count_nonzero(bottom_region_orange, axis=1)
        rows_with_pixels = np.sum(row_counts_orange > 10)

        # 검증 2: 좌측 영역 픽셀
        left_region_orange = bottom_region_orange[:, :int(width * 0.3)]
        left_orange_pixels = cv2.countNonZero(left_region_orange)
        left_orange_ratio = (left_orange_pixels / orange_pixel_count * 100) if orange_pixel_count > 0 else 0

        # Orange Fragmented 조건
        if rows_with_pixels > 20 and left_orange_pixels > 1000 and 10 < left_orange_ratio < 50:
            coords = cv2.findNonZero(bottom_region_orange)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                y += bottom_offset
            else:
                x, y, w, h = 0, bottom_offset, width, int(height * 0.5)
            return True, [('orange_fragmented', x, y, w, h)], orange_pixel_count

    return False, [], 0


# ============================================================================
# 구간 병합
# ============================================================================

def find_continuous_segments(waveform_frames, fps, gap_threshold=0.5):
    """
    연속된 파형 프레임을 구간으로 병합

    타입이 같고 시간 간격이 gap_threshold 이하인 프레임들을
    하나의 연속 구간으로 병합합니다.

    Args:
        waveform_frames: 파형이 검출된 프레임 리스트
        fps: 영상 FPS
        gap_threshold: 구간 병합 임계값 (초)

    Returns:
        구간 리스트 [{'start_time', 'end_time', 'type'}, ...]
    """
    if not waveform_frames:
        return []

    segments = []
    first_region = waveform_frames[0].get('regions')
    current_segment = {
        'start_frame': waveform_frames[0]['frame'],
        'start_time': waveform_frames[0]['timestamp'],
        'end_frame': waveform_frames[0]['frame'],
        'end_time': waveform_frames[0]['timestamp'],
        'type': waveform_frames[0].get('type'),
        'region': first_region[0] if first_region else None
    }

    for i in range(1, len(waveform_frames)):
        prev_time = waveform_frames[i-1]['timestamp']
        curr_time = waveform_frames[i]['timestamp']
        prev_type = waveform_frames[i-1].get('type')
        curr_type = waveform_frames[i].get('type')

        # 타입 정규화 (orange 계열 통일)
        normalized_prev = normalize_waveform_type(prev_type)
        normalized_curr = normalize_waveform_type(curr_type)

        # 시간상 연속이고 타입이 동일하면 병합
        if curr_time - prev_time <= gap_threshold and normalized_prev == normalized_curr:
            current_segment['end_frame'] = waveform_frames[i]['frame']
            current_segment['end_time'] = waveform_frames[i]['timestamp']
            if not current_segment.get('region'):
                regions = waveform_frames[i].get('regions')
                if regions:
                    current_segment['region'] = regions[0]
        else:
            segments.append(current_segment)
            regions = waveform_frames[i].get('regions')
            current_segment = {
                'start_frame': waveform_frames[i]['frame'],
                'start_time': waveform_frames[i]['timestamp'],
                'end_frame': waveform_frames[i]['frame'],
                'end_time': waveform_frames[i]['timestamp'],
                'type': waveform_frames[i].get('type'),
                'region': regions[0] if regions else None
            }

    segments.append(current_segment)
    return segments


def merge_adjacent_segments(segments, gap_threshold=0.5):
    """Merge neighboring segments separated by short gaps of the same type.

    Detection occasionally drops for a couple of frames or temporarily
    classifies the waveform differently. After filtering short segments, this
    helper stitches the surrounding valid segments back together when:
      * the time gap is shorter than ``gap_threshold`` seconds, and
      * their normalized waveform types match (``None`` acts as a wildcard).
    """
    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = max(seg['start_time'] - prev['end_time'], 0.0)

        prev_type = normalize_waveform_type(prev.get('type'))
        curr_type = normalize_waveform_type(seg.get('type'))
        types_match = (
            prev_type == curr_type
            or prev_type is None
            or curr_type is None
        )

        if gap <= gap_threshold and types_match:
            # Extend the previous segment to include the current one
            if seg.get('end_time', prev['end_time']) > prev['end_time']:
                prev['end_time'] = seg['end_time']
                if seg.get('end_frame') is not None:
                    prev['end_frame'] = seg['end_frame']

            if not prev.get('region') and seg.get('region'):
                prev['region'] = seg['region']
            continue

        merged.append(seg)

    return merged


# ============================================================================
# 메인 분석 함수
# ============================================================================

def analyze_video_cv(video_path, verbose=True):
    """
    영상 전체 분석 및 파형 구간 검출

    파이프라인:
      1. 프레임별 파형 검출 (detect_waveform_region)
      2. 연속 구간 병합 (find_continuous_segments)
      3. 통계적 필터링 (false positive 제거)

    Args:
        video_path: 분석할 영상 파일 경로

    Returns:
        검출된 구간 리스트
    """
    # ========================================
    # Stage 1: 영상 정보 로드
    # ========================================
    cap = cv2.VideoCapture(video_path)

    # 영상 열기 실패 검사
    if not cap.isOpened():
        print(f"영상 열기 실패: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FPS 유효성 검사 (0, NaN, 매우 작은 값)
    if not fps or np.isnan(fps) or fps <= 1e-6:
        print("경고: FPS 미확인 → 30.0으로 대체")
        fps = 30.0

    video_duration = (total_frames / fps) if fps else 0

    def log(message):
        if verbose:
            print(message)

    log(f"\n영상: {video_path}")
    log(f"  FPS: {fps}, 총 프레임: {total_frames}, 길이: {video_duration:.1f}초")

    # ========================================
    # Stage 2: 프레임별 파형 검출
    # ========================================
    waveform_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        has_waveform, regions, pixel_count = detect_waveform_region(frame)

        if has_waveform:
            region_type = regions[0][0] if regions else None
            timestamp = frame_idx / fps
            waveform_frames.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'regions': regions,
                'pixel_count': pixel_count,
                'type': region_type,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0]
            })

        frame_idx += 1

        if total_frames and frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            log(f"  진행: {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()

    # ========================================
    # Stage 3: 연속 구간 병합
    # ========================================
    segments = find_continuous_segments(waveform_frames, fps)

    # ========================================
    # Stage 4: 통계적 필터링
    # ========================================
    valid_segments = []
    filtered_short_segments = []

    for seg in segments:
        seg_duration = seg['end_time'] - seg['start_time']

        # 구간 통계 계산
        seg_frames = [f for f in waveform_frames
                      if seg['start_time'] <= f['timestamp'] <= seg['end_time']]
        if not seg_frames:
            continue

        pixel_counts = [f['pixel_count'] for f in seg_frames if f['pixel_count'] > 0]
        if not pixel_counts:
            continue

        std_dev = np.std(pixel_counts)
        avg_count = np.mean(pixel_counts)
        variation_ratio = std_dev / avg_count if avg_count > 0 else 0

        segment_type = normalize_waveform_type(seg_frames[0].get('type'))
        region_meta = seg.get('region')
        density = None
        x_ratio = None
        width_box = None
        height_box = None
        if region_meta and len(region_meta) >= 5:
            _, rx, ry, rw, rh = region_meta
            width_box = rw
            height_box = rh
            area = max(rw * rh, 1)
            density = avg_count / area if avg_count > 0 else 0
            frame_width = seg_frames[0].get('frame_width')
            if frame_width:
                x_ratio = rx / frame_width

        avg_wide_rows = None
        if segment_type in ('gray', 'gray_left'):
            avg_wide_rows = calculate_avg_wide_rows(video_path, seg, fps)

        # 짧은 구간 허용 조건: 강한 신호 + 최소 길이 확보
        short_but_confident = (
            seg_duration >= SHORT_CONFIDENT_MIN_DURATION
            and avg_count >= SHORT_STRONG_PIXEL_COUNT
            and (avg_wide_rows is None or avg_wide_rows >= SHORT_MIN_WIDE_ROWS)
        )
        allow_density_bypass = short_but_confident and seg_duration < MIN_SEG_DURATION

        # 필터 1: 지속 시간 (짧지만 확신 구간만 예외 허용)
        if seg_duration < MIN_SEG_DURATION and not short_but_confident:
            filtered_short_segments.append(seg)
            continue

        # 필터 2: Gray 타입 false positive 제거
        if segment_type == 'gray':
            # 필터 2-1: 화면 전체가 밝은 정적 화면 (wide_rows↑ + 변동↓)
            if avg_wide_rows is not None and avg_wide_rows > 100 and variation_ratio < 0.05:
                continue
            # 정적 고밝기 화면 추가 컷
            if avg_wide_rows is not None and avg_wide_rows > 60 and variation_ratio < 0.08:
                continue

            # 필터 2-2: 고밝기 + 좁은 분포 = false positive
            if avg_count > 115000 and avg_wide_rows is not None and avg_wide_rows < 39:
                continue

            if density is not None and not allow_density_bypass:
                bypass_density_checks = (avg_wide_rows is not None and avg_wide_rows >= 35)
                if not bypass_density_checks:
                    if density < 1.0:
                        if (x_ratio is not None and x_ratio > 0.22) or \
                           (width_box is not None and height_box is not None and width_box >= 120 and height_box >= 120):
                            continue
                    if density < 2.0 and x_ratio is not None and x_ratio > 0.3:
                        if avg_wide_rows is None or avg_wide_rows < 35:
                            continue

            # 필터 2-3: 좌측 영역을 벗어난 넓은 회색 영역 (UI/배경 노이즈)
            if avg_wide_rows is not None and avg_wide_rows > 40 and x_ratio is not None and x_ratio > 0.6:
                continue

            # 필터 2-4: 저밝기 + 좁은 row + 짧은 길이 → 잡음 제거 (임시 비활성: 리콜 우선)
            # if seg_duration < 2.0 and avg_count < 40000 and (avg_wide_rows is not None and avg_wide_rows < 12):
            #     continue

        # 필터 3: 밝기별 변동 비율 검증
        if avg_count <= 150000:
            # 저밝기: 통과
            valid_segments.append(seg)
        elif avg_count <= 180000:
            # 중밝기: 변동 비율 9% 이상
            if variation_ratio > 0.09:
                valid_segments.append(seg)
        else:
            # 고밝기: 변동 비율 25% 이상
            if variation_ratio > 0.25:
                valid_segments.append(seg)

    # 짧은 간격으로 끊어진 동일 타임의 구간을 다시 연결
    valid_segments = merge_adjacent_segments(valid_segments, gap_threshold=0.7)

    # 너무 긴 구간은 모션 세분화 전에 분할하여 과도한 병합을 방지
    split_segments = []
    for seg in valid_segments:
        seg_duration = seg['end_time'] - seg['start_time']
        if seg_duration <= MAX_SEG_DURATION_BEFORE_MOTION:
            split_segments.append(seg)
            continue

        # 균등하게 자르기
        num_chunks = int(np.ceil(seg_duration / MAX_SEG_DURATION_BEFORE_MOTION))
        chunk_len = seg_duration / num_chunks
        for i in range(num_chunks):
            start = seg['start_time'] + i * chunk_len
            end = seg['start_time'] + (i + 1) * chunk_len
            split_segments.append({
                'start_time': start,
                'end_time': end,
                'type': seg.get('type'),
                'region': seg.get('region'),
                'start_frame': int(start * fps) if fps else seg.get('start_frame'),
                'end_frame': int(end * fps) if fps else seg.get('end_frame'),
            })

    valid_segments = split_segments

    # 감지 구간 앞뒤로 안전 패딩을 추가하여 경계 오차를 줄임
    for seg in valid_segments:
        seg['start_time'] = max(0.0, seg['start_time'] - EDGE_PAD_BEFORE)
        seg['end_time'] = min(video_duration, seg['end_time'] + EDGE_PAD_AFTER)
        if fps:
            seg['start_frame'] = max(0, int(seg['start_time'] * fps))
            seg['end_frame'] = max(seg['start_frame'], int(seg['end_time'] * fps))

    # 패딩으로 인접 구간이 붙으면 다시 머지
    valid_segments = merge_adjacent_segments(valid_segments, gap_threshold=0.7)

    # ========================================
    # Stage 5: 움직임 분석
    # ========================================
    for seg in valid_segments:
        motion_info = analyze_segment_motion(video_path, seg, fps)
        seg['motion_analysis'] = motion_info

        moving_subsegments = []
        for interval in motion_info.get('moving_intervals', []):
            start = interval.get('start')
            end = interval.get('end')
            if start is None or end is None:
                continue
            duration = max(end - start, 0.0)
            if duration <= 0:
                continue
            moving_subsegments.append({
                'start_time': start,
                'end_time': end,
                'duration': duration,
                'type': seg.get('type', 'unknown')
            })

        seg['moving_subsegments'] = moving_subsegments

    # 회색 긴 정적 구간 제거 (리콜보다 FP 억제 우선)
    filtered_motion_segments = []
    # ========================================
    # Stage 6: 움직임 기반 세분화 (모든 타입 적용)
    # ========================================
    segments_to_report = []

    for seg in valid_segments:
        seg_duration = seg['end_time'] - seg['start_time']
        seg_type = normalize_waveform_type(seg.get('type'))
        motion = seg.get('motion_analysis', {})
        moving = motion.get('moving_duration', 0.0)
        move_ratio = moving / seg_duration if seg_duration > 0 else 0.0

        # 모션 비율이 너무 낮으면 폐기
        if move_ratio < 0.3:
            continue

        # 긴 회색 정적 구간 컷
        if seg_type in ('gray', 'gray_left') and seg_duration > 8.0 and move_ratio < 0.5:
            continue

        moving_subs = seg.get('moving_subsegments') or []
        filtered_moves = [ms for ms in moving_subs if ms.get('duration', 0) >= MOTION_KEEP_MIN_DURATION]
        if not filtered_moves:
            continue

        # 모션 세분화
        filtered_moves.sort(key=lambda m: m['start_time'])
        merged = [filtered_moves[0].copy()]
        for ms in filtered_moves[1:]:
            gap = ms['start_time'] - merged[-1]['end_time']
            if gap <= MOTION_MERGE_GAP:
                merged[-1]['end_time'] = max(merged[-1]['end_time'], ms['end_time'])
                merged[-1]['duration'] = merged[-1]['end_time'] - merged[-1]['start_time']
            else:
                merged.append(ms.copy())

        for ms in merged:
            start = max(0.0, ms['start_time'] - MOTION_PAD_BEFORE)
            end = min(video_duration, ms['end_time'] + MOTION_PAD_AFTER)
            if end - start < MIN_REFINED_SEGMENT:
                continue
            segments_to_report.append({
                'start_time': start,
                'end_time': end,
                'type': seg.get('type'),
                'region': seg.get('region'),
            })

    segments_to_report = merge_adjacent_segments(sorted(segments_to_report, key=lambda s: s['start_time']), gap_threshold=0.2)
    segments_to_report = [
        seg for seg in segments_to_report
        if (seg['end_time'] - seg['start_time']) >= MIN_FINAL_SEG_DURATION
        and (seg['end_time'] - seg['start_time']) <= MAX_FINAL_SEG_DURATION
    ]
    if fps:
        for seg in segments_to_report:
            seg['start_frame'] = max(0, int(seg['start_time'] * fps))
            seg['end_frame'] = max(seg['start_frame'], int(seg['end_time'] * fps))

    # ========================================
    # 결과 출력
    # ========================================
    log(f"\n  파형 감지 구간: {len(segments_to_report)}개")
    for i, seg in enumerate(segments_to_report, 1):
        duration = seg['end_time'] - seg['start_time']
        seg_type = seg.get('type', 'unknown')
        log(f"    구간 {i}: {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 ({duration:.2f}초) [type: {seg_type}]")
        motion_info = seg.get('motion_analysis')
        if motion_info:
            moving_dur = motion_info.get('moving_duration', 0.0)
            paused_dur = motion_info.get('paused_duration', 0.0)
            status = motion_info.get('status', 'moving')
            moving_pct = (moving_dur / duration * 100) if duration > 0 else 0
            log(f"      움직임 요약: {status}, 동작 {moving_dur:.2f}초 ({moving_pct:.1f}%), 정지 {paused_dur:.2f}초")
            if motion_info.get('has_paused_sections'):
                log(f"      정지 구간 ({len(motion_info['paused_intervals'])}개):")
                for pause in motion_info['paused_intervals']:
                    log(f"        정지 구간: {pause['start']:.2f}초 ~ {pause['end']:.2f}초")
        moving_segments = seg.get('moving_subsegments', [])
        if moving_segments:
            log(f"      동작 구간 ({len(moving_segments)}개):")
            for ms in moving_segments:
                log(f"        동작 구간: {ms['start_time']:.2f}초 ~ {ms['end_time']:.2f}초 ({ms['duration']:.2f}초)")

    if filtered_short_segments:
        log(f"\n  짧은 구간 (3초 미만, 필터링됨): {len(filtered_short_segments)}개")
        for i, seg in enumerate(filtered_short_segments, 1):
            duration = seg['end_time'] - seg['start_time']
            seg_type = seg.get('type', 'unknown')
            min_sec = int(seg['start_time'] // 60)
            start_sec = seg['start_time'] % 60
            log(f"    {i}. {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 ({duration:.2f}초) [{min_sec}분 {start_sec:.1f}초, type: {seg_type}]")

    return segments_to_report


# ============================================================================
# 퍼블릭 API: 기본은 시계열(1D) 방식
# ============================================================================

def analyze_video(video_path, verbose=True, mode='timeseries'):
    """
    영상 분석 및 파형 구간 검출

    Args:
        video_path: 분석할 영상 파일 경로
        verbose: 로그 출력 여부
        mode: 'timeseries'(기본) 또는 'cv'
    """
    if mode in ('cv', 'vision', 'opencv'):
        return analyze_video_cv(video_path, verbose=verbose)
    return analyze_video_timeseries(video_path, verbose=verbose)


# ============================================================================
# 메인 엔트리 포인트
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'assets/heartbeat-samples/28w-126bpm.mp4'

    analyze_video(video_path)
