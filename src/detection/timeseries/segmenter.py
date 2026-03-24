"""
Segment generation for timeseries waveform detection.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .features import extract_timeseries_features, smooth_1d


def find_binary_segments(times, mask, min_duration_sec, merge_gap_sec):
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


def adaptive_hysteresis_threshold(series, min_on, min_off):
    series = np.asarray(series, dtype=np.float64)
    if series.size == 0:
        return float(min_on), float(min_off)

    high = float(np.quantile(series, config.TS_HYSTERESIS_HIGH_Q))
    on = max(float(min_on), high * float(config.TS_HYSTERESIS_ON_SCALE))
    off = max(float(min_off), high * float(config.TS_HYSTERESIS_OFF_SCALE))
    if off >= on:
        off = on * 0.8
    return float(on), float(off)


def refine_segment_bounds(times_arr, raw_series, start, end, threshold):
    idx = np.where((times_arr >= start) & (times_arr <= end) & (raw_series >= threshold))[0]
    if idx.size == 0:
        return float(start), float(end)
    return float(times_arr[int(idx[0])]), float(times_arr[int(idx[-1])])


def analyze_video_timeseries(video_path, verbose=True):
    """ROI 고정/시계열 기반으로 파형 구간을 검출."""
    video_path = str(video_path)
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
        if frame_idx % config.TS_STRIDE_FRAMES != 0:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        black_ratio, orange_score, white_score, top_black_ratio = extract_timeseries_features(frame)
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
    orange_s = smooth_1d(orange_raw, config.TS_SMOOTH_WINDOW)
    white_s = smooth_1d(white_raw, config.TS_SMOOTH_WINDOW)
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

    orange_on, orange_off = adaptive_hysteresis_threshold(
        orange_arr,
        config.TS_ORANGE_MIN_ON,
        config.TS_ORANGE_MIN_OFF,
    )
    white_on, white_off = adaptive_hysteresis_threshold(
        white_arr,
        config.TS_WHITE_MIN_ON,
        config.TS_WHITE_MIN_OFF,
    )

    # 구간 밖 오탐을 막기 위해, 동영상별 적응형 임계값 대신
    # 고정 임계값(최소치)을 사용합니다.
    orange_on, orange_off = config.TS_ORANGE_MIN_ON, config.TS_ORANGE_MIN_OFF
    white_on, white_off = config.TS_WHITE_MIN_ON, config.TS_WHITE_MIN_OFF

    orange_mask = hysteresis_mask(orange_arr, orange_on, orange_off)

    # 오렌지 파형이 충분히 존재하는 프레임에서는 white 채널을 비활성화하여
    # (오렌지 파형이 white 임계값까지 끌어올리는 경우) 중복/오탐을 줄입니다.
    white_for_detect = np.where(
        orange_arr >= config.TS_ORANGE_SUPPRESS_WHITE_THRESHOLD,
        0.0,
        white_arr,
    )
    white_mask = hysteresis_mask(white_for_detect, white_on, white_off)

    orange_segments = find_binary_segments(
        times,
        orange_mask,
        config.TS_ORANGE_MIN_SEGMENT_SEC,
        config.TS_MERGE_GAP_SEC,
    )
    white_segments = find_binary_segments(
        times,
        white_mask,
        config.TS_WHITE_MIN_SEGMENT_SEC,
        config.TS_MERGE_GAP_SEC,
    )

    segments = sorted(orange_segments + white_segments)
    merged_segments = []
    for s, e in segments:
        if not merged_segments:
            merged_segments.append([s, e])
            continue
        if s - merged_segments[-1][1] <= config.TS_MERGE_GAP_SEC:
            merged_segments[-1][1] = max(merged_segments[-1][1], e)
        else:
            merged_segments.append([s, e])
    segments = [(float(s), float(e)) for s, e in merged_segments]

    segments_to_report = []
    for start, end in segments:
        start = max(0.0, start)
        end = min(video_duration, end)
        duration = end - start
        if duration < config.TS_WHITE_MIN_SEGMENT_SEC or duration > config.TS_MAX_SEGMENT_SEC:
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

        if peak_top_black < config.TS_TOP_BLACK_GATE_MIN:
            continue

        if duration <= config.TS_SHORT_SEGMENT_SEC:
            if mean_black_ratio < config.TS_SHORT_BLACK_RATIO_MIN:
                continue
            if peak_orange < config.TS_SHORT_ORANGE_PEAK_MIN and peak_white < config.TS_SHORT_WHITE_PEAK_MIN:
                continue
            if peak_orange >= peak_white and mean_orange < config.TS_SHORT_ORANGE_MEAN_MIN:
                continue

        refined_start, refined_end = refine_segment_bounds(
            times_arr,
            np.maximum(orange_raw, white_raw),
            start,
            end,
            min(orange_off, white_off),
        )
        start, end = refined_start, refined_end
        if end - start < config.TS_WHITE_MIN_SEGMENT_SEC:
            continue

        segments_to_report.append(
            {
                "start_time": start,
                "end_time": end,
                "type": "timeseries",
            }
        )

    if fps:
        for seg in segments_to_report:
            seg["start_frame"] = max(0, int(seg["start_time"] * fps))
            seg["end_frame"] = max(seg["start_frame"], int(seg["end_time"] * fps))

    log(f"\n  [TS] 파형 감지 구간: {len(segments_to_report)}개")
    for i, seg in enumerate(segments_to_report, 1):
        dur = seg["end_time"] - seg["start_time"]
        log(f"    구간 {i}: {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 ({dur:.2f}초)")

    return segments_to_report


__all__ = ["analyze_video_timeseries"]
