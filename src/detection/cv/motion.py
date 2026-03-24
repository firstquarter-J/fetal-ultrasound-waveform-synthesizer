"""
Motion analysis helpers for the OpenCV detection path.
"""

from __future__ import annotations

import cv2
import numpy as np

from .frame_detector import detect_waveform_region


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

    return ("combined", x1, y1, x2 - x1, y2 - y1)


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


def analyze_segment_motion(
    video_path,
    segment,
    fps,
    motion_threshold=0.008,
    min_pause_duration=0.5,
    sample_stride=3,
):
    """
    세그먼트 내 프레임 움직임 분석
    """

    def default_result():
        duration = max(segment["end_time"] - segment["start_time"], 0.0)
        result = {
            "has_paused_sections": False,
            "paused_intervals": [],
            "moving_intervals": [],
            "moving_duration": duration,
            "paused_duration": 0.0,
            "avg_motion_score": 0.0,
            "max_motion_score": 0.0,
            "status": "moving",
        }
        if duration > 0:
            result["moving_intervals"].append(
                {"start": segment["start_time"], "end": segment["end_time"]}
            )
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default_result()

    start_frame = max(int(segment["start_time"] * fps), 0)
    end_frame = max(int(segment["end_time"] * fps), start_frame + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return default_result()

    segment_region = segment.get("region")
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
        timestamp = frame_idx / fps if fps else segment["start_time"]
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
    state_start = segment["start_time"]

    def finalize_state(state, start, end):
        nonlocal paused_duration, moving_duration
        duration = max(end - start, 0.0)
        if duration <= 0 or state is None:
            return

        if state == "paused":
            if duration >= min_pause_duration:
                paused_intervals.append({"start": start, "end": end})
                paused_duration += duration
            else:
                moving_intervals.append({"start": start, "end": end})
                moving_duration += duration
        else:
            moving_intervals.append({"start": start, "end": end})
            moving_duration += duration

    for timestamp, score in motion_curve:
        state = "paused" if score < motion_threshold else "moving"
        if current_state is None:
            current_state = state
            state_start = segment["start_time"]

        if state != current_state:
            finalize_state(current_state, state_start, timestamp)
            state_start = timestamp
            current_state = state

    finalize_state(current_state, state_start, segment["end_time"])

    result = {
        "has_paused_sections": bool(paused_intervals),
        "paused_intervals": paused_intervals,
        "moving_intervals": moving_intervals,
        "moving_duration": moving_duration,
        "paused_duration": paused_duration,
        "avg_motion_score": float(np.mean(motion_scores)) if motion_scores else 0.0,
        "max_motion_score": float(np.max(motion_scores)) if motion_scores else 0.0,
    }

    if paused_duration > 0 and paused_duration >= moving_duration:
        result["status"] = "paused"
    else:
        result["status"] = "moving"

    return result


__all__ = [
    "analyze_segment_motion",
    "build_combined_region",
    "detect_motion_in_waveform",
    "get_waveform_region_bounds",
]
