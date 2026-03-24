"""
Post-processing and full-video orchestration for the OpenCV detection path.
"""

from __future__ import annotations

import cv2
import numpy as np

from .frame_detector import detect_waveform_region
from .motion import analyze_segment_motion


MIN_SEG_DURATION = 3.0
SHORT_CONFIDENT_MIN_DURATION = 1.0
SHORT_STRONG_PIXEL_COUNT = 50000
SHORT_MIN_WIDE_ROWS = 30
EDGE_PAD_BEFORE = 1.2
EDGE_PAD_AFTER = 1.0
MOTION_KEEP_MIN_DURATION = 0.1
MOTION_MERGE_GAP = 0.3
MOTION_PAD_BEFORE = 0.4
MOTION_PAD_AFTER = 0.4
MIN_REFINED_SEGMENT = 1.0
MIN_FINAL_SEG_DURATION = 1.0
MAX_SEG_DURATION_BEFORE_MOTION = 4.0
MAX_FINAL_SEG_DURATION = 12.0


def normalize_waveform_type(waveform_type):
    """
    파형 타입 정규화

    orange와 orange_fragmented를 같은 타입으로 취급하여
    단일 프레임 타입 변경으로 인한 구간 분리를 방지합니다.
    """
    if waveform_type in ["orange", "orange_fragmented"]:
        return "orange"
    if waveform_type in ["gray_left"]:
        return "gray"
    return waveform_type


def calculate_avg_wide_rows(video_path, segment, fps):
    """
    구간의 평균 wide_rows 계산

    Wide rows: 화면 너비의 50% 이상을 차지하는 행의 개수
    실제 파형과 false positive를 구분하는 핵심 지표입니다.
    """
    cap = cv2.VideoCapture(video_path)
    start_frame = int(segment["start_time"] * fps)
    end_frame = int(segment["end_time"] * fps)
    wide_rows_list = []

    region = segment.get("region")

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


def find_continuous_segments(waveform_frames, fps, gap_threshold=0.5):
    """
    연속된 파형 프레임을 구간으로 병합
    """
    if not waveform_frames:
        return []

    segments = []
    first_region = waveform_frames[0].get("regions")
    current_segment = {
        "start_frame": waveform_frames[0]["frame"],
        "start_time": waveform_frames[0]["timestamp"],
        "end_frame": waveform_frames[0]["frame"],
        "end_time": waveform_frames[0]["timestamp"],
        "type": waveform_frames[0].get("type"),
        "region": first_region[0] if first_region else None,
    }

    for i in range(1, len(waveform_frames)):
        prev_time = waveform_frames[i - 1]["timestamp"]
        curr_time = waveform_frames[i]["timestamp"]
        prev_type = waveform_frames[i - 1].get("type")
        curr_type = waveform_frames[i].get("type")

        normalized_prev = normalize_waveform_type(prev_type)
        normalized_curr = normalize_waveform_type(curr_type)

        if curr_time - prev_time <= gap_threshold and normalized_prev == normalized_curr:
            current_segment["end_frame"] = waveform_frames[i]["frame"]
            current_segment["end_time"] = waveform_frames[i]["timestamp"]
            if not current_segment.get("region"):
                regions = waveform_frames[i].get("regions")
                if regions:
                    current_segment["region"] = regions[0]
        else:
            segments.append(current_segment)
            regions = waveform_frames[i].get("regions")
            current_segment = {
                "start_frame": waveform_frames[i]["frame"],
                "start_time": waveform_frames[i]["timestamp"],
                "end_frame": waveform_frames[i]["frame"],
                "end_time": waveform_frames[i]["timestamp"],
                "type": waveform_frames[i].get("type"),
                "region": regions[0] if regions else None,
            }

    segments.append(current_segment)
    return segments


def merge_adjacent_segments(segments, gap_threshold=0.5):
    """Merge neighboring segments separated by short gaps of the same type."""
    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = max(seg["start_time"] - prev["end_time"], 0.0)

        prev_type = normalize_waveform_type(prev.get("type"))
        curr_type = normalize_waveform_type(seg.get("type"))
        types_match = prev_type == curr_type or prev_type is None or curr_type is None

        if gap <= gap_threshold and types_match:
            if seg.get("end_time", prev["end_time"]) > prev["end_time"]:
                prev["end_time"] = seg["end_time"]
                if seg.get("end_frame") is not None:
                    prev["end_frame"] = seg["end_frame"]

            if not prev.get("region") and seg.get("region"):
                prev["region"] = seg["region"]
            continue

        merged.append(seg)

    return merged


def analyze_video_cv(video_path, verbose=True):
    """
    영상 전체 분석 및 파형 구간 검출

    파이프라인:
      1. 프레임별 파형 검출
      2. 연속 구간 병합
      3. 통계적 필터링
      4. 모션 기반 세분화
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"영상 열기 실패: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or np.isnan(fps) or fps <= 1e-6:
        print("경고: FPS 미확인 → 30.0으로 대체")
        fps = 30.0

    video_duration = (total_frames / fps) if fps else 0

    def log(message):
        if verbose:
            print(message)

    log(f"\n영상: {video_path}")
    log(f"  FPS: {fps}, 총 프레임: {total_frames}, 길이: {video_duration:.1f}초")

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
            waveform_frames.append(
                {
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "regions": regions,
                    "pixel_count": pixel_count,
                    "type": region_type,
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
                }
            )

        frame_idx += 1

        if total_frames and frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            log(f"  진행: {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()

    segments = find_continuous_segments(waveform_frames, fps)

    valid_segments = []
    filtered_short_segments = []

    for seg in segments:
        seg_duration = seg["end_time"] - seg["start_time"]
        seg_frames = [
            f
            for f in waveform_frames
            if seg["start_time"] <= f["timestamp"] <= seg["end_time"]
        ]
        if not seg_frames:
            continue

        pixel_counts = [f["pixel_count"] for f in seg_frames if f["pixel_count"] > 0]
        if not pixel_counts:
            continue

        std_dev = np.std(pixel_counts)
        avg_count = np.mean(pixel_counts)
        variation_ratio = std_dev / avg_count if avg_count > 0 else 0

        segment_type = normalize_waveform_type(seg_frames[0].get("type"))
        region_meta = seg.get("region")
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
            frame_width = seg_frames[0].get("frame_width")
            if frame_width:
                x_ratio = rx / frame_width

        avg_wide_rows = None
        if segment_type in ("gray", "gray_left"):
            avg_wide_rows = calculate_avg_wide_rows(video_path, seg, fps)

        short_but_confident = (
            seg_duration >= SHORT_CONFIDENT_MIN_DURATION
            and avg_count >= SHORT_STRONG_PIXEL_COUNT
            and (avg_wide_rows is None or avg_wide_rows >= SHORT_MIN_WIDE_ROWS)
        )
        allow_density_bypass = short_but_confident and seg_duration < MIN_SEG_DURATION

        if seg_duration < MIN_SEG_DURATION and not short_but_confident:
            filtered_short_segments.append(seg)
            continue

        if segment_type == "gray":
            if avg_wide_rows is not None and avg_wide_rows > 100 and variation_ratio < 0.05:
                continue
            if avg_wide_rows is not None and avg_wide_rows > 60 and variation_ratio < 0.08:
                continue
            if avg_count > 115000 and avg_wide_rows is not None and avg_wide_rows < 39:
                continue

            if density is not None and not allow_density_bypass:
                bypass_density_checks = avg_wide_rows is not None and avg_wide_rows >= 35
                if not bypass_density_checks:
                    if density < 1.0:
                        if (x_ratio is not None and x_ratio > 0.22) or (
                            width_box is not None
                            and height_box is not None
                            and width_box >= 120
                            and height_box >= 120
                        ):
                            continue
                    if density < 2.0 and x_ratio is not None and x_ratio > 0.3:
                        if avg_wide_rows is None or avg_wide_rows < 35:
                            continue

            if avg_wide_rows is not None and avg_wide_rows > 40 and x_ratio is not None and x_ratio > 0.6:
                continue

        if avg_count <= 150000:
            valid_segments.append(seg)
        elif avg_count <= 180000:
            if variation_ratio > 0.09:
                valid_segments.append(seg)
        else:
            if variation_ratio > 0.25:
                valid_segments.append(seg)

    valid_segments = merge_adjacent_segments(valid_segments, gap_threshold=0.7)

    split_segments = []
    for seg in valid_segments:
        seg_duration = seg["end_time"] - seg["start_time"]
        if seg_duration <= MAX_SEG_DURATION_BEFORE_MOTION:
            split_segments.append(seg)
            continue

        num_chunks = int(np.ceil(seg_duration / MAX_SEG_DURATION_BEFORE_MOTION))
        chunk_len = seg_duration / num_chunks
        for i in range(num_chunks):
            start = seg["start_time"] + i * chunk_len
            end = seg["start_time"] + (i + 1) * chunk_len
            split_segments.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "type": seg.get("type"),
                    "region": seg.get("region"),
                    "start_frame": int(start * fps) if fps else seg.get("start_frame"),
                    "end_frame": int(end * fps) if fps else seg.get("end_frame"),
                }
            )

    valid_segments = split_segments

    for seg in valid_segments:
        seg["start_time"] = max(0.0, seg["start_time"] - EDGE_PAD_BEFORE)
        seg["end_time"] = min(video_duration, seg["end_time"] + EDGE_PAD_AFTER)
        if fps:
            seg["start_frame"] = max(0, int(seg["start_time"] * fps))
            seg["end_frame"] = max(seg["start_frame"], int(seg["end_time"] * fps))

    valid_segments = merge_adjacent_segments(valid_segments, gap_threshold=0.7)

    for seg in valid_segments:
        motion_info = analyze_segment_motion(video_path, seg, fps)
        seg["motion_analysis"] = motion_info

        moving_subsegments = []
        for interval in motion_info.get("moving_intervals", []):
            start = interval.get("start")
            end = interval.get("end")
            if start is None or end is None:
                continue
            duration = max(end - start, 0.0)
            if duration <= 0:
                continue
            moving_subsegments.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                    "type": seg.get("type", "unknown"),
                }
            )

        seg["moving_subsegments"] = moving_subsegments

    segments_to_report = []

    for seg in valid_segments:
        seg_duration = seg["end_time"] - seg["start_time"]
        seg_type = normalize_waveform_type(seg.get("type"))
        motion = seg.get("motion_analysis", {})
        moving = motion.get("moving_duration", 0.0)
        move_ratio = moving / seg_duration if seg_duration > 0 else 0.0

        if move_ratio < 0.3:
            continue

        if seg_type in ("gray", "gray_left") and seg_duration > 8.0 and move_ratio < 0.5:
            continue

        moving_subs = seg.get("moving_subsegments") or []
        filtered_moves = [
            ms for ms in moving_subs if ms.get("duration", 0) >= MOTION_KEEP_MIN_DURATION
        ]
        if not filtered_moves:
            continue

        filtered_moves.sort(key=lambda m: m["start_time"])
        merged = [filtered_moves[0].copy()]
        for ms in filtered_moves[1:]:
            gap = ms["start_time"] - merged[-1]["end_time"]
            if gap <= MOTION_MERGE_GAP:
                merged[-1]["end_time"] = max(merged[-1]["end_time"], ms["end_time"])
                merged[-1]["duration"] = merged[-1]["end_time"] - merged[-1]["start_time"]
            else:
                merged.append(ms.copy())

        for ms in merged:
            start = max(0.0, ms["start_time"] - MOTION_PAD_BEFORE)
            end = min(video_duration, ms["end_time"] + MOTION_PAD_AFTER)
            if end - start < MIN_REFINED_SEGMENT:
                continue
            segments_to_report.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "type": seg.get("type"),
                    "region": seg.get("region"),
                }
            )

    segments_to_report = merge_adjacent_segments(
        sorted(segments_to_report, key=lambda s: s["start_time"]),
        gap_threshold=0.2,
    )
    segments_to_report = [
        seg
        for seg in segments_to_report
        if (seg["end_time"] - seg["start_time"]) >= MIN_FINAL_SEG_DURATION
        and (seg["end_time"] - seg["start_time"]) <= MAX_FINAL_SEG_DURATION
    ]
    if fps:
        for seg in segments_to_report:
            seg["start_frame"] = max(0, int(seg["start_time"] * fps))
            seg["end_frame"] = max(seg["start_frame"], int(seg["end_time"] * fps))

    log(f"\n  파형 감지 구간: {len(segments_to_report)}개")
    for i, seg in enumerate(segments_to_report, 1):
        duration = seg["end_time"] - seg["start_time"]
        seg_type = seg.get("type", "unknown")
        log(
            f"    구간 {i}: {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 "
            f"({duration:.2f}초) [type: {seg_type}]"
        )
        motion_info = seg.get("motion_analysis")
        if motion_info:
            moving_dur = motion_info.get("moving_duration", 0.0)
            paused_dur = motion_info.get("paused_duration", 0.0)
            status = motion_info.get("status", "moving")
            moving_pct = (moving_dur / duration * 100) if duration > 0 else 0
            log(
                f"      움직임 요약: {status}, 동작 {moving_dur:.2f}초 "
                f"({moving_pct:.1f}%), 정지 {paused_dur:.2f}초"
            )
            if motion_info.get("has_paused_sections"):
                log(f"      정지 구간 ({len(motion_info['paused_intervals'])}개):")
                for pause in motion_info["paused_intervals"]:
                    log(
                        f"        정지 구간: {pause['start']:.2f}초 ~ "
                        f"{pause['end']:.2f}초"
                    )
        moving_segments = seg.get("moving_subsegments", [])
        if moving_segments:
            log(f"      동작 구간 ({len(moving_segments)}개):")
            for ms in moving_segments:
                log(
                    f"        동작 구간: {ms['start_time']:.2f}초 ~ "
                    f"{ms['end_time']:.2f}초 ({ms['duration']:.2f}초)"
                )

    if filtered_short_segments:
        log(f"\n  짧은 구간 (3초 미만, 필터링됨): {len(filtered_short_segments)}개")
        for i, seg in enumerate(filtered_short_segments, 1):
            duration = seg["end_time"] - seg["start_time"]
            seg_type = seg.get("type", "unknown")
            min_sec = int(seg["start_time"] // 60)
            start_sec = seg["start_time"] % 60
            log(
                f"    {i}. {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 "
                f"({duration:.2f}초) [{min_sec}분 {start_sec:.1f}초, type: {seg_type}]"
            )

    return segments_to_report


__all__ = [
    "analyze_video_cv",
    "calculate_avg_wide_rows",
    "find_continuous_segments",
    "merge_adjacent_segments",
    "normalize_waveform_type",
]
