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

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        bottom_region = frame[int(height * 0.7):, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        row_counts = np.count_nonzero(thresh, axis=1)
        wide_rows = np.sum(row_counts > width * 0.5)
        wide_rows_list.append(wide_rows)

    cap.release()
    return np.mean(wide_rows_list) if wide_rows_list else 0


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
    bottom_region = frame[int(height * 0.7):, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(thresh)

    if white_pixel_count > 40000:
        # 검증 1: 수평 분포 (도플러 파형은 가로로 넓게 분포)
        row_counts = np.count_nonzero(thresh, axis=1)
        wide_rows = np.sum(row_counts > width * 0.5)

        # 검증 2: 좌측 영역 픽셀 (도플러 파형은 화면 왼쪽에서 시작)
        left_region = thresh[:, :int(width * 0.3)]
        left_pixels = np.count_nonzero(left_region)
        left_ratio = (left_pixels / white_pixel_count * 100) if white_pixel_count > 0 else 0

        # Gray Doppler 조건
        if wide_rows >= 2 and left_pixels > 5000 and 10 < left_ratio < 60:
            return True, [('gray', 0, int(height * 0.7), width, int(height * 0.3))], white_pixel_count

    # ========================================
    # Type 3: Orange Fragmented
    # ========================================
    bottom_region_orange = orange_mask[int(height * 0.5):, :]
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
            return True, [('orange_fragmented', 0, int(height * 0.5), width, int(height * 0.5))], orange_pixel_count

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
    current_segment = {
        'start_frame': waveform_frames[0]['frame'],
        'start_time': waveform_frames[0]['timestamp'],
        'end_frame': waveform_frames[0]['frame'],
        'end_time': waveform_frames[0]['timestamp'],
        'type': waveform_frames[0].get('type')
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
        else:
            segments.append(current_segment)
            current_segment = {
                'start_frame': waveform_frames[i]['frame'],
                'start_time': waveform_frames[i]['timestamp'],
                'end_frame': waveform_frames[i]['frame'],
                'end_time': waveform_frames[i]['timestamp'],
                'type': waveform_frames[i].get('type')
            }

    segments.append(current_segment)
    return segments


# ============================================================================
# 메인 분석 함수
# ============================================================================

def analyze_video(video_path):
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

    duration = (total_frames / fps) if fps else 0

    print(f"\n영상: {video_path}")
    print(f"  FPS: {fps}, 총 프레임: {total_frames}, 길이: {duration:.1f}초")

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
                'type': region_type
            })

        frame_idx += 1

        if total_frames and frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  진행: {frame_idx}/{total_frames} ({pct:.1f}%)")

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
        duration = seg['end_time'] - seg['start_time']

        # 필터 1: 지속 시간 (3초 미만 제거)
        if duration < 3.0:
            filtered_short_segments.append(seg)
            continue

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

        segment_type = seg_frames[0].get('type')

        # 필터 2: Gray 타입 false positive 제거
        if segment_type == 'gray':
            avg_wide_rows = calculate_avg_wide_rows(video_path, seg, fps)

            # 필터 2-1: 화면 전체가 밝은 정적 화면
            # 예: 8w-165bpm false positive (wide_rows 133~138)
            if avg_wide_rows > 100:
                continue

            # 필터 2-2: 고밝기 + 좁은 분포 = false positive
            # 실제 파형: 26w-141bpm(avg=141K, wide=70.7), 34w-151bpm(avg=108K, wide=58.1)
            # False positive: 12w-161bpm seg1(avg=115K, wide=9.1), 12w-180bpm seg2-7(avg=117~166K, wide=6.7~37.9)
            if avg_count > 115000 and avg_wide_rows < 39:
                continue

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

    # ========================================
    # 결과 출력
    # ========================================
    print(f"\n  파형 감지 구간: {len(valid_segments)}개")
    for i, seg in enumerate(valid_segments, 1):
        duration = seg['end_time'] - seg['start_time']
        seg_type = seg.get('type', 'unknown')
        print(f"    구간 {i}: {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 ({duration:.2f}초) [type: {seg_type}]")

    if filtered_short_segments:
        print(f"\n  짧은 구간 (3초 미만, 필터링됨): {len(filtered_short_segments)}개")
        for i, seg in enumerate(filtered_short_segments, 1):
            duration = seg['end_time'] - seg['start_time']
            seg_type = seg.get('type', 'unknown')
            min_sec = int(seg['start_time'] // 60)
            start_sec = seg['start_time'] % 60
            print(f"    {i}. {seg['start_time']:.2f}초 ~ {seg['end_time']:.2f}초 ({duration:.2f}초) [{min_sec}분 {start_sec:.1f}초, type: {seg_type}]")

    return valid_segments


# ============================================================================
# 메인 엔트리 포인트
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'assets/heartbeat-samples/28w-126bpm.mp4'

    analyze_video(video_path)
