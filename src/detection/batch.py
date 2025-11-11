#!/usr/bin/env python3
"""
전체 영상 배치 분석 및 결과 저장
"""

import cv2
import numpy as np
import json
import glob
import os
from datetime import datetime


def calculate_avg_wide_rows(video_path, segment, fps):
    """구간의 평균 wide_rows 계산"""
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


def detect_waveform_region(frame):
    """파형 감지 (analyze_single_video.py와 동일)"""
    height, width = frame.shape[:2]

    # 타입 1: 오렌지색 파형
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([40, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((3, 3), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    # 방법 1: 큰 연속된 컨투어 찾기
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

    # 타입 2: 회색 도플러 파형
    bottom_region = frame[int(height * 0.7):, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(thresh)

    if white_pixel_count > 40000:
        row_counts = np.count_nonzero(thresh, axis=1)
        wide_rows = np.sum(row_counts > width * 0.5)

        left_region = thresh[:, :int(width * 0.3)]
        left_pixels = np.count_nonzero(left_region)
        left_ratio = (left_pixels / white_pixel_count * 100) if white_pixel_count > 0 else 0

        if wide_rows >= 2 and left_pixels > 5000 and 10 < left_ratio < 60:
            return True, [('gray', 0, int(height * 0.7), width, int(height * 0.3))], white_pixel_count

    # 타입 3: Fragmented 오렌지 파형
    bottom_region_orange = orange_mask[int(height * 0.5):, :]
    orange_pixel_count = cv2.countNonZero(bottom_region_orange)

    if 5000 < orange_pixel_count < 80000:
        row_counts_orange = np.count_nonzero(bottom_region_orange, axis=1)
        rows_with_pixels = np.sum(row_counts_orange > 10)

        left_region_orange = bottom_region_orange[:, :int(width * 0.3)]
        left_orange_pixels = cv2.countNonZero(left_region_orange)
        left_orange_ratio = (left_orange_pixels / orange_pixel_count * 100) if orange_pixel_count > 0 else 0

        if rows_with_pixels > 20 and left_orange_pixels > 1000 and 10 < left_orange_ratio < 50:
            return True, [('orange_fragmented', 0, int(height * 0.5), width, int(height * 0.5))], orange_pixel_count

    return False, [], 0


def normalize_waveform_type(waveform_type):
    """파형 타입 정규화 - orange 계열은 모두 orange로 통일"""
    if waveform_type in ['orange', 'orange_fragmented']:
        return 'orange'
    return waveform_type


def find_continuous_segments(waveform_frames, fps, gap_threshold=0.5):
    """연속 구간 찾기"""
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

        # 타입 정규화 후 비교 (orange 계열은 같은 타입으로 취급)
        normalized_prev = normalize_waveform_type(prev_type)
        normalized_curr = normalize_waveform_type(curr_type)

        # 시간상 연속이고 정규화된 타입이 동일하면 같은 구간으로 병합
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


def analyze_video(video_path):
    """영상 분석"""
    cap = cv2.VideoCapture(video_path)

    # 영상 열기 실패 검사
    if not cap.isOpened(): 
        print(f"  오류: 영상 열기 실패")
        return [], 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FPS 유효성 검사 (0, NaN, 매우 작은 값)
    if not fps or np.isnan(fps) or fps <= 1e-6:
        print("  경고: FPS 미확인 → 30.0으로 대체")
        fps = 30.0

    video_duration = (total_frames / fps) if fps else 0

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

    cap.release()

    # 연속 구간 찾기
    segments = find_continuous_segments(waveform_frames, fps)

    # 동적 변화 체크 및 길이 필터링
    valid_segments = []
    for seg in segments:
        seg_duration = seg['end_time'] - seg['start_time']
        if seg_duration < 3.0:
            continue

        # 구간 내 픽셀 수 변동성 체크
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

        # Gray 타입 필터링: 조합 조건으로 false positive 제거
        if segment_type == 'gray':
            avg_wide_rows = calculate_avg_wide_rows(video_path, seg, fps)

            # 필터 1: wide_rows > 100 (화면 전체가 밝은 정적 화면)
            # 8w-165bpm false positive: wide_rows 133~138
            if avg_wide_rows > 100:
                continue

            # 필터 2: 고밝기(>115K) + 좁은 분포(<39) = false positive
            # 실제 파형: 26w-141bpm(avg=141K, wide=70.7), 34w-151bpm(avg=108K, wide=58.1)
            # False positive: 12w-161bpm seg1(avg=115K, wide=9.1), 12w-180bpm seg2-7(avg=117~166K, wide=6.7~37.9)
            if avg_count > 115000 and avg_wide_rows < 39:
                continue

        # 평균 픽셀 수에 따른 변동 비율 기준
        if avg_count <= 150000:
            valid_segments.append(seg)
        elif avg_count <= 180000:
            if variation_ratio > 0.09:
                valid_segments.append(seg)
        else:
            if variation_ratio > 0.25:
                valid_segments.append(seg)

    return valid_segments, fps, total_frames, video_duration


def format_time(seconds):
    """초를 MM:SS 형식으로 변환"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def main():
    """메인 함수"""
    video_pattern = 'assets/heartbeat-samples/*w-*bpm.mp4'
    video_files = sorted(glob.glob(video_pattern))

    if not video_files:
        print("영상 파일을 찾을 수 없습니다.")
        return

    results = []

    print(f"총 {len(video_files)}개 영상 분석 시작...\n")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"분석 중: {filename}")

        try:
            segments, fps, total_frames, duration = analyze_video(video_path)

            result = {
                'filename': filename,
                'video_info': {
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration_seconds': round(duration, 2)
                },
                'detected_segments': len(segments),
                'segments': []
            }

            for i, seg in enumerate(segments, 1):
                seg_duration = seg['end_time'] - seg['start_time']
                result['segments'].append({
                    'segment_number': i,
                    'start_time_seconds': round(seg['start_time'], 2),
                    'end_time_seconds': round(seg['end_time'], 2),
                    'duration_seconds': round(seg_duration, 2),
                    'start_time_formatted': format_time(seg['start_time']),
                    'end_time_formatted': format_time(seg['end_time']),
                    'type': seg.get('type', 'unknown')
                })

            results.append(result)
            print(f"  → {len(segments)}개 구간 감지")

        except Exception as e:
            print(f"  → 오류: {str(e)}")
            results.append({
                'filename': filename,
                'error': str(e)
            })

    # 결과 저장
    output_file = 'waveform_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'total_videos': len(video_files),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n분석 완료! 결과 저장: {output_file}")

    # 요약 출력
    print("\n=== 분석 요약 ===")
    for result in results:
        if 'error' in result:
            print(f"{result['filename']}: 오류 발생")
        else:
            print(f"{result['filename']}: {result['detected_segments']}개 구간")
            for seg in result['segments']:
                print(f"  구간 {seg['segment_number']}: {seg['start_time_formatted']} ~ {seg['end_time_formatted']} ({seg['duration_seconds']}초, {seg['type']})")


if __name__ == '__main__':
    main()
