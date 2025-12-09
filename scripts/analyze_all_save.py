#!/usr/bin/env python3
"""
assets/ultrasound-samples 폴더의 모든 영상을 분석하여
결과만 JSON으로 저장하는 스크립트.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.analyzer import analyze_video  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="폴더 내 모든 초음파 영상을 분석하고 결과를 JSON으로 저장")
    parser.add_argument(
        '--pattern',
        default='assets/ultrasound-samples/*.mp4',
        help='분석할 영상 글롭 패턴 (기본: assets/ultrasound-samples/*.mp4)',
    )
    parser.add_argument(
        '--output',
        default='waveform_batch_results.json',
        help='결과 JSON 저장 경로 (기본: waveform_batch_results.json)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_paths = sorted(Path().glob(args.pattern))

    if not video_paths:
        print("분석할 영상이 없습니다.")
        return

    results = []

    for video_path in video_paths:
        print(f"분석 시작: {video_path}")
        segments = analyze_video(str(video_path), verbose=False)
        print(f"  완료: {len(segments)}개 구간 감지")

        moving_segments = []
        for idx, seg in enumerate(segments, 1):
            for ms in seg.get('moving_subsegments', []):
                moving_segments.append({
                    'segment_number': idx,
                    'start_time': round(ms['start_time'], 2),
                    'end_time': round(ms['end_time'], 2),
                    'duration': round(ms['duration'], 2),
                    'type': seg.get('type', 'unknown'),
                })

        results.append({
            'video_path': str(video_path),
            'segment_count': len(segments),
            'moving_segment_count': len(moving_segments),
            'moving_segments': moving_segments,
            'segments': segments,
        })

    payload = {
        'generated_at': datetime.now().isoformat(),
        'video_count': len(results),
        'results': results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"결과 저장 완료: {output_path} (총 {len(results)}개 영상)")


if __name__ == '__main__':
    main()
