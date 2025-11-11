#!/usr/bin/env python3
"""
단일 초음파 영상 분석 스크립트

Usage:
    python scripts/analyze_single.py <video_path>
    python scripts/analyze_single.py  # 기본 샘플 영상 사용
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# analyzer 모듈의 analyze_video 함수 임포트
from src.detection.analyzer import analyze_video


def main():
    """메인 실행 함수"""
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 기본 샘플 영상 (경로 수정됨)
        video_path = 'assets/ultrasound-samples/28w-126bpm.mp4'

    analyze_video(video_path)


if __name__ == '__main__':
    main()
