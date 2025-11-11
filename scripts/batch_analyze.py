#!/usr/bin/env python3
"""
배치 초음파 영상 분석 스크립트

assets/ultrasound-samples/ 폴더의 모든 MP4 파일을 일괄 분석하고
결과를 JSON 파일로 저장합니다.

Usage:
    python scripts/batch_analyze.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# batch 모듈의 main 함수 임포트
from src.detection.batch import main


if __name__ == '__main__':
    main()
