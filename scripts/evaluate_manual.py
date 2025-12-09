#!/usr/bin/env python3
"""
샘플 수동 구간과 자동 검출 결과를 비교하여
Precision/Recall(시간 커버리지 기준)을 출력하는 스크립트.

Usage:
  . venv/bin/activate && python scripts/evaluate_manual.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.analyzer import analyze_video  # noqa: E402

# 수동으로 표기한 파형 구간 (초 단위)
MANUAL_INTERVALS = {
    "8w-160bpm.mp4": [(0, 17), (20 * 60 + 30, 20 * 60 + 36)],
    "8w-165bpm.mp4": [(14, 21)],
    "12w-159bpm.mp4": [(30, 39)],
    "12w-161bpm.mp4": [(39, 50)],
    "12w-180bpm.mp4": [(38, 45)],
    "26w-141bpm.mp4": [(2 * 60 + 3, 2 * 60 + 13)],
    "27w-137bpm.mp4": [(1 * 60 + 57, 2 * 60 + 6), (4 * 60 + 0, 4 * 60 + 9)],
    "28w-126bpm.mp4": [(0, 15)],
    "34w-151bpm.mp4": [(1 * 60 + 44, 1 * 60 + 53), (2 * 60 + 4, 2 * 60 + 5), (1 * 60 + 28, 1 * 60 + 37)],
    "35w-141bpm.mp4": [(1 * 60 + 28, 1 * 60 + 37)],
}

SAMPLES_DIR = Path("assets/ultrasound-samples")


def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0.0, end - start)


def union_length(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def evaluate_video(video_path: Path):
    filename = video_path.name
    manual = MANUAL_INTERVALS.get(filename, [])
    detected = []
    segments = analyze_video(str(video_path), verbose=False)
    for seg in segments:
        detected.append((seg["start_time"], seg["end_time"]))

    manual_len = union_length(manual)
    detected_len = union_length(detected)

    overlap = 0.0
    for m in manual:
        for d in detected:
            overlap += interval_overlap(m, d)

    recall = overlap / manual_len if manual_len > 0 else 0.0
    precision = overlap / detected_len if detected_len > 0 else 0.0

    return {
        "filename": filename,
        "manual_intervals": manual,
        "detected_intervals": detected,
        "manual_total_sec": manual_len,
        "detected_total_sec": detected_len,
        "overlap_sec": overlap,
        "recall": recall,
        "precision": precision,
    }


def parse_args():
    p = argparse.ArgumentParser(description="수동 구간 대비 자동 검출 Precision/Recall 평가")
    p.add_argument(
        "--videos",
        nargs="*",
        help="평가할 파일명 리스트 (기본: 샘플 폴더 mp4 전체)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    targets = []
    if args.videos:
        for name in args.videos:
            path = (SAMPLES_DIR / name)
            if path.exists():
                targets.append(path)
    if not targets:
        targets = sorted(SAMPLES_DIR.glob("*.mp4"))

    results = []
    for video_path in targets:
        res = evaluate_video(video_path)
        results.append(res)
        print(f"{res['filename']}: recall {res['recall']:.2f}, precision {res['precision']:.2f}, "
              f"manual {res['manual_total_sec']:.1f}s, detected {res['detected_total_sec']:.1f}s")

    summary = {
        "videos": results,
        "avg_recall": sum(r["recall"] for r in results) / len(results),
        "avg_precision": sum(r["precision"] for r in results) / len(results),
    }

    Path("evaluation_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("결과 저장: evaluation_results.json")


if __name__ == "__main__":
    main()
