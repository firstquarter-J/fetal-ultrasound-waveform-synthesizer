#!/usr/bin/env python3
"""
시계열(timeseries) 기반 파형 검출 파라미터를 간단한 그리드 서치로 튜닝합니다.

특징(orange/white/black/top-black) 시계열을 한 번만 추출한 뒤,
임계값/게이트 조합을 바꿔가며 수동 구간 대비 Precision/Recall을 비교합니다.

Usage:
  ./venv/bin/python scripts/tune_timeseries.py
  ./venv/bin/python scripts/tune_timeseries.py --top 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.detection.analyzer as an  # noqa: E402


SAMPLES_DIR = Path("assets/ultrasound-samples")

# scripts/evaluate_manual.py와 동일한 수동 구간(초 단위)
MANUAL_INTERVALS: Dict[str, List[Tuple[float, float]]] = {
    "8w-160bpm.mp4": [(0, 17), (20 * 60 + 30, 20 * 60 + 36)],
    "8w-165bpm.mp4": [(14, 21)],
    "12w-159bpm.mp4": [(30, 39)],
    "12w-161bpm.mp4": [(39, 50)],
    "12w-180bpm.mp4": [(38, 45)],
    "26w-141bpm.mp4": [(2 * 60 + 3, 2 * 60 + 13)],
    "27w-137bpm.mp4": [(1 * 60 + 57, 2 * 60 + 6), (4 * 60 + 0, 4 * 60 + 9)],
    "28w-126bpm.mp4": [(0, 15)],
    "34w-151bpm.mp4": [
        (1 * 60 + 44, 1 * 60 + 53),
        (2 * 60 + 4, 2 * 60 + 5),
    ],
    "35w-141bpm.mp4": [(1 * 60 + 28, 1 * 60 + 37)],
}


def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0.0, end - start)


def union_length(intervals: Sequence[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return float(sum(e - s for s, e in merged))


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def hysteresis_mask(series: np.ndarray, on_threshold: float, off_threshold: float) -> np.ndarray:
    active = False
    out = np.zeros(series.size, dtype=bool)
    for i, v in enumerate(series):
        v = float(v)
        if not active:
            if v >= on_threshold:
                active = True
        else:
            if v <= off_threshold:
                active = False
        out[i] = active
    return out


def find_binary_segments(
    times: np.ndarray,
    mask: np.ndarray,
    min_duration_sec: float,
    merge_gap_sec: float,
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    active = False
    start: float | None = None
    for ts, val in zip(times, mask):
        ts = float(ts)
        if not active and bool(val):
            active = True
            start = ts
        elif active and not bool(val):
            end = ts
            if start is not None and end - start >= min_duration_sec:
                segments.append((float(start), float(end)))
            active = False
            start = None

    if active and start is not None:
        end = float(times[-1])
        if end - start >= min_duration_sec:
            segments.append((float(start), float(end)))

    if not segments:
        return []

    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]


def refine_segment_bounds(
    times_arr: np.ndarray,
    raw_series: np.ndarray,
    start: float,
    end: float,
    threshold: float,
) -> Tuple[float, float]:
    idx = np.where((times_arr >= start) & (times_arr <= end) & (raw_series >= threshold))[0]
    if idx.size == 0:
        return float(start), float(end)
    return float(times_arr[int(idx[0])]), float(times_arr[int(idx[-1])])


@dataclass(frozen=True)
class Features:
    fps: float
    video_duration: float
    times: np.ndarray
    black: np.ndarray
    top_black: np.ndarray
    orange_raw: np.ndarray
    white_raw: np.ndarray


def extract_features(video_path: Path) -> Features:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or np.isnan(fps) or fps <= 1e-6:
        fps = 30.0

    video_duration = (total_frames / fps) if fps else 0.0

    frame_idx = 0
    times: List[float] = []
    black: List[float] = []
    top_black: List[float] = []
    orange: List[float] = []
    white: List[float] = []

    while True:
        if frame_idx % an.TS_STRIDE_FRAMES != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ok, frame = cap.read()
        if not ok:
            break

        br, o, w, tb = an._extract_timeseries_features_fixed_roi(frame)
        times.append(frame_idx / fps)
        black.append(br)
        top_black.append(tb)
        orange.append(o)
        white.append(w)
        frame_idx += 1

    cap.release()

    return Features(
        fps=float(fps),
        video_duration=float(video_duration),
        times=np.array(times, dtype=np.float64),
        black=np.array(black, dtype=np.float64),
        top_black=np.array(top_black, dtype=np.float64),
        orange_raw=np.array(orange, dtype=np.float64),
        white_raw=np.array(white, dtype=np.float64),
    )


@dataclass(frozen=True)
class Params:
    orange_on: float
    orange_off: float
    white_on: float
    white_off: float
    top_black_gate_min: float
    short_orange_mean_min: float


def detect_segments(features: Features, p: Params) -> List[Tuple[float, float]]:
    if features.times.size == 0:
        return []

    orange_arr = smooth_1d(features.orange_raw, an.TS_SMOOTH_WINDOW)
    white_arr = smooth_1d(features.white_raw, an.TS_SMOOTH_WINDOW)

    orange_mask = hysteresis_mask(orange_arr, p.orange_on, p.orange_off)

    white_for_detect = np.where(orange_arr >= an.TS_ORANGE_SUPPRESS_WHITE_THRESHOLD, 0.0, white_arr)
    white_mask = hysteresis_mask(white_for_detect, p.white_on, p.white_off)

    orange_segments = find_binary_segments(features.times, orange_mask, an.TS_ORANGE_MIN_SEGMENT_SEC, an.TS_MERGE_GAP_SEC)
    white_segments = find_binary_segments(features.times, white_mask, an.TS_WHITE_MIN_SEGMENT_SEC, an.TS_MERGE_GAP_SEC)

    merged = sorted(orange_segments + white_segments)
    if not merged:
        return []

    segments: List[List[float]] = []
    for s, e in merged:
        if not segments:
            segments.append([float(s), float(e)])
            continue
        if s - segments[-1][1] <= an.TS_MERGE_GAP_SEC:
            segments[-1][1] = max(segments[-1][1], float(e))
        else:
            segments.append([float(s), float(e)])

    out: List[Tuple[float, float]] = []
    for start, end in segments:
        start = max(0.0, float(start))
        end = min(float(features.video_duration), float(end))
        duration = end - start
        if duration < an.TS_WHITE_MIN_SEGMENT_SEC or duration > an.TS_MAX_SEGMENT_SEC:
            continue

        idx = (features.times >= start) & (features.times <= end)
        if not np.any(idx):
            continue

        mean_black_ratio = float(features.black[idx].mean())
        peak_top_black = float(features.top_black[idx].max())
        peak_orange = float(orange_arr[idx].max())
        peak_white = float(white_arr[idx].max())
        mean_orange = float(orange_arr[idx].mean())

        if peak_top_black < p.top_black_gate_min:
            continue

        if duration <= an.TS_SHORT_SEGMENT_SEC:
            if mean_black_ratio < an.TS_SHORT_BLACK_RATIO_MIN:
                continue
            if peak_orange < an.TS_SHORT_ORANGE_PEAK_MIN and peak_white < an.TS_SHORT_WHITE_PEAK_MIN:
                continue
            if peak_orange >= peak_white and mean_orange < p.short_orange_mean_min:
                continue

        refined_start, refined_end = refine_segment_bounds(
            features.times,
            np.maximum(features.orange_raw, features.white_raw),
            start,
            end,
            min(p.orange_off, p.white_off),
        )
        if refined_end - refined_start < an.TS_WHITE_MIN_SEGMENT_SEC:
            continue
        out.append((float(refined_start), float(refined_end)))

    return out


def evaluate(manual: Sequence[Tuple[float, float]], detected: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    manual_len = union_length(manual)
    detected_len = union_length(detected)

    overlap = 0.0
    for m in manual:
        for d in detected:
            overlap += interval_overlap(m, d)

    recall = overlap / manual_len if manual_len > 0 else 0.0
    precision = overlap / detected_len if detected_len > 0 else 0.0
    return float(recall), float(precision)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="timeseries 파형 검출 파라미터 튜닝(그리드 서치)")
    p.add_argument("--top", type=int, default=10, help="상위 결과 출력 개수")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    targets = [SAMPLES_DIR / name for name in MANUAL_INTERVALS.keys()]
    features_by_name: Dict[str, Features] = {}
    for vp in targets:
        features_by_name[vp.name] = extract_features(vp)

    baseline = Params(
        orange_on=an.TS_ORANGE_MIN_ON,
        orange_off=an.TS_ORANGE_MIN_OFF,
        white_on=an.TS_WHITE_MIN_ON,
        white_off=an.TS_WHITE_MIN_OFF,
        top_black_gate_min=an.TS_TOP_BLACK_GATE_MIN,
        short_orange_mean_min=getattr(an, "TS_SHORT_ORANGE_MEAN_MIN", 0.08),
    )

    def score_params(p: Params) -> Tuple[float, float]:
        recalls = []
        precisions = []
        for name, feats in features_by_name.items():
            manual = MANUAL_INTERVALS.get(name, [])
            detected = detect_segments(feats, p)
            r, pr = evaluate(manual, detected)
            recalls.append(r)
            precisions.append(pr)
        return float(np.mean(recalls)), float(np.mean(precisions))

    base_r, base_pr = score_params(baseline)
    print(f"baseline avg_recall={base_r:.4f} avg_precision={base_pr:.4f} ({baseline})")

    grid = {
        "orange_on": [0.030, 0.035, 0.040, 0.045],
        "orange_off": [0.020, 0.022, 0.025, 0.030],
        "white_on": [0.045, 0.050, 0.055],
        "white_off": [0.025, 0.030, 0.035],
        "top_black_gate_min": [0.75, 0.80, 0.85],
        "short_orange_mean_min": [0.06, 0.08, 0.10, 0.12],
    }

    results: List[Tuple[float, float, Params]] = []
    for orange_on, orange_off, white_on, white_off, top_black_gate_min, short_orange_mean_min in product(
        grid["orange_on"],
        grid["orange_off"],
        grid["white_on"],
        grid["white_off"],
        grid["top_black_gate_min"],
        grid["short_orange_mean_min"],
    ):
        if orange_off >= orange_on or white_off >= white_on:
            continue
        p = Params(
            orange_on=float(orange_on),
            orange_off=float(orange_off),
            white_on=float(white_on),
            white_off=float(white_off),
            top_black_gate_min=float(top_black_gate_min),
            short_orange_mean_min=float(short_orange_mean_min),
        )
        r, pr = score_params(p)
        results.append((r, pr, p))

    min_recall = base_r - 0.01
    results.sort(key=lambda x: (x[1], x[0]), reverse=True)

    print(f"\nTop candidates (require avg_recall >= {min_recall:.4f}):")
    printed = 0
    for r, pr, p in results:
        if r < min_recall:
            continue
        print(f"avg_recall={r:.4f} avg_precision={pr:.4f} {p}")
        printed += 1
        if printed >= int(args.top):
            break


if __name__ == "__main__":
    main()
