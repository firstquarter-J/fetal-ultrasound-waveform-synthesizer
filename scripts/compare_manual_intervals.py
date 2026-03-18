#!/usr/bin/env python3
"""
현재 timeseries 기반 검출 결과를 annotations/manual_waveform_intervals.md와 비교해
manual 파일과 동일한 bullet 형식으로 비교 리포트를 기록합니다.

Outputs:
  - results/manual_waveform_intervals_comparison.md
  - results/manual_interval_diffs.json

Usage:
  ./venv/bin/python scripts/compare_manual_intervals.py
  ./venv/bin/python scripts/compare_manual_intervals.py --videos 8w-165bpm.mp4 12w-159bpm.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.api import analyze_video  # noqa: E402
from src.common.manual_intervals import parse_time_to_seconds  # noqa: E402


SAMPLES_DIR = PROJECT_ROOT / "assets/ultrasound-samples"
MANUAL_MD = PROJECT_ROOT / "annotations/manual_waveform_intervals.md"
OUT_DIR = PROJECT_ROOT / "results"
OUT_MD = OUT_DIR / "manual_waveform_intervals_comparison.md"
OUT_JSON = OUT_DIR / "manual_interval_diffs.json"


def fmt_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 0:
        return f"-{fmt_seconds(-sec)}"
    m = int(sec // 60)
    s = sec - m * 60
    if m > 0:
        return f"{m}m{s:05.2f}s"
    return f"{s:.2f}s"


@dataclass(frozen=True)
class ParsedBulletLine:
    raw_line: str
    filename: str
    manual_indices: List[int]


def parse_manual_md_with_lines(md_text: str) -> Tuple[Dict[str, List[Tuple[float, float]]], List[str], List[ParsedBulletLine]]:
    manual_map: Dict[str, List[Tuple[float, float]]] = {}
    raw_lines: List[str] = md_text.splitlines()
    parsed_lines: List[ParsedBulletLine] = []

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        content = line[2:]
        if ":" not in content:
            continue
        filename, rest = content.split(":", 1)
        filename = filename.strip()

        rest_no_parens = re.sub(r"\(.*?\)", "", rest)
        matches = re.findall(
            r"([0-9]+m[0-9]+s|[0-9]+s)\s*~\s*([0-9]+m[0-9]+s|[0-9]+s)",
            rest_no_parens,
        )
        if not matches:
            continue

        out = manual_map.setdefault(filename, [])
        indices: List[int] = []
        for a, b in matches:
            start = parse_time_to_seconds(a)
            end = parse_time_to_seconds(b)
            if end <= start:
                continue
            indices.append(len(out))
            out.append((start, end))

        parsed_lines.append(ParsedBulletLine(raw_line=raw_line, filename=filename, manual_indices=indices))

    return manual_map, raw_lines, parsed_lines


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


@dataclass(frozen=True)
class Match:
    manual_idx: int
    detected_idx: int
    overlap_sec: float


def match_intervals(
    manual: List[Tuple[float, float]],
    detected: List[Tuple[float, float]],
) -> Tuple[List[Match], List[int], List[int]]:
    if not manual or not detected:
        return [], list(range(len(manual))), list(range(len(detected)))

    overlaps = []
    for i, m in enumerate(manual):
        for j, d in enumerate(detected):
            ov = interval_overlap(m, d)
            overlaps.append((ov, i, j))
    overlaps.sort(reverse=True, key=lambda x: x[0])

    used_m = set()
    used_d = set()
    matches: List[Match] = []
    for ov, i, j in overlaps:
        if ov <= 0:
            break
        if i in used_m or j in used_d:
            continue
        used_m.add(i)
        used_d.add(j)
        matches.append(Match(manual_idx=i, detected_idx=j, overlap_sec=float(ov)))

    unmatched_m = [i for i in range(len(manual)) if i not in used_m]
    unmatched_d = [j for j in range(len(detected)) if j not in used_d]
    return matches, unmatched_m, unmatched_d


def analyze_detected(video_path: Path) -> List[Tuple[float, float]]:
    segs = analyze_video(str(video_path), verbose=False)
    out = []
    for s in segs:
        out.append((float(s["start_time"]), float(s["end_time"])))
    return sorted(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="annotations/manual_waveform_intervals.md 대비 자동 검출 구간 차이 기록")
    p.add_argument(
        "--videos",
        nargs="*",
        help="비교할 파일명 리스트(기본: manual md에 등장하는 전부)",
    )
    return p.parse_args()


def main() -> None:
    manual_text = MANUAL_MD.read_text(encoding="utf-8")
    manual_map, raw_lines, parsed_bullets = parse_manual_md_with_lines(manual_text)

    args = parse_args()
    targets = []
    if args.videos:
        targets = [name for name in args.videos if name in manual_map]
    if not targets:
        targets = sorted(manual_map.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_from": str(MANUAL_MD),
        "videos": [],
        "overall": {},
    }

    total_manual = 0.0
    total_detected = 0.0
    total_overlap = 0.0

    per_video_render: Dict[str, Dict[str, object]] = {}
    detected_only_lines: List[str] = []

    for name in targets:
        video_path = SAMPLES_DIR / name
        if not video_path.exists():
            continue

        manual = manual_map.get(name, [])
        detected = analyze_detected(video_path)

        matches, unmatched_m, unmatched_d = match_intervals(manual, detected)
        manual_to_match: Dict[int, Match] = {m.manual_idx: m for m in matches}

        manual_len = union_length(manual)
        detected_len = union_length(detected)
        overlap_len = 0.0
        for m in manual:
            for d in detected:
                overlap_len += interval_overlap(m, d)

        total_manual += manual_len
        total_detected += detected_len
        total_overlap += overlap_len

        recall = overlap_len / manual_len if manual_len > 0 else 0.0
        precision = overlap_len / detected_len if detected_len > 0 else 0.0

        per_video = {
            "filename": name,
            "manual_intervals": manual,
            "detected_intervals": detected,
            "manual_total_sec": manual_len,
            "detected_total_sec": detected_len,
            "overlap_sec": overlap_len,
            "recall": recall,
            "precision": precision,
            "matches": [],
            "unmatched_manual": unmatched_m,
            "unmatched_detected": unmatched_d,
        }

        for m in matches:
            m_iv = manual[m.manual_idx]
            d_iv = detected[m.detected_idx]
            ds = d_iv[0] - m_iv[0]
            de = d_iv[1] - m_iv[1]
            m_len = max(0.0, m_iv[1] - m_iv[0])
            d_len = max(0.0, d_iv[1] - d_iv[0])
            denom = (m_len + d_len - m.overlap_sec)
            iou = (m.overlap_sec / denom) if denom > 1e-9 else 0.0
            per_video["matches"].append(
                {
                    "manual_idx": m.manual_idx,
                    "detected_idx": m.detected_idx,
                    "manual": m_iv,
                    "detected": d_iv,
                    "delta_start_sec": ds,
                    "delta_end_sec": de,
                    "overlap_sec": m.overlap_sec,
                    "iou": iou,
                }
            )
        report["videos"].append(per_video)

        # detected-only(매칭 실패) 구간도 bullet 형식으로 별도 기록
        for j in unmatched_d:
            s, e = detected[j]
            detected_only_lines.append(f"- {name}: {fmt_seconds(s)} ~ {fmt_seconds(e)}")

        per_video_render[name] = {
            "manual": manual,
            "detected": detected,
            "manual_to_match": manual_to_match,
        }

    overall_recall = total_overlap / total_manual if total_manual > 0 else 0.0
    overall_precision = total_overlap / total_detected if total_detected > 0 else 0.0
    report["overall"] = {
        "manual_total_sec": total_manual,
        "detected_total_sec": total_detected,
        "overlap_sec": total_overlap,
        "recall": overall_recall,
        "precision": overall_precision,
    }

    # manual 파일과 동일한 bullet 형식 유지: 기존 라인에 detected 정보만 덧붙임
    bullet_lookup: Dict[str, List[ParsedBulletLine]] = {}
    for pb in parsed_bullets:
        bullet_lookup.setdefault(pb.filename, []).append(pb)

    # 빠른 조회용: (raw_line) -> ParsedBulletLine
    raw_to_pb: Dict[str, ParsedBulletLine] = {pb.raw_line: pb for pb in parsed_bullets}

    out_lines: List[str] = []
    for raw_line in raw_lines:
        pb = raw_to_pb.get(raw_line)
        if pb is None or pb.filename not in per_video_render:
            out_lines.append(raw_line)
            continue

        render = per_video_render[pb.filename]
        manual_list: List[Tuple[float, float]] = render["manual"]  # type: ignore[assignment]
        detected_list: List[Tuple[float, float]] = render["detected"]  # type: ignore[assignment]
        manual_to_match = render["manual_to_match"]  # type: ignore[assignment]

        pieces: List[str] = []
        for manual_idx in pb.manual_indices:
            if manual_idx < 0 or manual_idx >= len(manual_list):
                continue
            m_iv = manual_list[manual_idx]
            m_match = manual_to_match.get(manual_idx)
            if m_match is None:
                pieces.append(f"detected MISS (manual {fmt_seconds(m_iv[0])} ~ {fmt_seconds(m_iv[1])})")
                continue
            d_iv = detected_list[m_match.detected_idx]
            ds = d_iv[0] - m_iv[0]
            de = d_iv[1] - m_iv[1]
            pieces.append(
                f"detected {fmt_seconds(d_iv[0])} ~ {fmt_seconds(d_iv[1])} (Δstart {ds:+.2f}s, Δend {de:+.2f}s)"
            )

        suffix = ""
        if pieces:
            suffix = " | " + "; ".join(pieces)

        out_lines.append(raw_line + suffix)

    if detected_only_lines:
        out_lines.append("")
        out_lines.append("Detected-only intervals (not in manual):")
        out_lines.extend(detected_only_lines)

    out_lines.append("")
    out_lines.append(
        f"Overall (manual_total={total_manual:.2f}s, detected_total={total_detected:.2f}s, "
        f"overlap={total_overlap:.2f}s, recall={overall_recall:.3f}, precision={overall_precision:.3f})"
    )

    OUT_MD.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {OUT_MD}")
    print(f"wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
