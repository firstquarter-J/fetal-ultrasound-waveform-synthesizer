"""
Utilities for loading manual waveform interval annotations from markdown.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple


def parse_time_to_seconds(token: str) -> float:
    token = token.strip()
    match = re.fullmatch(r"(?:(\d+)m)?(?:(\d+)s)?", token)
    if not match:
        raise ValueError(f"시간 토큰 파싱 실패: {token!r}")
    minutes = int(match.group(1) or 0)
    seconds = int(match.group(2) or 0)
    return float(minutes * 60 + seconds)


def parse_manual_intervals_md(md_text: str) -> Dict[str, List[Tuple[float, float]]]:
    intervals: Dict[str, List[Tuple[float, float]]] = {}

    for raw_line in md_text.splitlines():
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

        out = intervals.setdefault(filename, [])
        for start_token, end_token in matches:
            start = parse_time_to_seconds(start_token)
            end = parse_time_to_seconds(end_token)
            if end > start:
                out.append((start, end))

    return intervals


def load_manual_intervals(md_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    return parse_manual_intervals_md(md_path.read_text(encoding="utf-8"))
