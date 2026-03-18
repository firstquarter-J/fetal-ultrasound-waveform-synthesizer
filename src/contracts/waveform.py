"""
Contracts for waveform detection outputs.

Public detection APIs should return JSON-serializable segment dicts that at
least satisfy the core keys defined in this module.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, NotRequired, TypeAlias, TypedDict


WaveformAnalysisMode: TypeAlias = Literal["timeseries", "cv", "vision", "opencv"]
WaveformSegmentPayload: TypeAlias = dict[str, Any]


class MovingSubsegment(TypedDict):
    start_time: float
    end_time: float
    duration: float
    type: NotRequired[str]


class WaveformSegment(TypedDict):
    start_time: float
    end_time: float
    type: str
    start_frame: NotRequired[int]
    end_frame: NotRequired[int]
    moving_subsegments: NotRequired[list[MovingSubsegment]]
    motion_analysis: NotRequired[dict[str, Any]]


def _coerce_float(mapping: Mapping[str, Any], field_name: str) -> float:
    value = mapping[field_name]
    return float(value)


def normalize_moving_subsegment(subsegment: Mapping[str, Any]) -> WaveformSegmentPayload:
    normalized = dict(subsegment)
    start_time = _coerce_float(subsegment, "start_time")
    end_time = _coerce_float(subsegment, "end_time")
    if end_time < start_time:
        raise ValueError(f"moving_subsegment end_time < start_time: {subsegment!r}")

    normalized["start_time"] = start_time
    normalized["end_time"] = end_time
    normalized["duration"] = float(subsegment.get("duration", end_time - start_time))
    if "type" in subsegment and subsegment["type"] is not None:
        normalized["type"] = str(subsegment["type"])
    return normalized


def normalize_waveform_segment(segment: Mapping[str, Any]) -> WaveformSegmentPayload:
    """
    Validate and normalize a detection segment while preserving extra fields.
    """
    normalized = dict(segment)
    start_time = _coerce_float(segment, "start_time")
    end_time = _coerce_float(segment, "end_time")
    if end_time < start_time:
        raise ValueError(f"segment end_time < start_time: {segment!r}")

    normalized["start_time"] = start_time
    normalized["end_time"] = end_time
    normalized["type"] = str(segment.get("type", "unknown"))

    if "start_frame" in segment and segment["start_frame"] is not None:
        normalized["start_frame"] = int(segment["start_frame"])
    if "end_frame" in segment and segment["end_frame"] is not None:
        normalized["end_frame"] = int(segment["end_frame"])

    moving_subsegments = segment.get("moving_subsegments")
    if moving_subsegments is not None:
        normalized["moving_subsegments"] = [
            normalize_moving_subsegment(item)
            for item in moving_subsegments
        ]

    return normalized


def normalize_waveform_segments(
    segments: Iterable[Mapping[str, Any]],
) -> list[WaveformSegmentPayload]:
    return [normalize_waveform_segment(segment) for segment in segments]
