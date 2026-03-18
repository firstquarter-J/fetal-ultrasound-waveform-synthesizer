from __future__ import annotations

from pathlib import Path
import unittest

from src.detection.analyzer import analyze_video as legacy_analyze_video
from src.detection.api import analyze_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = PROJECT_ROOT / "assets/ultrasound-samples/28w-126bpm.mp4"


class DetectionApiTests(unittest.TestCase):
    def test_public_api_matches_legacy_default_output(self) -> None:
        legacy_segments = legacy_analyze_video(str(SAMPLE_VIDEO), verbose=False)
        public_segments = analyze_video(SAMPLE_VIDEO, verbose=False)
        self.assertEqual(public_segments, legacy_segments)

    def test_public_api_returns_required_core_fields(self) -> None:
        segments = analyze_video(SAMPLE_VIDEO, verbose=False)

        self.assertGreaterEqual(len(segments), 1)
        for segment in segments:
            self.assertIn("start_time", segment)
            self.assertIn("end_time", segment)
            self.assertIn("type", segment)
            self.assertLessEqual(segment["start_time"], segment["end_time"])


if __name__ == "__main__":
    unittest.main()
