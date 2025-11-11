# 태아 초음파 파형 합성기 (Fetal Ultrasound Waveform Synthesizer)

태아 초음파 영상에서 심박 파형을 자동으로 검출하고, 심박음을 합성하는 오픈소스 프로젝트입니다.

## 📋 프로젝트 소개

이 프로젝트는 MP4 형식의 태아 초음파 영상을 분석하여:

1. **파형 검출**: 프레임별로 심박 파형이 존재하는 구간을 자동으로 탐지
2. **파형 분류**: Orange, Gray, Fragmented 등 파형 타입 자동 분류
3. **심박음 합성** (개발 예정): 검출된 파형 구간에 심박음 오디오 합성

## ✨ 주요 기능

### 현재 구현된 기능

- ✅ **자동 파형 검출**: 컴퓨터 비전 기반 심박 파형 영역 인식
- ✅ **파형 타입 분류**:
  - `orange`: 주황색 파형
  - `gray`: 회색 파형
  - `orange_fragmented`: 단편화된 주황색 파형
- ✅ **프레임 단위 분석**: 각 프레임의 파형 존재 여부 및 위치 정보
- ✅ **배치 처리**: 여러 영상을 한 번에 분석

### 개발 예정

- 🔲 심박음 합성 엔진
- 🔲 실시간 처리 최적화
- 🔲 다양한 출력 포맷 지원

## 🚀 설치 방법

### 요구사항

- Python 3.11 이상
- OpenCV
- NumPy

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/fetal-ultrasound-waveform-synthesizer.git
cd fetal-ultrasound-waveform-synthesizer

# 의존성 설치
pip3 install -r requirements.txt
```

## 💻 사용 방법

### 단일 영상 분석

```bash
# macOS/Linux
python3 scripts/analyze_single.py

# 특정 영상 분석
python3 scripts/analyze_single.py path/to/video.mp4

# Windows
python scripts/analyze_single.py path/to/video.mp4
```

### 배치 분석

```bash
# macOS/Linux
python3 scripts/batch_analyze.py

# Windows
python scripts/batch_analyze.py

# 결과는 waveform_analysis_results.json 파일로 저장됩니다
```

### 출력 예시

```json
{
  "video_path": "assets/ultrasound-samples/12w-159bpm.mp4",
  "waveform_type": "orange",
  "total_frames": 450,
  "waveform_frames": 380,
  "waveform_percentage": 84.4,
  "frame_details": [...]
}
```

## 📊 샘플 데이터

프로젝트에는 실제 태아 초음파 샘플 영상이 포함되어 있습니다:

| 임신 주차 | BPM     | 파일명                              |
| --------- | ------- | ----------------------------------- |
| 8주       | 160     | `8w-160bpm.mp4`                     |
| 8주       | 165     | `8w-165bpm.mp4`                     |
| 12주      | 159-180 | `12w-159bpm.mp4` ~ `12w-180bpm.mp4` |
| 26-28주   | 126-141 | `26w-141bpm.mp4` ~ `28w-126bpm.mp4` |
| 34-35주   | 141-151 | `34w-151bpm.mp4` ~ `35w-141bpm.mp4` |

**특수 샘플:**

- `35w-141bpm_no_audio.mp4`: 오디오 트랙이 없는 순수 영상 데이터

## 🛠 기술 스택

- **언어**: Python 3.11+
- **영상 처리**: OpenCV (cv2)
- **수치 연산**: NumPy
- **파형 검출**: 컴퓨터 비전 알고리즘 (정규화, 픽셀 분석)

## 📁 프로젝트 구조

```
fetal-ultrasound-waveform-synthesizer/
├── src/                         # 소스 코드 패키지
│   ├── __init__.py
│   ├── detection/               # 파형 검출 모듈 (Phase 1)
│   │   ├── __init__.py
│   │   ├── analyzer.py          # 단일 영상 분석
│   │   └── batch.py             # 배치 처리
│   └── synthesis/               # 심박음 합성 모듈 (Phase 2, 개발 예정)
│       └── __init__.py
├── scripts/                     # 실행 스크립트
│   ├── analyze_single.py        # 단일 영상 분석 실행
│   └── batch_analyze.py         # 배치 분석 실행
├── assets/
│   └── ultrasound-samples/      # 샘플 초음파 영상 (10개)
├── .gitignore                   # Git 제외 파일
├── LICENSE                      # MIT 라이선스
├── README.md                    # 프로젝트 문서
└── requirements.txt             # Python 의존성
```

## 🔬 알고리즘 개요

1. **영상 로드**: OpenCV로 MP4 파일 읽기
2. **프레임 추출**: 각 프레임을 개별 이미지로 분리
3. **정규화**: 픽셀 값 정규화 및 전처리
4. **파형 검출**:
   - Wide rows 계산 (가로로 넓게 분포된 픽셀 영역)
   - 색상 패턴 분석 (주황색/회색 감지)
   - 파형 타입 분류
5. **결과 저장**: JSON 형식으로 분석 결과 출력

## 🗺 로드맵

- [x] Phase 1: 파형 검출 엔진 개발
- [ ] Phase 2: 심박음 합성 엔진 개발
- [ ] Phase 3: CLI 도구 개발
- [ ] Phase 4: GUI 애플리케이션
- [ ] Phase 5: 웹 API 서비스

## 🤝 기여하기

이 프로젝트는 오픈소스이며, 기여를 환영합니다!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📧 연락처

프로젝트 이슈: [GitHub Issues](https://github.com/yourusername/fetal-ultrasound-waveform-synthesizer/issues)

## 🙏 감사의 말

이 프로젝트는 개인 프로젝트로 전환되어 오픈소스로 공개되었습니다.

---

**⚠️ 면책조항**: 이 프로젝트는 연구 및 교육 목적으로 제공됩니다. 의료 진단 용도로 사용하지 마세요.
