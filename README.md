# 태아 초음파 파형 합성기

태아 초음파 영상에서 심박 파형이 나타나는 구간을 검출하고, 장기적으로는 그 파형에 맞는 태아 심박음을 생성하기 위한 프로젝트입니다.

## 현재 판단

현재 구현 기준으로는 샘플 영상에서의 파형 구간 검출은 1차 목표를 달성한 상태로 봅니다.

- 기본 검출기는 `timeseries` 기반 구간 검출기입니다.
- 기본 출력은 "파형이 존재하는 시간 구간"이며, 프레임 단위의 정밀 라벨링 도구로 고정된 상태는 아닙니다.
- 심박음 합성 엔진은 아직 구현 전입니다.

## 샘플셋 기준 성능

`scripts/evaluate_manual.py`와 `results/evaluation_results.json` 기준:

- 평가 대상: 샘플 영상 10개
- 평균 recall: `0.9104`
- 평균 precision: `0.9530`
- 대표적인 미스 케이스: `28w-126bpm.mp4`는 수동 구간 `0s~15s` 대비 자동 검출이 `4.2s~15.5s`로 시작 경계를 늦게 잡습니다.

즉, 샘플 환경에서는 "파형이 있는 구간을 찾아내는 기능"은 충분히 유효하지만, 합성 입력으로 쓰려면 출력 계약과 경계 정밀도를 더 다듬는 단계가 남아 있습니다.

## 현재 구현된 기능

- 초음파 MP4에서 파형 구간 검출
- 시작/종료 시간과 프레임 번호 반환
- 샘플 영상 배치 분석 및 JSON 저장
- 수동 구간 대비 precision/recall 평가
- 수동 구간 문서와 자동 검출 결과 비교 리포트 생성
- 시계열 파라미터 튜닝 스크립트 제공

## 현재 구현되지 않은 기능

- 태아 심박음 합성 엔진
- 검출 결과와 오디오 합성을 연결하는 안정된 출력 스키마
- 기본 모드에서의 신뢰 가능한 파형 타입 분류
- 프레임 단위 정밀 라벨링 워크플로

## 설치

### 요구사항

- Python 3.11+
- OpenCV
- NumPy

### 설정

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 사용 방법

### 단일 영상 분석

```bash
./venv/bin/python scripts/analyze_single.py assets/ultrasound-samples/28w-126bpm.mp4
```

기본 출력은 검출된 구간 로그입니다.

예시:

```text
[TS] 영상: assets/ultrasound-samples/28w-126bpm.mp4
  FPS: 30.0, 총 프레임: 4792, 길이: 159.7초

  [TS] 파형 감지 구간: 1개
    구간 1: 4.20초 ~ 15.50초 (11.30초)
```

### 배치 분석 결과 저장

```bash
./venv/bin/python scripts/analyze_all_save.py
```

기본 결과 파일:

- `results/waveform_batch_results.json`

### 수동 구간 대비 평가

```bash
./venv/bin/python scripts/evaluate_manual.py
```

결과 파일:

- `results/evaluation_results.json`

### 수동 구간 문서와 차이 리포트 생성

```bash
./venv/bin/python scripts/compare_manual_intervals.py
```

결과 파일:

- `results/manual_waveform_intervals_comparison.md`
- `results/manual_interval_diffs.json`

### 시계열 파라미터 튜닝

```bash
./venv/bin/python scripts/tune_timeseries.py
```

## 기본 출력 형식

기본 공개 API `analyze_video()`는 현재 `timeseries` 모드로 동작하며, 구간 리스트를 반환합니다.

예시:

```json
[
  {
    "start_time": 4.2,
    "end_time": 15.5,
    "type": "timeseries",
    "start_frame": 126,
    "end_frame": 465
  }
]
```

이 구조는 다음 단계에서 합성 엔진이 소비할 수 있는 형태로 정리될 필요가 있습니다.

## 프로젝트 구조

```text
fetal-ultrasound-waveform-synthesizer/
├── docs/
│   └── REFACTORING_PLAN.md
├── annotations/
│   └── manual_waveform_intervals.md
├── src/
│   ├── detection/
│   │   └── analyzer.py
│   └── synthesis/
│       └── __init__.py
├── scripts/
│   ├── analyze_all_save.py
│   ├── analyze_single.py
│   ├── compare_manual_intervals.py
│   ├── evaluate_manual.py
│   └── tune_timeseries.py
├── results/
│   ├── evaluation_results.json
│   ├── manual_interval_diffs.json
│   ├── manual_waveform_intervals_comparison.md
│   └── waveform_batch_results.json
├── README.md
└── requirements.txt
```

## 다음 단계

- 검출 결과를 합성 친화적인 스키마로 고정
- 구간 시작/종료 경계 정밀도 개선
- 파형 특성에서 심박음 파라미터로의 매핑 정의
- `src/synthesis/`에 실제 심박음 생성 파이프라인 구현

## 주의

이 프로젝트는 연구 및 개발 목적입니다. 의료 진단 용도로 사용하면 안 됩니다.
