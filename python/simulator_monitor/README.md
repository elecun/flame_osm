# Simulator Monitor & Visual Stimulus System

PyQt6 및 OpenCV 기반의 시뮬레이터 모니터링, 피험자 시각 자극 제시 및 아이트래커(Pupil Labs Neon) 통합 시스템입니다. macOS 및 Ubuntu(Linux) 환경을 모두 지원하며, 모듈화된 아키텍처로 구현되었습니다.

---

## 1. 주요 기능

- **이중 GUI 윈도우 지원 (Qt6 .ui 기반)**:
  - **운영자 제어/모니터링 창 (`OperatorWindow`)**:
    - 최대 4개 USB 카메라 실시간 영상 표시 (좌측 상단 Camera ID 표시).
    - 미연결 카메라는 검정 바탕 중앙에 `no camera`로 깔끔하게 표시.
    - Advantech USB-4750-CE 채널 0 핸들 스위치 상태 표시 (`idle` / `pressed`).
    - Pupil Labs Neon 아이트래커 연결 상태 실시간 모니터링 (`Searching...` / `Connected` / `Not Found (Offline)`).
    - 시나리오(`*.scenario`) 선택 드롭다운 메뉴.
    - 핸들 스위치 사운드 피드백 토글 체크박스.
    - 실험 진행 상태 배지 (`IDLE` / `RECORDING`) 및 상호 배타적 START / STOP 버튼.
    - 피험자 창 전체화면 토글 버튼 (F11 단축키 지원).
  - **피험자 자극 제시창 (`SubjectWindow`)**:
    - 기본 검정색 배경 (시나리오 미실행 시 완전한 블랙 스크린 유지).
    - `exp.cfg` 설정 해상도 지원 및 F11 / 더블클릭 전체화면 전환.
    - OpenCV 기반 `.scenario` 스케줄 재생 및 종료 시 자동 STOP 전환.
- **최대 4채널 독립 스레드 카메라 원본 영상 녹화**:
  - 프레임 드롭 방지를 위한 카메라별 독립 `QThread` 제어.
  - 녹화 영상에는 워터마크 없이 **원본(Raw) 영상**을 그대로 AVI(`cam_<id>.avi`) 무음성 저장.
  - STOP 시 녹화만 안전하게 마무리되며, 실시간 화면 미리보기는 계속 유지.
- **Pupil Labs Neon 아이트래커 자동 연동**:
  - 시작 시 네트워크 mDNS/IP를 통해 Neon 기기를 자동 검색 및 연결.
  - 장치 부재 시에도 프로그램이 정상 실행되며, GUI에 `Not Found (Offline)` 상태 표시.
  - START 시 자동 녹화 시작 (`recording_start()`), STOP 시 녹화 안전 저장 (`recording_stop_and_save()`).
- **Advantech USB-4750-CE DAQ & 정밀 밀리초 로깅**:
  - 채널 0 입력 감지 시 GUI에 `pressed` 표시.
  - 입력 시마다 첫 번째 컬럼에 밀리초 포맷(`YYYY-MM-DD HH:MM:SS.mmm`)으로 `handle_switch.csv`에 실시간 플러시 기록.
  - macOS 등 드라이버 미지원 환경에서도 자동 Mock 모드로 동작하며 Space 키/버튼으로 스위치 테스트 지원.
- **저지연 사운드 피드백**:
  - 프로그램 시작 시 사운드 파일(`assets/beep.wav`)을 메모리에 사전 로드 (`QSoundEffect`).
  - 빠른 연타 시 이전 사운드가 믹싱되지 않고 즉시 중단된 후 새로 재생.

---

## 2. 디렉토리 구조

```
simulator_monitor/
├── venv/                       # Python 가상환경 (Python >= 3.10)
├── exp.cfg                     # 핵심 파라메터 설정 파일
├── default.scenario            # 기본 시나리오
├── scenario/                   # 시나리오 파일 디렉토리 (*.scenario)
│   ├── default.scenario
│   └── reaction_time.scenario
├── assets/
│   └── beep.wav                # 스위치 피드백 사운드
├── run.py                      # 메인 실행 엔트리포인트
├── requirment.txt              # pip freeze 결과 라이브러리 목록
├── requirements.txt            # 표준 라이브러리 의존성 명세
├── ui/
│   ├── operator_window.ui      # 운영자 화면 Qt UI 정의
│   └── subject_window.ui       # 피험자 화면 Qt UI 정의
├── src/
│   ├── config.py               # 설정 파서
│   ├── camera/
│   │   ├── camera_thread.py    # 카메라 캡처 & 무손실 원본 AVI 녹화 스레드
│   │   └── camera_manager.py   # 다중 카메라 동기 제어 매니저
│   ├── daq/
│   │   ├── daq_worker.py       # USB-4750 Polling 및 Mock 지원
│   │   └── daq_logger.py       # handle_switch.csv 밀리초 로거
│   ├── eyetracker/
│   │   └── neon_worker.py      # Pupil Labs Neon 검색 및 녹화 제어
│   ├── sound/
│   │   └── sound_player.py     # 저지연 사전로드 사운드 플레이어
│   ├── scenario/
│   │   └── scenario_player.py  # OpenCV 기반 시나리오 렌더러
│   ├── gui/
│   │   ├── operator_window.py  # 운영자 창 비즈니스 로직
│   │   └── subject_window.py   # 피험자 창 비즈니스 로직
│   └── utils/
│       └── time_utils.py       # 밀리초 정밀도 시간 포맷 유틸리티
└── tests/
    ├── test_simulator.py       # 단위 테스트
    └── test_integration.py     # 통합 세션 테스트
```

---

## 3. 실행 방법

```bash
# 가상환경 활성화 (Python >= 3.10)
source venv/bin/activate

# 프로그램 실행 (기본 exp.cfg 로드)
python run.py --config exp.cfg
```

---

## 4. 설정 파일 (`exp.cfg`) 안내

```ini
[SYSTEM]
output_root_dir = ./records

[CAMERAS]
camera_ids = 0, 1, 2, 3
fps = 30
width = 640
height = 480
codec = XVID
mock_if_missing = true

[SUBJECT_DISPLAY]
width = 1920
height = 1080
fullscreen = false
scenario_dir = ./scenario
scenario_file = default.scenario

[DAQ]
device_description = USB-4750,BID#0
profile_path = 
port = 0
channel = 0
poll_interval_ms = 10
mock_mode = auto

[SOUND]
sound_file = assets/beep.wav
sound_feedback = true

[EYETRACKER]
auto_discover = true
device_address = 
device_port = 8080
search_timeout_sec = 2.0
```

---

## 5. 테스트 실행

```bash
./venv/bin/python -m unittest discover tests
```
