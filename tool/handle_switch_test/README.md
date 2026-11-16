# USB-4750 입력 감시기 (Ubuntu 22.04)

`handle_switch_test.py`는 Advantech USB-4750의 IDI0~IDI15를 감시합니다.
입력이 변할 때만 콘솔에 ISO-8601 타임스탬프와 16개 채널 값을 `0`/`1`로
출력합니다. `Ctrl+C`를 누르면 정상 종료합니다.

로그 값은 액티브 로우 입력을 반전해 표시합니다. `1`은 버튼 눌림,
`0`은 버튼 뗌을 뜻합니다.

## 사전 조건

* Ubuntu 22.04 64-bit
* USB-4750 연결
* 실행 중인 커널의 헤더와 `gcc`
* Advantech **DAQNavi Driver for Linux**

Advantech의 현재 Linux DAQNavi 패키지는 Ubuntu 22.04를 테스트 대상으로
명시하며 USB-4750을 지원합니다.

## DAQNavi 드라이버 설치

다음은 공식 64-bit 패키지(2026-10-16)의 다운로드 및 USB-4750 드라이버만
설치하는 명령입니다. 설치는 커널 모듈과 `/opt/advantech` 파일을 추가하므로
`sudo` 비밀번호가 필요합니다.

```bash
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r)
cd /tmp
curl -fL 'https://downloadt.advantech.com/download/downloadsr.aspx?File_Id=1-35MFHWM' \
  -o DAQ_Linux_20261016_64bit.run
chmod +x DAQ_Linux_20261016_64bit.run
sudo ./DAQ_Linux_20261016_64bit.run silent install usb4750
sudo ldconfig
sudo reboot
```

설치 뒤 장치를 꽂은 상태에서 다음 명령으로 인식 여부를 확인합니다.

```bash
lsusb | grep -i advantech
lsmod | grep -E 'bio4750|biokernbase'
```

현재 시스템에서 `lsusb`는 `1809:4703 Advantech DAQ in HS Mode`를 보여야
합니다. 장치의 Board ID가 0이 아니면 실행 시 `--device`로 바꿉니다.

## 실행

```bash
cd /home/iae-vc/dev/flame_osm/tool/handle_switch_test
../venv/bin/python handle_switch_test.py
```

기본 장치는 `USB-4750,BID#0`입니다. 예를 들어 Board ID 2라면:

```bash
../venv/bin/python handle_switch_test.py --device 'USB-4750,BID#2'
```

출력 예시:

```text
2026-11-16T17:04:12.531+09:00 IDI0=1 IDI1=0 IDI2=1 IDI3=1 IDI4=1 IDI5=1 IDI6=1 IDI7=1 IDI8=0 IDI9=1 IDI10=1 IDI11=1 IDI12=1 IDI13=1 IDI14=1 IDI15=1
```

`requirements.txt`는 비어 있는 것이 정상입니다. 이 프로그램은 PyPI
패키지를 사용하지 않고 DAQNavi가 설치하는 Python `ctypes` API와
`libbiodaq.so`를 사용합니다.
