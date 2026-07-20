# POTI 값 변경
8V(0xff)가 선형적으로 설정하면 되는것을 감안하여 6.5V는 0xCF 설정
- POC전압 8V설정 (Default)
```
ADR_POTI              2 0x11 0xff                         # set poc for slot 0: 0xff for 8V
ADR_POTI              2 0x12 0xff
```
- POC전압 6.5V설정
```
ADR_POTI              2 0x11 0xCF                         # set poc for slot 0: 6.5V 설정
ADR_POTI              2 0x12 0xCF
```


# D-Phy 및 CSI-2 클록 미스매치
- UB953은 센서(OV2778)로부터 영상을 받아 UB954로 보내고, UB954는 다시 proFRAME FPGA로 MIPI CSI-2 스트림을 쏩니다. 이 과정에서 클록 모드 미스매치가 누적될 수 있을 가능성.
- proFRAME의 MIPI 캡처 코어는 데이터가 안 들어오는 구간에도 클록이 유지되는 것을 선호하기 때문에, 954의 0x33 0x03은 Non-continuous 모드이므로, 영상 데이터가 잠깐 안 들어올 때(수직 동기화 블랭킹 구간 등) proFRAME FPGA가 이를 수신 대기 에러(CAPTURE_ERROR)로 판단할 가능성
- TI954 레지스터 0x33을 0x43으로 바꾸어 Continuous Clock 모드로 강제 고정 필요.
- proFRAME FPGA 수신 안정성을 위해 Continuous Clock 모드 활성화 (0x03 -> 0x43)
```
ADR_DESERIALIZER 2 0x33 0x43 # enable CSI-2 output with 4 lanes & Continuous Clock
```