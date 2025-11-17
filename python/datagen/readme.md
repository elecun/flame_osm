1_bodykps_gen.py 사용 방법
```
기본 사용 (rotate 없이)
python 1_bodykps_gen.py --path <path> --rotate 0 --check --autofill
ID 0번만 회전
python 1_bodykps_gen.py --path /path/to/videos --rotate 0
여러 ID 회전 (0, 1, 2번)
python 1_bodykps_gen.py --path /path/to/videos --rotate 0 1 2
Check 모드 활성화
python 1_bodykps_gen.py --path /path/to/videos --check
모든 옵션 함께 사용
python 1_bodykps_gen.py --path /path/to/videos --rotate 0 --check --autofill
출력 파일
Check 모드 활성화 시
각 ID별로 check_frame_<id>.jpg 파일이 생성됩니다:
check_frame_0.jpg - ID 0의 첫 번째 프레임 시각화
check_frame_1.jpg - ID 1의 첫 번째 프레임 시각화
```

5_merge.py 사용 방법
```
기본 사용 (2개 파일)
python 5_merge.py file1.csv file2.csv
여러 파일 merge
python 5_merge.py file1.csv file2.csv file3.csv file4.csv
출력 파일명 지정
python 5_merge.py file1.csv file2.csv --output custom_output.csv
예시
file1.csv (30Hz, 기준):
timestamp,data1
0.000,10
0.033,20
0.066,30
file2.csv (10Hz):
timestamp,data2
0.000,100
0.100,200
merged.csv (30Hz, file1 기준):
timestamp,data1,data2
0.000,10,100
0.033,20,133.3  (선형보간: 100 + (200-100)*(0.033/0.100))
0.066,30,166.6  (선형보간)
```