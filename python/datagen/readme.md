사용 방법
기본 사용 (rotate 없이)
python 1_bodykps_gen.py --path /path/to/videos
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