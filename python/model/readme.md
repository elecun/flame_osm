python 2_facekps_dense_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --autofill --model yolo11x-pose.pt --rotate 0

python 2_facekps_dense_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --autofill --check --rotate 0

python 2_facekps_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --autofill --check --rotate 0 --device cuda --type 2d

python 3_headpose_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --face-landmark face_kps_0.csv --rotate 0 --check