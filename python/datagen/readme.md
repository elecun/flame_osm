# Data Generation for Testing

## 1_bodykps_gen
Body Keypoints Data Generation using YOLOv11-pose
```
$ python 1_bodykps_gen.py --path test.avi --no-batch --check --model yolo11x-pose.pt --rotate
```

## 2_facekps_gen
```
$ python 2_facekps_gen.py --path test.avi --no-batch --check --device cuda --type both --rotate

```

## 3_headpose_gen
```
$ python 3_headpose_gen.py --path test.avi --no-batch --face-landmark face_kps_test.csv --check --rotate
```

## 4_gaze_gen
```
python 4_gaze_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --model mpiifacegaze --device cuda --rotate 0 --check
```

## 5 EEG
```
python 5_eeg_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --time-reference 0 --method linear
```

## 6 Eyetracker
```
python 6_eyetracker_gen.py --path /media/iae-vc/T9/dataset/iae_avsim/raw/6_남재현/2024-12-12-14-53-24/ --time-reference 0 --window-size 30
```

## 7 Merge