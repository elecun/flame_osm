import cv2
import mediapipe as mp
import time

# Mediapipe 모듈
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Pose 모델 초기화
with mp_pose.Pose(
    static_image_mode=True,           # 단일 이미지 입력 모드
    model_complexity=0,               # 0, 1, 2 중 선택 (2가 가장 정확)
    enable_segmentation=False,        # 사람 영역 분할 사용 여부
    min_detection_confidence=0.5,      # 최소 탐지 신뢰도
    smooth_landmarks=True           # 랜드마크 위치 보정 사용 여부
) as pose:

    # 이미지 읽기
    image = cv2.imread("sample_colorized.jpg")
    if image is None:
        raise FileNotFoundError("sample_colorized.jpg 파일이 없습니다.")

    # BGR → RGB 변환
    color_start = time.perf_counter()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_end = time.perf_counter()
    color_time = (color_end - color_start) * 1000  # ms로 변환

    # 추론
    process_start = time.perf_counter()
    results = pose.process(image_rgb)
    process_end = time.perf_counter()
    process_time = (process_end - process_start) * 1000  # ms로 변환

    # 시간 측정 결과 출력
    print("=== Timing Results ===")
    print(f"Color conversion time: {color_time:.3f}ms")
    print(f"Process time: {process_time:.3f}ms")
    print(f"Total time: {(color_time + process_time):.3f}ms")
    print()

    # 결과 출력
    if not results.pose_landmarks:
        print("포즈 랜드마크를 탐지하지 못했습니다.")
    else:
        print("포즈 랜드마크 탐지 완료 ✅")
        for idx, lm in enumerate(results.pose_landmarks.landmark[:5]):
            print(f"Landmark {idx}: (x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f})")

        # 시각화
        annotated = image.copy()
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
        )

        # 결과 저장
        cv2.imwrite("result_pose.jpg", annotated)
        print("결과 이미지가 result_pose.jpg로 저장되었습니다.")