import cv2
import mediapipe as mp
import time

# Mediapipe 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Holistic 모델 초기화
with mp_holistic.Holistic(
    static_image_mode=True,          # 단일 이미지 모드
    model_complexity=2,              # 0~2 (높을수록 정확하지만 느림)
    enable_segmentation=False,       # 세그멘테이션 사용 여부
    refine_face_landmarks=True,      # 정밀 얼굴(iris 포함) 랜드마크 사용
    min_detection_confidence=0.5
) as holistic:

    # 테스트 이미지 로드
    image = cv2.imread("./sample_colorized.jpg")
    if image is None:
        raise FileNotFoundError("sample_colorized.jpg 파일이 현재 경로에 없습니다.")

    # BGR → RGB 변환 시간 측정
    color_start = time.perf_counter()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_end = time.perf_counter()
    color_time = (color_end - color_start) * 1000  # ms로 변환

    # 추론 수행 시간 측정
    process_start = time.perf_counter()
    results = holistic.process(image_rgb)
    process_end = time.perf_counter()
    process_time = (process_end - process_start) * 1000  # ms로 변환

    # 시간 측정 결과 출력
    print("=== Timing Results ===")
    print(f"Color conversion time: {color_time:.3f}ms")
    print(f"Process time: {process_time:.3f}ms")
    print(f"Total time: {(color_time + process_time):.3f}ms")
    print()

    # 결과 출력
    print("=== Pose Landmarks ===")
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark[:5]):
            print(f"Landmark {idx}: (x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f})")
    else:
        print("Pose landmarks not detected.")

    print("\n=== Face Landmarks ===")
    print("Detected" if results.face_landmarks else "None")

    print("\n=== Left Hand Landmarks ===")
    print("Detected" if results.left_hand_landmarks else "None")

    print("\n=== Right Hand Landmarks ===")
    print("Detected" if results.right_hand_landmarks else "None")

    # 시각화
    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(
        annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 결과 저장
    cv2.imwrite("result_holistic.jpg", annotated)
    print("\n✅ result_holistic.jpg 파일로 결과가 저장되었습니다.")