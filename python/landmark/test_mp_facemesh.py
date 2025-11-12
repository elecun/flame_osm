import cv2
import mediapipe as mp
import time

# Mediapipe 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# FaceDetection 모델 초기화 (BlazeFace detection bbox 확인용)
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,               # 0: 2m 이내 단거리, 1: 5m 이내 장거리
    min_detection_confidence=0.5
)

# FaceMesh 모델 초기화
with mp_face_mesh.FaceMesh(
    static_image_mode=True,          # 단일 이미지 모드
    max_num_faces=1,                 # 감지할 최대 얼굴 수
    refine_landmarks=True,           # 정밀 랜드마크 (iris 포함)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # 테스트 이미지 로드
    image = cv2.imread("./sample_face.jpg")
    if image is None:
        raise FileNotFoundError("sample_face.jpg 파일이 현재 경로에 없습니다.")

    # BGR → RGB 변환 시간 측정
    color_start = time.perf_counter()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_end = time.perf_counter()
    color_time = (color_end - color_start) * 1000  # ms로 변환

    # BlazeFace detection 실행 (detection bbox 확인용)
    detection_start = time.perf_counter()
    detection_results = face_detection.process(image_rgb)
    detection_end = time.perf_counter()
    detection_time = (detection_end - detection_start) * 1000  # ms로 변환

    # FaceMesh 추론 수행 시간 측정
    process_start = time.perf_counter()
    results = face_mesh.process(image_rgb)
    process_end = time.perf_counter()
    process_time = (process_end - process_start) * 1000  # ms로 변환

    # 시간 측정 결과 출력
    print("=== Timing Results ===")
    print(f"Color conversion time: {color_time:.3f}ms")
    print(f"BlazeFace detection time: {detection_time:.3f}ms")
    print(f"FaceMesh process time: {process_time:.3f}ms")
    print(f"Total time: {(color_time + detection_time + process_time):.3f}ms")
    print()

    # BlazeFace detection 결과 출력
    print("=== BlazeFace Detection Results ===")
    if detection_results.detections:
        print(f"Detected {len(detection_results.detections)} face(s)")
        for idx, detection in enumerate(detection_results.detections):
            print(f"\nDetection {idx + 1}:")
            print(f"  Confidence: {detection.score[0]:.3f}")
            bbox = detection.location_data.relative_bounding_box
            print(f"  Bounding box (relative): x={bbox.xmin:.3f}, y={bbox.ymin:.3f}, w={bbox.width:.3f}, h={bbox.height:.3f}")
    else:
        print("No face detected.")
    print()

    # FaceMesh 결과 출력
    print("=== Face Mesh Results ===")
    if results.multi_face_landmarks:
        print(f"Detected {len(results.multi_face_landmarks)} face(s)")

        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            print(f"\nFace {face_idx + 1}:")
            print(f"  Total landmarks: {len(face_landmarks.landmark)}")
    else:
        print("No face detected.")

    # 시각화
    annotated = image.copy()
    img_h, img_w, _ = image.shape

    # BlazeFace detection bounding box 그리기 (빨간색 - 굵은 선)
    if detection_results.detections:
        for detection in detection_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * img_w)
            y_min = int(bbox.ymin * img_h)
            w = int(bbox.width * img_w)
            h = int(bbox.height * img_h)

            # BlazeFace detection bbox (빨간색, 굵게)
            cv2.rectangle(annotated, (x_min, y_min), (x_min + w, y_min + h), (0, 0, 255), 3)
            cv2.putText(annotated, f"Confidence: {detection.score[0]:.2f}",
                       (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # FaceMesh landmarks 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Face mesh tesselation (전체 메쉬)
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Contours (얼굴 윤곽선)
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Irises (홍채) - refine_landmarks=True일 때만 478개 랜드마크 포함
            if len(face_landmarks.landmark) >= 478:
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

    # 결과 저장
    cv2.imwrite("result_facemesh.jpg", annotated)
    print("\n✅ result_facemesh.jpg 파일로 결과가 저장되었습니다.")

# FaceDetection 정리
face_detection.close()
