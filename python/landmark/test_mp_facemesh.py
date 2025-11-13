import cv2
import mediapipe as mp
import time
import numpy as np

# Mediapipe 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# 3D Head Pose Estimation을 위한 3D 모델 포인트 (표준 얼굴 모델)
# IPD(Inter-Pupillary Distance) 기준: 63mm
# 주요 랜드마크 인덱스: 코끝, 턱, 왼쪽눈 왼쪽 모서리, 오른쪽눈 오른쪽 모서리, 왼쪽 입꼬리, 오른쪽 입꼬리
IPD_MM = 63.0  # Inter-Pupillary Distance in mm (평균 성인 기준)

# 3D 모델 포인트 (mm 단위, IPD 63mm 기준으로 스케일 조정)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # 코끝 (index 1)
    (0.0, -80.0, -30.0),         # 턱 (index 152)
    (-45.0, 40.0, -30.0),        # 왼쪽눈 왼쪽 모서리 (index 33) - IPD/2
    (45.0, 40.0, -30.0),         # 오른쪽눈 오른쪽 모서리 (index 263) - IPD/2
    (-25.0, -40.0, -20.0),       # 왼쪽 입꼬리 (index 61)
    (25.0, -40.0, -20.0)         # 오른쪽 입꼬리 (index 291)
], dtype=np.float64)

# 대응하는 FaceMesh 랜드마크 인덱스
FACE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# 좌우 눈동자 중심 랜드마크 인덱스 (IPD 계산용)
LEFT_EYE_CENTER = 468   # 왼쪽 눈동자 중심 (iris center)
RIGHT_EYE_CENTER = 473  # 오른쪽 눈동자 중심 (iris center)

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
    image = cv2.imread("./sample3.jpg")
    if image is None:
        raise FileNotFoundError("sample3.jpg 파일이 현재 경로에 없습니다.")

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

    # 3D Head Pose Estimation
    print("\n=== 3D Head Pose Estimation ===")
    if results.multi_face_landmarks:
        img_h, img_w, _ = image.shape

        # Camera matrix (focal length 추정)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # 왜곡 계수 (왜곡 없음 가정)
        dist_coeffs = np.zeros((4, 1))

        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # 2D 이미지 포인트 추출
            image_points = np.array([
                (face_landmarks.landmark[idx].x * img_w,
                 face_landmarks.landmark[idx].y * img_h)
                for idx in FACE_LANDMARK_INDICES
            ], dtype=np.float64)

            # solvePnP로 rotation과 translation 계산
            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Rotation vector를 Euler angles로 변환
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)

                # 3x4 projection matrix (decomposeProjectionMatrix는 3x4 필요)
                pose_mat = cv2.hconcat((rotation_mat, translation_vec))

                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

                pitch, yaw, roll = euler_angles.flatten()[:3]

                # IPD를 이용한 실제 거리 추정
                # 이미지에서 측정된 IPD (픽셀)
                if len(face_landmarks.landmark) >= 474:  # iris 랜드마크가 있는 경우
                    left_eye = face_landmarks.landmark[LEFT_EYE_CENTER]
                    right_eye = face_landmarks.landmark[RIGHT_EYE_CENTER]
                else:  # iris가 없으면 눈 모서리 사용
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]

                ipd_pixels = np.sqrt(
                    ((right_eye.x - left_eye.x) * img_w) ** 2 +
                    ((right_eye.y - left_eye.y) * img_h) ** 2
                )

                # 스케일 팩터 계산 (픽셀 당 mm)
                # focal_length와 실제 거리를 고려한 스케일
                scale_factor = IPD_MM / ipd_pixels

                # Translation vector를 실제 mm 단위로 변환
                # translation_vec은 카메라 좌표계에서의 위치 (임의 단위)
                # IPD 비율로 스케일 조정
                tx_mm = translation_vec[0][0] * scale_factor
                ty_mm = translation_vec[1][0] * scale_factor
                tz_mm = translation_vec[2][0] * scale_factor

                # 거리 계산 (카메라로부터의 거리)
                distance_mm = np.sqrt(tx_mm**2 + ty_mm**2 + tz_mm**2)
                distance_cm = distance_mm / 10.0
                distance_m = distance_mm / 1000.0

                print(f"\nFace {face_idx + 1} Head Pose:")
                print(f"  방향 (Orientation):")
                print(f"    Pitch (상하): {pitch:.2f}°")
                print(f"    Yaw (좌우): {yaw:.2f}°")
                print(f"    Roll (기울기): {roll:.2f}°")
                print(f"\n  위치 (Position - 카메라 기준):")
                print(f"    X (좌우): {tx_mm:.1f}mm ({tx_mm/10:.1f}cm)")
                print(f"    Y (상하): {ty_mm:.1f}mm ({ty_mm/10:.1f}cm)")
                print(f"    Z (깊이): {tz_mm:.1f}mm ({tz_mm/10:.1f}cm)")
                print(f"    거리 (Distance): {distance_cm:.1f}cm ({distance_m:.2f}m)")

    # 시각화
    annotated = image.copy()
    img_h, img_w, _ = image.shape

    # Camera matrix 재사용
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

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

    # FaceMesh landmarks 및 3D Head Pose 그리기
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

            # 3D Head Pose 축 그리기
            # 2D 이미지 포인트 추출
            image_points = np.array([
                (face_landmarks.landmark[idx].x * img_w,
                 face_landmarks.landmark[idx].y * img_h)
                for idx in FACE_LANDMARK_INDICES
            ], dtype=np.float64)

            # solvePnP로 rotation과 translation 계산
            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # 코끝을 중심으로 3D 축 그리기
                nose_tip = (int(face_landmarks.landmark[1].x * img_w),
                           int(face_landmarks.landmark[1].y * img_h))

                # 3D 축 포인트 (X: 빨강, Y: 초록, Z: 파랑)
                axis_length = 200.0
                axis_3d = np.float32([
                    [axis_length, 0, 0],      # X축 (빨강 - 좌우)
                    [0, axis_length, 0],      # Y축 (초록 - 상하)
                    [0, 0, -axis_length]      # Z축 (파랑 - 앞뒤)
                ])

                # 3D 축을 2D로 투영
                axis_2d, _ = cv2.projectPoints(
                    axis_3d,
                    rotation_vec,
                    translation_vec,
                    camera_matrix,
                    dist_coeffs
                )

                # 축 그리기
                axis_2d = axis_2d.reshape(-1, 2)
                # X축 (빨강)
                cv2.line(annotated, nose_tip, tuple(axis_2d[0].astype(int)), (0, 0, 255), 3)
                # Y축 (초록)
                cv2.line(annotated, nose_tip, tuple(axis_2d[1].astype(int)), (0, 255, 0), 3)
                # Z축 (파랑)
                cv2.line(annotated, nose_tip, tuple(axis_2d[2].astype(int)), (255, 0, 0), 3)

    # 결과 저장
    cv2.imwrite("result_facemesh.jpg", annotated)
    print("\n✅ result_facemesh.jpg 파일로 결과가 저장되었습니다.")

# FaceDetection 정리
face_detection.close()
