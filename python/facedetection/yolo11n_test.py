import argparse
import time
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO11n-face test script for face detection")
    
    # 상호 배타적 그룹으로 --image와 --video를 설정 (둘 중 하나만 선택)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', action='store_true', help='입력 파일이 이미지인 경우 사용')
    group.add_argument('--video', action='store_true', help='입력 파일이 동영상인 경우 사용')
    
    # --in 옵션으로 파일명 입력. dest='input_file'을 사용하여 파이썬 예약어 'in' 회피
    parser.add_argument('--in', dest='input_file', required=True, help='입력 파일명 (실행 위치 기준)')
    
    # --out 옵션
    parser.add_argument('--out', action='store_true', help='바운딩 박스가 그려진 결과물 저장')
    
    # --model 옵션
    parser.add_argument('--model', required=True, help='사용할 모델 파일명 (.pt, .onnx 등)')

    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_path}")
        return

    model_path = args.model
    if not os.path.exists(model_path) and not model_path.endswith('.pt'):
        # .pt 파일은 로컬에 없으면 다운로드될 수 있지만, onnx 등은 로컬에 있어야 하므로 체크
        print(f"[Warning] 로컬에서 모델 파일을 찾지 못했습니다. Ultralytics에서 다운로드를 시도할 수 있습니다: {model_path}")

    # 모델 로드
    print(f"모델 로딩 중... ({model_path})")
    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"[Error] 모델 로딩/다운로드 실패: {e}")
        return

    out_path = None
    if args.out:
        p = Path(input_path)
        # 확장자 앞에 _result를 붙여서 출력 파일명 생성
        out_path = f"{p.stem}_result{p.suffix}"
        print(f"출력 파일 저장 경로: {out_path}")

    total_frames = 0
    total_time = 0.0

    if args.image:
        print(f"이미지 처리 시작: {input_path}")
        img = cv2.imread(input_path)
        if img is None:
            print("[Error] 이미지를 읽을 수 없습니다.")
            return
        
        # Inference 시간 측정
        start_time = time.time()
        results = model(img, verbose=False)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        total_frames += 1

        if args.out:
            # 모델 예측 결과 시각화 (바운딩 박스 포함된 ndarray 반환)
            res_plotted = results[0].plot()
            cv2.imwrite(out_path, res_plotted)
            print(f"결과 이미지가 저장되었습니다: {out_path}")

    elif args.video:
        print(f"동영상 처리 시작: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("[Error] 동영상을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if args.out:
            # 동영상 확장자에 따른 코덱 설정
            ext = Path(input_path).suffix.lower()
            if ext == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print("프레임 처리 중...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            results = model(frame, verbose=False)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_frames += 1

            if args.out:
                res_plotted = results[0].plot()
                writer.write(res_plotted)
                
            # 진행 상태 간단히 출력 (100 프레임마다)
            if total_frames % 100 == 0:
                print(f"  {total_frames} 프레임 처리 완료...")

        cap.release()
        if writer:
            writer.release()
            print(f"결과 동영상이 저장되었습니다: {out_path}")

    # 결과 요약 출력
    if total_frames > 0:
        avg_time = total_time / total_frames
        print("\n" + "="*30)
        print("         처리 요약")
        print("="*30)
        print(f"총 처리 프레임 수 : {total_frames} 프레임")
        print(f"프레임당 처리 시간: {avg_time * 1000:.2f} ms")
        print(f"총 소요 시간      : {total_time:.2f} s")
        print("="*30)

if __name__ == "__main__":
    main()
