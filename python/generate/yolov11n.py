import os
import shutil
import sys
from ultralytics import YOLO

def main():
    print("--- YOLO11n TensorRT Engine Generation Start ---")
    
    # 1. Load pre-trained model (downloads automatically if not found locally)
    model_name = "yolo11n.pt"
    print(f"Loading/downloading pre-trained model: {model_name}...")
    model = YOLO(model_name)

    # 2. Export to TensorRT format
    print("Exporting model to TensorRT engine format using GPU (FP16)...")
    try:
        # device=0 selects the first GPU; half=True exports in FP16 precision
        export_path = model.export(format="engine", device=0, half=True)
        print(f"Export command finished. Expected path: {export_path}")
    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        sys.exit(1)

    # If the exporter returned empty, search for yolo11n.engine in the current workspace
    if not export_path or not os.path.exists(export_path):
        if os.path.exists("yolo11n.engine"):
            export_path = "yolo11n.engine"
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "yolo11n.engine")):
            export_path = os.path.join(os.path.dirname(__file__), "yolo11n.engine")
        else:
            # Check under the current execution directory
            for root, dirs, files in os.walk("."):
                if "yolo11n.engine" in files:
                    export_path = os.path.join(root, "yolo11n.engine")
                    break

    if not export_path or not os.path.exists(export_path):
        print("Error: Could not locate the generated yolo11n.engine file.", file=sys.stderr)
        sys.exit(1)

    print(f"Located engine file at: {export_path}")

    # 3. Create target directory
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin/x86_64/models/pretrained"))
    print(f"Ensuring destination directory exists: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    dest_file = os.path.join(dest_dir, "yolov11n.engine")

    # 4. Move the engine file to target location
    print(f"Moving engine file from {export_path} to {dest_file}...")
    try:
        shutil.move(export_path, dest_file)
        print("--- Successfully moved model file to target location! ---")
    except Exception as e:
        print(f"Error moving file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
