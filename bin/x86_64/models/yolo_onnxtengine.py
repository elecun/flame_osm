from ultralytics import YOLO

model = YOLO("yolo11x-pose.pt")
model.export(format="engine")  # creates 'yolo11n.engine'