from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

model.predict(
    source = 1,
    imgsz = 640,
    conf=0.45,
    iou=0.5,
    vid_stride=2,
    show=True,
    save=False
)

