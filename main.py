from ultralytics import YOLO
import cv2
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (usar a gpu para rodar o modelo)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO("best.pt")
model.predict(
    source = 0,
    imgsz = 800,
    conf=0.45,
    iou=0.5,
    vid_stride=2,
    show=True,
    save=False
)
