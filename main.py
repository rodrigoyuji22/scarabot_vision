from vision import Vision
from pathlib import Path
from config import YOLO_PATH, CAM_ID, FRAME_WIDTH, FRAME_HEIGHT

cap = Vision(CAM_ID, YOLO_PATH)
cap.config_prop(FRAME_WIDTH, FRAME_WIDTH)
cap.run_vision()

