from vision import Vision
from pathlib import Path
from config import YOLO_PATH, CAM_ID, FRAME_WIDTH, FRAME_HEIGHT

def main():
    cap = Vision(CAM_ID, YOLO_PATH)
    cap.config_prop(FRAME_WIDTH, FRAME_WIDTH)
    cap.run_vision()


main()

