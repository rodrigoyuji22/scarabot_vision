from vision import Vision
from config import CAM_ID, FRAME_WIDTH, FRAME_HEIGHT

def main():
    cap = Vision(CAM_ID)
    cap.config_prop(FRAME_WIDTH, FRAME_HEIGHT)
    cap.run_vision()

main()

