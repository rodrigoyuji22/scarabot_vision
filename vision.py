from ultralytics import YOLO
import cv2 as cv
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (usar a gpu para rodar o modelo)

class Vision:
    def __init__(self, cam, modelPath):
        self.cap = cv.VideoCapture(cam)
        self.model = YOLO(modelPath)

    def config_prop(self, w, h):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    def run_vision(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame, imgsz = 800, conf = 0.6, iou = 0.5)
            annotated = results[0].plot()

            cv.imshow("capture", annotated)
        
        self.cap.release()
        cv.destroyAllWindows()


        
