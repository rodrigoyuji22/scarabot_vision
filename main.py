from ultralytics import YOLO
import cv2

# model = YOLO("yolov11m.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("camera nao foi encontrada")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("camera desconectada")
        break

    cv2.imshow("teste", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
