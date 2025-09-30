from ultralytics import YOLO
import cv2 as cv
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (usar a gpu para rodar o modelo)

# configuracao da captura de imagem
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO("best.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
# tenta abrir a cam, se nao estiver capturando nada quebra o loop
    result = model(frame, imgsz = 640, conf = 0.6, iou = 0.5)
    annotation = result[0].plot()
    # pega o primeiro frame e gera a bounding box + classe + conf
    cv.imshow("captura", annotation)
    # gera a janela de visualizacao

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    # bind para quebrar o loop

cap.release()
cv.destroyAllWindows()
# libera recurso da cam e fecha a janela do imshow()
