import cv2 as cv
import numpy as np
from config import CAM_ID, FRAME_WIDTH, FRAME_HEIGHT, HOMOGRAPHY_PATH

# Coordenadas reais medidas em mm (ordem horaria)
pts_world = np.array([
    [0, 0],
    [0, 315],
    [670, 315],
    [670, 0]
], dtype=np.float32)

WINDOW_TITLE = "Calibracao - clique nos 4 pontos da esteira"

cap = cv.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open camera {CAM_ID}")
cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cv.namedWindow(WINDOW_TITLE)
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Ponto {len(clicked_points)}: ({x}, {y})")


cv.setMouseCallback(WINDOW_TITLE, mouse_callback)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    for (x, y) in clicked_points:
        cv.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
    cv.imshow(WINDOW_TITLE, frame)
    if cv.waitKey(1) == 27 or len(clicked_points) >= 4:
        break

cap.release()
cv.destroyAllWindows()

if len(clicked_points) < 4:
    raise ValueError("Voce precisa clicar em 4 pontos!")

pts_image = np.array(clicked_points, dtype=np.float32)
H, _ = cv.findHomography(pts_image, pts_world, cv.RANSAC)
np.save(str(HOMOGRAPHY_PATH), H)
print(f"Homografia salva em {HOMOGRAPHY_PATH}")
print(H)
