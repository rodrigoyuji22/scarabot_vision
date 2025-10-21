import cv2 as cv
import numpy as np

H = np.load("homography.npy")

def img_to_world(u, v):
    p = np.array([u, v, 1.0])
    Pw = H @ p
    Pw /= Pw[2]
    return float(Pw[0]), float(Pw[1])

cap = cv.VideoCapture(0)

#39 cm a partir do canto superior direito X 5 cm de altura
#39,9 cm a partir do canto inferior esquerdo X 5 cm de altura

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

print("Resolução real:", cap.get(cv.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv.CAP_PROP_FRAME_HEIGHT))


while True:
    ok, frame = cap.read()
    if not ok: break

    u, v = 960, 540  # centro do frame (ajuste se quiser)
    X, Y = img_to_world(u, v)

    cv.circle(frame, (u, v), 5, (0, 255, 0), -1)
    cv.putText(frame, f"X={X:.1f} mm Y={Y:.1f} mm", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.imshow("Teste Homografia", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
