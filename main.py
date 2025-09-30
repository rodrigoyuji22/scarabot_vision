from vision import Vision
from pathlib import Path

path = Path("best.pt")

cap = Vision(0, path)
cap.config_prop(1280,720)
cap.run_vision()

