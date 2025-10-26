# app/services/face_utils.py
import cv2
import numpy as np

def crop_and_zoom(image: np.ndarray, box, zoom_factor: float = 1.2) -> np.ndarray:
    """Crop and zoom slightly around the bounding box."""
    h, w, _ = image.shape
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = x2 - x1, y2 - y1
    new_w, new_h = int(bw * zoom_factor), int(bh * zoom_factor)
    x1_new = max(0, cx - new_w // 2)
    y1_new = max(0, cy - new_h // 2)
    x2_new = min(w, cx + new_w // 2)
    y2_new = min(h, cy + new_h // 2)
    return image[y1_new:y2_new, x1_new:x2_new]
