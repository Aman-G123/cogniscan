import cv2
import numpy as np

def detect_document(gray):
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_contour = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.5 < aspect_ratio < 2.0:
                if area > max_area:
                    best_contour = approx
                    max_area = area

    if best_contour is not None:
        return best_contour

    h, w = gray.shape
    return np.array([
        [[0, 0]],
        [[w, 0]],
        [[w, h]],
        [[0, h]]
    ])