import cv2
import numpy as np

def enhance(image, mode="bw"):
    if mode == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, scanned = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return scanned

    elif mode == "color":
        result = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        return result

    else:
        return image


def auto_crop(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]