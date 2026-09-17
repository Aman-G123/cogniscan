import cv2

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Error loading image")
    return image

def preprocess(image):
    ratio = image.shape[0] / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, resized, gray, blurred, ratio