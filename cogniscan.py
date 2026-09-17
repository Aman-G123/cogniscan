import cv2
import numpy as np
import argparse
import img2pdf

from utils.preprocess import load_image, preprocess
from utils.edge_detection import detect_document
from utils.enhancement import enhance, auto_crop

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def grabcut_refine(image):
    mask = np.zeros(image.shape[:2], np.uint8)

    h, w = image.shape[:2]

    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    result = image * mask2[:, :, np.newaxis]

    return result

def save_as_pdf(image_path, pdf_path):
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_path))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", default="bw", choices=["bw", "color", "original"])
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    image = load_image(args.input)
    image = grabcut_refine(image)
    orig, resized, gray, blurred, ratio = preprocess(image)

    contour = detect_document(blurred)

    if contour is None:
        print("No document detected")
        return

    contour = contour.reshape(4, 2) * ratio

    warped = four_point_transform(orig, contour)

    if args.mode == "original":
        scanned = warped
    else:
        scanned = enhance(warped, mode=args.mode)
    scanned = auto_crop(scanned)

    output_path = f"output/result_{args.mode}.jpg"
    cv2.imwrite(output_path, scanned, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Scan saved to {output_path}")

    if args.pdf:
        pdf_path = f"output/result_{args.mode}.pdf"
        save_as_pdf(output_path, pdf_path)
        print(f"PDF saved to {pdf_path}")

if __name__ == "__main__":
    main()