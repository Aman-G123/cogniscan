# CogniScan

CogniScan is a simple document scanner built using Python and OpenCV. It takes an image of a document placed on a surface, detects the document automatically, straightens it, enhances it, and optionally converts it into a PDF.

---

## What it does

* Detects document edges automatically
* Corrects perspective (removes tilt/skew)
* Removes background noise using GrabCut
* Enhances the image (black & white or color)
* Crops the final output cleanly
* Exports the result as an image or PDF

---

## Project Structure

```
cogniscan/
│
├── cogniscan.py              # main script
├── utils/
│   ├── preprocess.py        # image loading + preprocessing
│   ├── edge_detection.py    # document detection logic
│   ├── enhancement.py       # scan enhancement + cropping
│   ├── transform.py         # perspective transform
│
├── samples/                 # input images (doc1, doc2, etc.)
├── output/                  # generated results
├── requirements.txt
└── README.md
```

---

## Installation

It is recommended to use **Python 3.11** and set up a virtual environment to avoid any dependency conflicts.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

---

## How to use

Basic usage:

```
python cogniscan.py --input samples/doc1.jpg
```

---

### Available modes

You can control the output type:

```
python cogniscan.py --input samples/doc1.jpg --mode bw
python cogniscan.py --input samples/doc1.jpg --mode color
python cogniscan.py --input samples/doc1.jpg --mode original
```

---

### Export as PDF

```
python cogniscan.py --input samples/doc1.jpg --mode bw --pdf
```

This will generate both:

* processed image (`.jpg`)
* PDF file (`.pdf`)

---

## Output

All outputs are saved in the `output/` folder.

Example:

```
output/
├── result_bw.jpg
├── result_bw.pdf
```

---

## How it works (brief)

1. Image is resized and blurred
2. Edges are detected and contours are found
3. The largest valid 4-point contour is assumed to be the document
4. Perspective transform is applied
5. Image is enhanced (thresholding or color enhancement)
6. Final image is cropped and saved

---

## Limitations

* Works best when the document is clearly visible
* Very cluttered backgrounds can affect detection
* Extreme lighting conditions may reduce accuracy

---

## Why this project

This was built to understand real-world computer vision pipelines — not just theory, but how systems like document scanners actually work step by step.

---

## Future improvements

* Batch scanning (multiple images → single PDF)
* Simple UI (web or desktop)
* Auto-rotation correction
* Better edge detection using ML

---