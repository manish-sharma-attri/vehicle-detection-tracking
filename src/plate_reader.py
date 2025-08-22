# src/plate_reader.py
import easyocr
import cv2

# Initialize EasyOCR reader once (English letters + digits)
reader = easyocr.Reader(['en'])

def read_plate(frame, bbox):
    """
    Extracts number plate text from an image frame using EasyOCR.

    Args:
        frame (numpy.ndarray): Original image frame (BGR from OpenCV).
        bbox (tuple/list): Bounding box (x1, y1, x2, y2).

    Returns:
        str: Detected plate text (empty string if not found).
        float: Confidence score.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop plate
    plate_crop = frame[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return "", 0.0

    # Convert to grayscale (better for OCR)
    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Run OCR
    results = reader.readtext(gray_plate)

    if results:
        text = results[0][1]       # recognized text
        confidence = results[0][2] # confidence score
        return text, confidence
    else:
        return "", 0.0
