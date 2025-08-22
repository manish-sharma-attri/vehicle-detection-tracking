import cv2
from plate_reader import read_plate

def detect_and_read_plate(frame, vehicle_bbox, anpr_model, conf=0.5):
    """
    Detects number plate inside a vehicle bbox and reads text using EasyOCR.

    Args:
        frame (numpy.ndarray): Full frame (BGR).
        vehicle_bbox (tuple/list): Vehicle bounding box (x1, y1, x2, y2).
        anpr_model (YOLO): YOLO ANPR model instance.
        conf (float): Confidence threshold for ANPR model.

    Returns:
        list of dict: [{"bbox": (x1, y1, x2, y2), "text": str, "conf": float}]
    """
    x1, y1, x2, y2 = map(int, vehicle_bbox)

    # Crop vehicle region
    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return []

    # Run ANPR model
    plate_results = anpr_model.predict(vehicle_crop, conf=conf)

    plates_info = []
    for r in plate_results:
        for plate in r.boxes:
            px1, py1, px2, py2 = map(int, plate.xyxy[0])

            # Convert coords relative to full frame
            plate_x1, plate_y1 = x1 + px1, y1 + py1
            plate_x2, plate_y2 = x1 + px2, y1 + py2

            # Run OCR on plate
            text, ocr_conf = read_plate(frame, (plate_x1, plate_y1, plate_x2, plate_y2))

            plates_info.append({
                "bbox": (plate_x1, plate_y1, plate_x2, plate_y2),
                "text": text,
                "conf": ocr_conf
            })

    return plates_info
