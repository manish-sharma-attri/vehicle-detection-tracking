import cv2
from detector import YoloDetector
from tracker import Tracker
from utils import detect_and_read_plate
from ultralytics import YOLO

# ---------------------------
# Load models
# ---------------------------
vehicle_detector = YoloDetector(
    model_path="E:/vehicle_number_plate_detection/models/yolo11m.pt",
    confidence=0.5
)

anpr_model = YOLO("E:/vehicle_number_plate_detection/models/ANPR.pt")
tracker = Tracker()

# ---------------------------
# Load video
# ---------------------------
cap = cv2.VideoCapture("E:/vehicle_number_plate_detection/input_videos/testing_video.mp4")
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video or error reading frame.")
        break

    # ---------------------------
    # Step 1: Detect vehicles
    # ---------------------------
    detections = vehicle_detector.detect(frame)

    # Convert detections for DeepSort
    deep_sort_detections = []
    for det in detections:
        bbox, class_id, conf, class_name = det
        x, y, w, h = bbox
        deep_sort_detections.append(([x, y, w, h], conf, class_name))

    # ---------------------------
    # Step 2: Track vehicles
    # ---------------------------
    tracking_ids, tracked_boxes = tracker.track(deep_sort_detections, frame)

    # ---------------------------
    # Step 3: Detect + OCR number plates
    # ---------------------------
    for tracking_id, bbox in zip(tracking_ids, tracked_boxes):
        x1, y1, x2, y2 = map(int, bbox)

        # Draw vehicle box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {tracking_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 🔹 Run ANPR + OCR helper
        plates = detect_and_read_plate(frame, (x1, y1, x2, y2), anpr_model)

        for plate in plates:
            px1, py1, px2, py2 = plate["bbox"]
            text, conf = plate["text"], plate["conf"]

            # Draw plate box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            if text:
                cv2.putText(frame, f"{text} ({conf:.2f})",
                            (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

    # ---------------------------
    # Show frame
    # ---------------------------
    cv2.imshow("Vehicle + Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
