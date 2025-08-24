import cv2
from .detector import YoloDetector
from .tracker import Tracker
from .utils import detect_and_read_plate
from ultralytics import YOLO
from .speed_estimate import SpeedEstimator
from .logger import VehicleLogger

SPEED_LIMITS = {
    "car": 80,
    "truck": 60,
    "bus": 60,
    "motorbike": 70
}

def process_video(input_video,output_video,output_csv):
    # ---------------------------
    # Load models
    # ---------------------------
    vehicle_detector = YoloDetector(
        model_path=r"E:\vehicle_number_plate_detection\models\yolo11n.pt",
        confidence=0.5
    )

    anpr_model = YOLO(r"E:\vehicle_number_plate_detection\models\ANPR.pt")
    tracker = Tracker()
    class_map={}
    # ---------------------------
    # Setup video I/O
    # ---------------------------
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # ---------------------------
    # Init helpers
    # ---------------------------
    speed_estimator = SpeedEstimator(ppm=10, fps=fps)  # adjust ppm for calibration
    logger = VehicleLogger(filepath=output_csv)
    max_frames = fps * 8  # 13 seconds only
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count>=max_frames:
            print("✅ End of video.")
            break

        frame_count += 1

        # Step 1: Vehicle detection
        detections = vehicle_detector.detect(frame)
        deep_sort_detections = []
        class_name="unknown"
        for det in detections:
            bbox, class_id, conf, class_name = det
            x, y, w, h = bbox
            deep_sort_detections.append(([x, y, w, h], conf, class_name))
            class_map[len(deep_sort_detections)-1] = class_name

        # Step 2: Tracking
        tracking_ids, tracked_boxes = tracker.track(deep_sort_detections, frame)

        # Step 3: Plate + Speed + Logging
        for idx,(tracking_id, bbox) in enumerate(zip(tracking_ids, tracked_boxes)):
            x1, y1, x2, y2 = map(int, bbox)
            class_name = class_map.get(idx,"unknown")
            # --- Speed ---
            speed_kmh = speed_estimator.estimate_speed(tracking_id, (x1, y1, x2, y2), frame_count)
            # Speed limit check
            speed_limit = SPEED_LIMITS.get(class_name, 80)
            # Check overspeed
            if speed_kmh > speed_limit:
                color = (0, 0, 255)  # RED for overspeed
                status = "OVERSPEED"
            else:
                color = (0, 255, 0)  # GREEN normal
                status = "NORMAL"
            # --- Vehicle box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(frame, f"ID {tracking_id} | {class_name} | {speed_kmh:.1f} km/h | {status}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 3)

            # --- Plate detection ---
            plates = detect_and_read_plate(frame, (x1, y1, x2, y2), anpr_model)
            for plate in plates:
                px1, py1, px2, py2 = plate["bbox"]
                text, conf = plate["text"], plate["conf"]

                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 5)
                if text:
                    cv2.putText(frame, f"{text} ({conf:.2f})",
                                (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3)

                # --- Log to CSV ---
                logger.log(frame_count,tracking_id,class_name, text, conf, speed_kmh,status)

        # Write frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("video saved")
