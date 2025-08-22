# app/logger.py
import csv
import os
from datetime import datetime

class VehicleLogger:
    def __init__(self, filepath="output/vehicle_log.csv"):
        self.filepath = filepath

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create file with header if not exists
        if not os.path.exists(filepath):
            with open(filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "vehicle_id", "plate", "confidence", "speed_kmh"])

    def log(self, vehicle_id, plate, conf, speed_kmh):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, vehicle_id, plate, f"{conf:.2f}", f"{speed_kmh:.2f}"])
