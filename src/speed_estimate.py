# app/speed_estimator.py
import math
import time

class SpeedEstimator:
    def __init__(self, ppm=10, fps=30):
        """
        Args:
            ppm (float): Pixels per meter (calibration needed for real-world speed).
            fps (int): Video frames per second.
        """
        self.ppm = ppm
        self.fps = fps
        self.prev_positions = {}  # {id: (x, y, frame_time)}

    def estimate_speed(self, object_id, bbox, frame_count):
        """
        Estimate speed of an object.

        Args:
            object_id (int): Unique tracking ID.
            bbox (tuple): (x1, y1, x2, y2) bounding box.
            frame_count (int): Current frame number.

        Returns:
            float: Estimated speed in km/h.
        """
        # Compute center of bbox
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Get previous position
        if object_id in self.prev_positions:
            (px, py, prev_frame) = self.prev_positions[object_id]
            frame_diff = frame_count - prev_frame

            if frame_diff > 0:
                # Pixel distance
                dist_pixels = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                # Convert to meters
                dist_meters = dist_pixels / self.ppm

                # Time in seconds
                time_sec = frame_diff / self.fps

                # Speed in m/s
                speed_m_s = dist_meters / time_sec

                # Convert to km/h
                speed_kmh = speed_m_s * 3.6
            else:
                speed_kmh = 0.0
        else:
            speed_kmh = 0.0

        # Update last position
        self.prev_positions[object_id] = (cx, cy, frame_count)

        return speed_kmh
