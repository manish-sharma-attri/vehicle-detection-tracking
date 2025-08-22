# 🚗 Automatic Vehicle Detection,Speed Estimation, Tracking & ANPR  

An **AI-powered system** for detecting vehicles, tracking their movement, recognizing license plates, and estimating vehicle speed using **YOLOv11, DeepSORT, and EasyOCR**.  

---

## 📌 Features  
- 🔹 **Vehicle Detection** using YOLOv11  
- 🔹 **Multi-Object Tracking** with DeepSORT  
- 🔹 **Number Plate Detection (ANPR)** with YOLO  
- 🔹 **License Plate Recognition (OCR)** using EasyOCR  
- 🔹 **Speed Estimation** of moving vehicles  
- 🔹 **Output Video** with bounding boxes, IDs, speed, and number plates  

---

## 🛠️ Tech Stack  
- **Python 3.10+**  
- [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics)  
- [DeepSORT-Realtime](https://github.com/levan92/deep-sort-realtime)  
- [OpenCV](https://opencv.org/)  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)  

---

## 📂 Project Structure  

```
vehicle_number_plate_detection/
│── detector.py        # Vehicle detection (YOLOv11)
│── tracker.py         # DeepSORT tracker
│── utils.py           # Helper functions (OCR, ANPR, speed calculation)
│── plate_reader.py    # EasyOCR number plate reader
│── speed_estimator.py # Speed calculation functions
│── main.py            # Main pipeline script
│── models/            # YOLO + ANPR model files
│── input/             # Input videos
│── output/            # Processed video output
│── requirements.txt   # Python dependencies
│── README.md          # Project documentation
```

---

## ⚙️ Installation  

1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/vehicle_number_plate_detection.git
cd vehicle_number_plate_detection
```

2️⃣ Create a virtual environment  
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

4️⃣ Place your models inside `models/`  
- `yolo11m.pt` (vehicle detection)  
- `ANPR.pt` (number plate detection)  

---

## ▶️ Usage  

Run the main pipeline:  
```bash
python main.py
```

- Input video: `input/testing_video.mp4`  
- Output video: `output/detection_tracking.mp4`  

---

## 🎯 Applications  
- Traffic surveillance  
- Law enforcement (overspeed detection, plate logging)  
- Toll booth automation  
- Smart city traffic management  

---

## 📸 Sample Output
![detection_tracking_1](https://github.com/user-attachments/assets/1f2cd52d-a221-4283-ad64-49240b80754b)

---

## 🤝 Contributing  
Contributions are welcome! Feel free to fork this repo, create a branch, and submit a PR.  

---

## 📜 License  
This project is licensed under the **MIT License**.  
