# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os,time
from src.main import process_video   # <-- we'll refactor your main loop into a function

app = FastAPI(title="Vehicle Detection API", version="1.0")

UPLOAD_DIR = "input_videos"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/process_video/")
async def process_video_api(file: UploadFile = File(...)):
    # Save uploaded video
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_video = os.path.join(OUTPUT_DIR, f"processed_{file.filename}")
    output_csv = os.path.join(OUTPUT_DIR, f"vehicle_log_{timestamp}.csv")

    # Run detection pipeline
    process_video(input_path, output_video, output_csv)

    return {
        "message": "Processing Started",
        "output_video": output_video,
        "output_csv": output_csv
    }

@app.get("/download_video/")
async def download_video(path: str):
    return FileResponse(path, media_type="video/mp4")

@app.get("/download_csv/")
async def download_csv(path: str):
    return FileResponse(path, media_type="text/csv")
