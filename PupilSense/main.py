# PupilSense/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil, uuid, os, sys

# Allow importing from make.py in same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from make import main

app = FastAPI()
@app.get("/")
def root():
    return {"status": "ok"}
# Allow requests from local + Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(
    leftImage: UploadFile = File(...),
    rightImage: UploadFile = File(...)
):
    left_path = os.path.join(UPLOAD_DIR, f"left_{uuid.uuid4()}.jpg")
    right_path = os.path.join(UPLOAD_DIR, f"right_{uuid.uuid4()}.jpg")

    with open(left_path, "wb") as f:
        shutil.copyfileobj(leftImage.file, f)

    with open(right_path, "wb") as f:
        shutil.copyfileobj(rightImage.file, f)

    result = main(left_path, right_path)

    os.remove(left_path)
    os.remove(right_path)

    return {"result": result}
