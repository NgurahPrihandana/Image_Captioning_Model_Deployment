from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uuid
from pathlib import Path
from model.model import predict_caption

BASE_DIR = Path(__file__).resolve(strict=True).parent
IMAGEDIR = f"{BASE_DIR}/model/images/"

app = FastAPI()

class TextIn(BaseModel):
  text: str

class PredictionOut(BaseModel):
  captions: str

@app.get("/")
def read_root():
  return {"Health_check" : "OK"}

@app.post("/predict/")
async def upload_and_predict(file: UploadFile = File(...)):
    # Generate a unique filename
    file_id = uuid.uuid4()
    file_name = f"{file_id}.jpg"
    file_path = f"{IMAGEDIR}{file_name}"

    # Read and save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Call the prediction function
    captions = predict_caption(file_name)

    # Return the prediction
    return {"captions": captions}