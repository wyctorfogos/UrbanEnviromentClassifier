import fastapi
from fastapi import File, UploadFile
import uvicorn
from model.ZeroShotClassifier import UrbanEnviromentClassifier
import numpy as np
import cv2

# Initialize the model classifier
model_classifier = UrbanEnviromentClassifier()

# Create a FastAPI app instance
app = fastapi.FastAPI()

# Define a route using the correct decorator
@app.get("/")
async def get_status():
    return {"message": "Hello!"}

@app.post("/envirementClassifier")
async def env_type_classification(file: UploadFile = File(...)):
    try:
        # Read the file contents
        contents = await file.read()
        # Decode image using OpenCV
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"message": "Invalid image! Please try again."}, 400
        
        # Perform inference
        prediction = model_classifier.infer(image)
        
        return prediction, 200
    except Exception as e:
        print(e)
        return {"message": "An error occurred during classification."}, 500

# Entry point for running the app
if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
