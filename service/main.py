import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler("/app/logs/logs.txt"),
                        logging.StreamHandler()
                    ])

# Load the trained Isolation Forest model
logging.info("Loading the model...")
model = joblib.load("/app/models/model.pkl")

# Create the FastAPI app
logging.info("Creating the FastAPI app...")
app = FastAPI()

class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: Optional[bool] = False

class PredictionResponse(BaseModel):
    is_inlier: int
    anomaly_score: Optional[float] = None

@app.post("/prediction")
async def prediction(request: PredictionRequest):
    try:
        feature_vector = np.array([request.feature_vector])
        prediction = model.predict(feature_vector)[0]
        response = {"is_inlier": int(prediction)}

        if request.score:
            anomaly_score = model.decision_function(feature_vector)[0]
            response["anomaly_score"] = anomaly_score
            logging.info("Prediction request processed successfully.")
            return PredictionResponse(**response)
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ModelInformation(BaseModel):
    parameters: dict

@app.get("/model_information")
async def model_information():
    try:
        params = model.get_params()
        return ModelInformation(parameters=params)
    except Exception as e:
        logging.error(f"Error in model_information: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    logging.info("Health check accessed")
    return {"status": "healthy"}

@app.get("/status")
async def status():
    logging.info("Status check accessed")
    return {"status": "running"}