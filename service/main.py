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
    """
    Represents a prediction request.

    Attributes:
        feature_vector (List[float]): The feature vector for the prediction.
        score (Optional[bool], optional): Whether to include the score in the prediction. Defaults to False.
    """
    feature_vector: List[float]
    score: Optional[bool] = False

class PredictionResponse(BaseModel):
    """
    Represents the response for a prediction.

    Attributes:
        is_inlier (int): Indicates whether the prediction is an inlier (1) or an outlier (0).
        anomaly_score (float, optional): The anomaly score of the prediction, if available.
    """

    is_inlier: int
    anomaly_score: Optional[float] = None

class ModelInformation(BaseModel):
    """
    Represents information about a model.
    
    Attributes:
        parameters (dict): A dictionary containing the model parameters.
    """
    parameters: dict

@app.post("/prediction")
async def prediction(request: PredictionRequest):
    """
    Perform prediction based on the given request.

    Args:
        request (PredictionRequest): The prediction request object.

    Returns:
        PredictionResponse: The prediction response object.

    Raises:
        HTTPException: If there is an error during prediction.
    """
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
        # Log the error at ERROR level
        logging.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_information")
async def model_information():
    """
    Retrieves the parameters of the model and returns the ModelInformation object.

    Returns:
        ModelInformation: An object containing the parameters of the model.
    
    Raises:
        HTTPException: If there is an error retrieving the model parameters.
    """
    try:
        params = model.get_params()
        return ModelInformation(parameters=params)
    except Exception as e:
        logging.error(f"Error in model_information: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """
    Returns the health status of the service.
    
    :return: A dictionary containing the health status.
    """
    logging.info("Health check accessed")
    return {"status": "healthy"}

@app.get("/status")
async def status():
    """
    Returns the status of the service.

    :return: A dictionary containing the status of the service.
    """
    logging.info("Status check accessed")
    return {"status": "running"}