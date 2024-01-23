import logging, time
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
from prometheus_client import generate_latest, make_asgi_app, Counter, Histogram, CONTENT_TYPE_LATEST


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

# Create Prometheu' metrics
prediction_counter = Counter("prediction_counter", "Number of prediction endpoint calls")
model_information_counter = Counter("model_information_counter", "Number of model_information endpoint calls")
predictions_output = Histogram('predictions_output', 'Histogram for tracking the model predictions response times', buckets=(0.1, 0.5, 1, 5, 10, float('inf')))

# Histogram to track the score sample predictions
predictions_scores = Histogram('predictions_scores', 'Histogram for tracking the scores of the model predictions')

# Histogram to track the function's latency
predictions_latency = Histogram('predictions_latency', 'Histogram for tracking the latency of the prediction function', buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')))


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
    prediction_counter.inc()
    start_time = time.time()  # Start time measurement

    try:
        feature_vector = np.array([request.feature_vector])
        prediction = model.predict(feature_vector)[0]
        response = {"is_inlier": int(prediction)}

        if request.score:
            anomaly_score = model.decision_function(feature_vector)[0]
            response["anomaly_score"] = anomaly_score

            # Record the score in the histogram
            predictions_scores.observe(anomaly_score)

        logging.info("Prediction request processed successfully.")

        # Record the prediction response time
        predictions_output.observe(time.time() - start_time)

        # Calculate and record the function's latency
        predictions_latency.observe(time.time() - start_time)  # Record latency

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
    model_information_counter.inc()
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

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Create the ASGI app using make_asgi_app()
app.mount("/metrics", make_asgi_app())
