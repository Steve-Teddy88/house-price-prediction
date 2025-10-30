"""House Price Prediction API

This module provides the FastAPI endpoints for house price prediction.
It handles model initialization and prediction requests.

Author: ML Engineer
Version: 2.0.0
"""

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.responses import JSONResponse

from ml_core import load_artifacts, predict_price

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model initialization on cold start
try:
    logger.info("Initializing ML model...")
    model, scaler, feature_names = load_artifacts()
    logger.info("ML model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ML model: {str(e)}")
    raise


class PredictionInput(BaseModel):
    """Validation model for prediction input data."""

    features: List[float] = Field(
        ..., description="List of 8 numerical features for prediction"
    )

    @validator("features")
    def validate_features_length(cls, v):
        if len(v) != 8:
            raise ValueError("Must provide exactly 8 features")
        return v


class PredictionResponse(BaseModel):
    """Structured response model for predictions."""

    prediction: float
    confidence: Optional[float]
    feature_importance: Optional[Dict[str, float]]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput) -> PredictionResponse:
    """Predict house price from input features."""
    try:
        if not (model and scaler):
            raise HTTPException(
                status_code=503, detail="Model is not available."
            )

        prediction = predict_price(model, scaler, input_data.features)
        return PredictionResponse(prediction=prediction)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))