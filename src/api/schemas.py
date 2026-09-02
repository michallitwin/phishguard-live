from pydantic import BaseModel, Field
from datetime import datetime


class DomainRequest(BaseModel):
    """Payload for the domain scoring request."""
    domain: str = Field(
        min_length=3,
        max_length=253,
        examples=["paypal-verify-login.tk"],
        description="Domain name without protocol or path",
    )


class ScoreResponse(BaseModel):
    """Result of the phishing classification."""
    domain: str
    prediction: str
    phishing_probability: float = Field(
        ge = 0.0,
        le=1.0,
        description="Phishing probability score",
    )


class HealthResponse(BaseModel):
    """API health status response."""
    status: str = "ok"
    model_loaded: bool


class ModelMetricsResponse(BaseModel):
    """Metrics from the latest model training."""
    model_name: str = Field(
        examples=["Random Forest"],
        description="Name of the deployed machine learning model",
    )
    f1_phishing: float = Field(
        ge=0.0,
        le=1.0,
        description="F1 score for the phishing class on the test set",
    )
    roc_auc: float = Field(
        ge=0.0,
        le=1.0,
        description="ROC-AUC score on the test set",
    )
    trained_at: datetime = Field(
        description="Timestamp of model training in UTC",
    )    
    