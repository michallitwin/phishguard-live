from pydantic import BaseModel, Field


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