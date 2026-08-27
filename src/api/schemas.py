from pydantic import BaseModel


class DomainRequest(BaseModel):
    """Payload for the domain scoring request."""
    domain: str


class ScoreResponse(BaseModel):
    """Result of the phishing classification."""
    domain: str
    prediction: str
    phishing_probability: float


class HealthResponse(BaseModel):
    """API health status response."""
    status: str