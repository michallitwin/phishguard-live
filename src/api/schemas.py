from pydantic import BaseModel


class DomainRequest(BaseModel):
    domain: str


class ScoreResponse(BaseModel):
    domain: str
    prediction: str
    phishing_probability: float

class HealthResponse(BaseModel):
    status: str
