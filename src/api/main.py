from fastapi import FastAPI

from src.api.schemas import DomainRequest, ScoreResponse, HealthResponse
from src.ml.predict import load_artifacts, score_domain
import uvicorn


app = FastAPI(
    title="PhishGuard Live API",
    description="Detects phishing domains using a trained ML model.",
    version="0.1.0",
)

model, le = load_artifacts()

@app.get("/health", response_model=HealthResponse)
def healthy() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None and le is not None,
        }


@app.post("/api/score", response_model=ScoreResponse)
def score(request: DomainRequest) -> dict:
    result = score_domain(request.domain, model, le)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    