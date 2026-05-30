"""
main.py - FastAPI Backend for World Cup AI Predictor
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import predict_match, train_model, FIFA_RANKINGS

app = FastAPI(
    title="⚽ World Cup AI Predictor",
    description="Predict FIFA World Cup 2026 match outcomes using Machine Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    if not os.path.exists("models/match_predictor.pkl"):
        print("🤖 Training model for the first time...")
        train_model()
    else:
        print("✅ Model ready!")

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_ranking: int
    away_ranking: int
    home_titles: int
    away_titles: int
    probabilities: dict
    predicted_winner: str

@app.get("/")
def root():
    return {
        "message": "⚽ World Cup 2026 AI Predictor API",
        "docs": "/docs",
        "endpoints": ["/predict", "/teams", "/health"],
    }

@app.get("/health")
def health_check():
    model_ready = os.path.exists("models/match_predictor.pkl")
    return {"status": "ok", "model_ready": model_ready}

@app.get("/teams")
def get_teams():
    teams = sorted(FIFA_RANKINGS.keys())
    return {"teams": teams, "count": len(teams)}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if request.home_team not in FIFA_RANKINGS:
        raise HTTPException(status_code=400, detail=f"Team '{request.home_team}' not found.")
    if request.away_team not in FIFA_RANKINGS:
        raise HTTPException(status_code=400, detail=f"Team '{request.away_team}' not found.")
    if request.home_team == request.away_team:
        raise HTTPException(status_code=400, detail="Teams must be different.")

    try:
        result = predict_match(request.home_team, request.away_team)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/{home_team}/{away_team}")
def predict_get(home_team: str, away_team: str):
    request = PredictionRequest(home_team=home_team, away_team=away_team)
    return predict(request)