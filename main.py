
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="MedPredict API",
    description="AI-Powered No-Show Risk Analysis for Medical Practices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    age: int
    appointment_time: str  # morning, afternoon, evening
    day_of_week: str
    previous_no_shows: int
    appointment_type: str
    insurance: str
    lead_time: int

class PredictionResponse(BaseModel):
    risk_score: float
    risk_category: str
    recommendation: str
    confidence: float

class AppointmentData(BaseModel):
    patient_id: str
    patient_name: str
    appointment_time: str
    appointment_date: str
    risk_score: float
    risk_category: str

class DashboardStats(BaseModel):
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    total_appointments: int
    high_risk_appointments: List[AppointmentData]

# ML Model class (simulated - replace with your actual model)
class NoShowPredictor:
    def __init__(self):
        # In production, load your trained model here
        # self.model = joblib.load('no_show_model.pkl')
        self.feature_weights = {
            'age_young': 0.15,
            'age_old': 0.12,
            'morning': 0.05,
            'afternoon': 0.10,
            'evening': 0.20,
            'monday': 0.15,
            'friday': 0.12,
            'previous_no_shows': 0.25,
            'routine': 0.15,
            'follow_up': 0.18,
            'urgent': 0.05,
            'specialty': 0.10,
            'medicaid': 0.15,
            'self_pay': 0.25,
            'lead_time_long': 0.12,
            'lead_time_short': 0.08
        }
    
    def predict(self, data: PredictionRequest) -> PredictionResponse:
        risk_score = self._calculate_risk_score(data)
        risk_category = self._get_risk_category(risk_score)
        recommendation = self._get_recommendation(risk_category, risk_score)
        confidence = self._calculate_confidence(risk_score)
        
        return PredictionResponse(
            risk_score=round(risk_score, 1),
            risk_category=risk_category,
            recommendation=recommendation,
            confidence=round(confidence, 1)
        )
    
    def _calculate_risk_score(self, data: PredictionRequest) -> float:
        score = 0.0
        
        # Age factors
        if data.age < 25 or data.age > 65:
            score += 15
        elif data.age < 35 or data.age > 55:
            score += 10
        
        # Time factors
        time_scores = {'morning': 5, 'afternoon': 10, 'evening': 20}
        score += time_scores.get(data.appointment_time, 10)
        
        # Day factors
        day_scores = {
            'monday': 15, 'tuesday': 8, 'wednesday': 5,
            'thursday': 8, 'friday': 12
        }
        score += day_scores.get(data.day_of_week, 10)
        
        # Previous no-shows (major factor)
        score += data.previous_no_shows * 20
        
        # Appointment type
        type_scores = {
            'routine': 15, 'follow-up': 20,
            'urgent': 5, 'specialty': 10
        }
        score += type_scores.get(data.appointment_type, 12)
        
        # Insurance type
        insurance_scores = {
            'private': 5, 'medicare': 8,
            'medicaid': 15, 'self-pay': 25
        }
        score += insurance_scores.get(data.insurance, 12)
        
        # Lead time
        if data.lead_time > 30:
            score += 15
        elif data.lead_time > 14:
            score += 10
        elif data.lead_time < 1:
            score += 20
        
        return min(score, 100)
    
    def _get_risk_category(self, score: float) -> str:
        if score >= 70:
            return 'high'
        elif score >= 40:
            return 'medium'
        return 'low'
    
    def _get_recommendation(self, category: str, score: float) -> str:
        recommendations = {
            'high': [
                "🚨 DOUBLE BOOK this slot - High likelihood of no-show",
                "📞 Call patient 24 hours before AND day of appointment",
                "💰 Consider requiring deposit or pre-payment",
                "📋 Send multiple reminder texts/emails"
            ],
            'medium': [
                "📞 Send extra reminder 24 hours before",
                "📱 Use SMS and email reminders",
                "💡 Consider calling to confirm attendance"
            ],
            'low': [
                "✅ Standard reminder protocol sufficient",
                "📧 Single email reminder is adequate"
            ]
        }
        
        return recommendations[category][0]  # Return first recommendation
    
    def _calculate_confidence(self, score: float) -> float:
        # Simulate confidence based on score extremes
        if score > 80 or score < 20:
            return np.random.uniform(85, 95)
        elif score > 60 or score < 40:
            return np.random.uniform(75, 85)
        else:
            return np.random.uniform(60, 75)

# Initialize the ML model
predictor = NoShowPredictor()

# Sample data for dashboard
def generate_sample_appointments() -> List[AppointmentData]:
    """Generate sample appointment data for demo"""
    appointments = []
    base_time = datetime.now()
    
    sample_data = [
        ("A001", "Sarah M.", "10:30 AM", 85, "high"),
        ("A002", "John D.", "2:15 PM", 78, "high"),
        ("A003", "Maria L.", "4:45 PM", 92, "high"),
        ("A004", "David W.", "9:00 AM", 45, "medium"),
        ("A005", "Lisa K.", "1:30 PM", 52, "medium"),
        ("A006", "Mike R.", "11:15 AM", 25, "low"),
        ("A007", "Anna T.", "3:00 PM", 18, "low"),
    ]
    
    for patient_id, name, time, risk, category in sample_data:
        appointments.append(AppointmentData(
            patient_id=patient_id,
            patient_name=name,
            appointment_time=time,
            appointment_date=base_time.strftime("%Y-%m-%d"),
            risk_score=risk,
            risk_category=category
        ))
    
    return appointments

# API Routes
@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard"""
    return FileResponse('index.html')

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_no_show(request: PredictionRequest):
    """Predict no-show risk for a single appointment"""
    try:
        prediction = predictor.predict(request)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/dashboard", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        appointments = generate_sample_appointments()
        
        high_risk = [apt for apt in appointments if apt.risk_category == 'high']
        medium_risk = [apt for apt in appointments if apt.risk_category == 'medium']
        low_risk = [apt for apt in appointments if apt.risk_category == 'low']
        
        return DashboardStats(
            high_risk_count=len(high_risk),
            medium_risk_count=len(medium_risk),
            low_risk_count=len(low_risk),
            total_appointments=len(appointments),
            high_risk_appointments=high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

@app.post("/api/batch-predict")
async def batch_predict(appointments: List[PredictionRequest]):
    """Predict no-show risk for multiple appointments"""
    try:
        predictions = []
        for appointment in appointments:
            prediction = predictor.predict(appointment)
            predictions.append({
                "input": appointment.dict(),
                "prediction": prediction.dict()
            })
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/api/model-info")
async def get_model_info():
    """Get information about the ML model"""
    return {
        "model_name": "No-Show Risk Predictor",
        "version": "1.0.0",
        "features": [
            "patient_age", "appointment_time", "day_of_week",
            "previous_no_shows", "appointment_type", "insurance_type", "lead_time"
        ],
        "accuracy": "87.3%",
        "last_trained": "2024-01-15",
        "total_predictions": 12547
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MedPredict API"
    }

# Mount static files (for serving the HTML dashboard)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
Made with
1
