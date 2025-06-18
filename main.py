from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = FastAPI(title="MedPredict API", description="No-Show Prediction for Medical Practices")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for CSS, JS, images)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class PredictionRequest(BaseModel):
    age: int
    appointment_type: str
    day_of_week: str
    time_slot: str
    previous_no_shows: int
    lead_time: int

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str

class AppointmentResponse(BaseModel):
    id: int
    patient: str
    time: str
    type: str
    risk: str
    probability: float

# Load or create model (placeholder)
def load_model():
    try:
        model = joblib.load("models/no_show_model.pkl")
        return model
    except:
        # Create a dummy model for demo purposes
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Train with dummy data
        X_dummy = np.random.rand(1000, 6)
        y_dummy = np.random.choice([0, 1], 1000)
        model.fit(X_dummy, y_dummy)
        return model

model = load_model()

# Feature encoding functions
def encode_features(data: PredictionRequest):
    """Convert categorical features to numerical"""
    features = []
    
    # Age (numerical)
    features.append(data.age)
    
    # Appointment type encoding
    apt_type_map = {
        "routine": 0, "followup": 1, "specialist": 2, 
        "emergency": 3, "procedure": 4
    }
    features.append(apt_type_map.get(data.appointment_type, 0))
    
    # Day of week encoding
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    features.append(day_map.get(data.day_of_week, 0))
    
    # Time slot encoding
    time_map = {"morning": 0, "afternoon": 1, "evening": 2}
    features.append(time_map.get(data.time_slot, 0))
    
    # Previous no-shows (numerical)
    features.append(data.previous_no_shows)
    
    # Lead time (numerical)
    features.append(data.lead_time)
    
    return np.array(features).reshape(1, -1)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    """Serve the main dashboard"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>MedPredict API</h1>
                <p>Dashboard not found. Please add index.html to static/ directory.</p>
                <p>API is running at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "MedPredict API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_no_show(request: PredictionRequest):
    """Predict no-show probability for a patient"""
    try:
        # Encode features
        features = encode_features(request)
        
        # Make prediction
        prediction_proba = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        # Get confidence (probability of predicted class)
        confidence = max(prediction_proba)
        
        # Determine prediction and risk level
        no_show_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
        
        if prediction == 1 or no_show_prob > 0.6:
            pred_label = "no-show"
            risk_level = "high"
        elif no_show_prob > 0.4:
            pred_label = "uncertain"
            risk_level = "medium"
        else:
            pred_label = "show"
            risk_level = "low"
        
        return PredictionResponse(
            prediction=pred_label,
            confidence=confidence,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/appointments/today")
async def get_today_appointments():
    """Get today's appointments with risk assessment"""
    # This would typically connect to your database
    # For now, returning sample data
    sample_appointments = [
        {"id": 1, "patient": "John S.", "time": "09:00 AM", "type": "Routine Checkup", "risk": "low", "probability": 0.15},
        {"id": 2, "patient": "Sarah J.", "time": "10:30 AM", "type": "Follow-up", "risk": "high", "probability": 0.78},
        {"id": 3, "patient": "Michael B.", "time": "02:00 PM", "type": "Specialist Consultation", "risk": "medium", "probability": 0.45},
        {"id": 4, "patient": "Emily D.", "time": "03:30 PM", "type": "Procedure", "risk": "low", "probability": 0.22},
        {"id": 5, "patient": "Robert W.", "time": "04:45 PM", "type": "Emergency", "risk": "high", "probability": 0.82}
    ]
    return sample_appointments

@app.get("/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    # This would typically aggregate from your database
    return {
        "total_appointments": 25,
        "high_risk": 8,
        "medium_risk": 7,
        "low_risk": 10,
        "accuracy": 0.87
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
