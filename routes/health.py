from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db, HealthReading, UserBaseline
from model import predict_risk, rule_based_risk
import numpy as np
import datetime

router = APIRouter()

class HealthData(BaseModel):
    user_id: str = "user_001"
    heart_rate: float
    spo2: float
    steps: int
    sleep_hours: float
    activity_level: str
    language: str = "english"

@router.post("/predict")
def predict(data: HealthData, db: Session = Depends(get_db)):

    # Check if user has a personal baseline
    baseline = db.query(UserBaseline)\
                 .filter(UserBaseline.user_id == data.user_id)\
                 .first()

    if baseline:
        hr_deviation = abs(data.heart_rate - baseline.avg_heart_rate)
        spo2_deviation = baseline.avg_spo2 - data.spo2

        hr_std = max(baseline.hr_std, 5)
        spo2_std = max(baseline.spo2_std, 1)

        hr_abnormal = hr_deviation > (2 * hr_std)
        spo2_abnormal = spo2_deviation > (2 * spo2_std)

        personalized = True

    else:
        hr_abnormal = None
        spo2_abnormal = None
        personalized = False

    # Get prediction
    result = predict_risk(
        data.heart_rate,
        data.spo2,
        data.steps,
        data.sleep_hours,
        data.activity_level,
        hr_abnormal,
        spo2_abnormal
    )

    # Save reading to database
    reading = HealthReading(
        user_id=data.user_id,
        heart_rate=data.heart_rate,
        spo2=data.spo2,
        steps=data.steps,
        sleep_hours=data.sleep_hours,
        activity_level=data.activity_level,
        risk_level=result["risk_level"],
        confidence=result["confidence"]
    )
    db.add(reading)
    db.commit()

    return {
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "message": result["translations"][data.language],
        "personalized": personalized,
        "advice": result["advice"],
        "data_saved": True
    }

@router.post("/baseline/{user_id}")
def set_baseline(user_id: str, db: Session = Depends(get_db)):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=24)

    readings = db.query(HealthReading)\
                 .filter(HealthReading.user_id == user_id)\
                 .filter(HealthReading.timestamp >= since)\
                 .all()

    if len(readings) < 5:
        return {
            "error": f"Not enough readings. Have {len(readings)}, need at least 5.",
            "tip": "Keep sending readings and try again later"
        }

    hrs = [r.heart_rate for r in readings]
    spo2s = [r.spo2 for r in readings]
    steps = [r.steps for r in readings]
    sleeps = [r.sleep_hours for r in readings]

    baseline = db.query(UserBaseline)\
                 .filter(UserBaseline.user_id == user_id)\
                 .first()

    if not baseline:
        baseline = UserBaseline(user_id=user_id)
        db.add(baseline)

    baseline.avg_heart_rate = round(float(np.mean(hrs)), 1)
    baseline.avg_spo2 = round(float(np.mean(spo2s)), 1)
    baseline.avg_steps = round(float(np.mean(steps)), 1)
    baseline.avg_sleep = round(float(np.mean(sleeps)), 1)
    baseline.hr_std = round(float(np.std(hrs)), 2)
    baseline.spo2_std = round(float(np.std(spo2s)), 2)
    baseline.updated_at = datetime.datetime.utcnow()

    db.commit()

    return {
        "message": "Baseline established successfully",
        "user_id": user_id,
        "your_normal_hr": baseline.avg_heart_rate,
        "your_normal_spo2": baseline.avg_spo2,
        "your_normal_sleep": baseline.avg_sleep,
        "your_normal_steps": baseline.avg_steps
    }

@router.get("/history/{user_id}")
def get_history(user_id: str, db: Session = Depends(get_db)):
    readings = db.query(HealthReading)\
                 .filter(HealthReading.user_id == user_id)\
                 .order_by(HealthReading.timestamp.desc())\
                 .limit(20)\
                 .all()

    return [{
        "id": r.id,
        "user_id": r.user_id,
        "heart_rate": round(r.heart_rate, 1),
        "spo2": round(r.spo2, 1),
        "steps": int(round(r.steps)),
        "sleep_hours": round(r.sleep_hours, 1),
        "activity_level": r.activity_level,
        "risk_level": r.risk_level,
        "confidence": round(r.confidence, 1),
        "timestamp": r.timestamp
    } for r in readings]

@router.get("/baseline/{user_id}")
def get_baseline(user_id: str, db: Session = Depends(get_db)):
    baseline = db.query(UserBaseline)\
                 .filter(UserBaseline.user_id == user_id)\
                 .first()

    if not baseline:
        return {"message": "No baseline found for this user"}

    return {
        "user_id": user_id,
        "avg_heart_rate": baseline.avg_heart_rate,
        "avg_spo2": baseline.avg_spo2,
        "avg_sleep": baseline.avg_sleep,
        "avg_steps": baseline.avg_steps,
        "created_at": baseline.created_at
    }

@router.get("/users")
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(HealthReading.user_id)\
              .distinct()\
              .all()

    result = []
    for (user_id,) in users:
        count = db.query(HealthReading)\
                  .filter(HealthReading.user_id == user_id)\
                  .count()

        latest = db.query(HealthReading)\
                   .filter(HealthReading.user_id == user_id)\
                   .order_by(HealthReading.timestamp.desc())\
                   .first()

        baseline = db.query(UserBaseline)\
                     .filter(UserBaseline.user_id == user_id)\
                     .first()

        result.append({
            "user_id": user_id,
            "total_readings": count,
            "latest_risk": latest.risk_level if latest else "N/A",
            "latest_hr": round(latest.heart_rate, 1) if latest else "N/A",
            "latest_spo2": round(latest.spo2, 1) if latest else "N/A",
            "has_baseline": baseline is not None,
            "last_seen": latest.timestamp if latest else None
        })

    return result

@router.delete("/users/{user_id}")
def delete_user(user_id: str, db: Session = Depends(get_db)):
    db.query(HealthReading)\
      .filter(HealthReading.user_id == user_id)\
      .delete()

    db.query(UserBaseline)\
      .filter(UserBaseline.user_id == user_id)\
      .delete()

    db.commit()

    return {"message": f"User {user_id} and all their data deleted successfully"}