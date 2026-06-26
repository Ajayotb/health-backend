import joblib
import numpy as np
import os
import base64
import pickle

def download_model():
    """Download model from Google Drive on startup"""
    model_path = 'health_model.pkl'

    # If already exists locally skip download
    if os.path.exists(model_path):
        print("✅ Model file found locally")
        return model_path

    try:
        import urllib.request
        file_id = "165eYKChidh_V_Q1v2yToDSa3JATPE2Qj"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print("⬇️ Downloading model from Google Drive...")
        urllib.request.urlretrieve(url, model_path)
        print("✅ Model downloaded successfully")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None

def load_scaler():
    """Load scaler from environment variable"""
    scaler_b64 = os.environ.get('SCALER_B64')
    if scaler_b64:
        try:
            scaler_bytes = base64.b64decode(scaler_b64.encode('utf-8'))
            scaler = pickle.loads(scaler_bytes)
            print("✅ Scaler loaded from environment variable")
            return scaler
        except Exception as e:
            print(f"❌ Failed to load scaler from env: {e}")

    # Fall back to local file
    if os.path.exists('scaler.pkl'):
        print("✅ Scaler loaded from local file")
        return joblib.load('scaler.pkl')

    print("❌ No scaler found")
    return None

# Load model and scaler on startup
model_path = download_model()
MODEL_AVAILABLE = False

try:
    if model_path:
        model = joblib.load(model_path)
        scaler = load_scaler()
        if model is not None and scaler is not None:
            MODEL_AVAILABLE = True
            print("✅ All models loaded successfully")
        else:
            model = None
            scaler = None
    else:
        model = None
        scaler = None
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None

RISK_LABELS = {
    0: "Normal",
    1: "Mild Risk",
    2: "Elevated Risk"
}

TRANSLATIONS = {
    "Normal": {
        "english": "Normal - Your health indicators look good",
        "yoruba": "Deede - Awọn afihan ilera rẹ dara",
        "hausa": "Al'ada - Alamomin lafiyar ku suna da kyau",
        "igbo": "Ọ dị mma - Ihe ngosi ahụike gị dị mma"
    },
    "Mild Risk": {
        "english": "Mild Risk - Monitor your health closely",
        "yoruba": "Ewu kekere - Ṣọra fun ilera rẹ",
        "hausa": "Haɗarin Ƙanƙanta - Kula da lafiyar ku",
        "igbo": "Ihe ize ndụ nta - Lelee ahụike gị nkenke"
    },
    "Elevated Risk": {
        "english": "Elevated Risk - Please consult a healthcare provider",
        "yoruba": "Ewu giga - Jọwọ kan si olupese ilera",
        "hausa": "Haɗarin Gaske - Don Allah tuntuɓi mai ba da lafiya",
        "igbo": "Ihe ize ndụ dị elu - Biko kpọtụrụ onye ọrụ ahụike"
    }
}


def generate_advice(heart_rate, spo2, steps, sleep_hours, activity_level, risk_level):
    advice = []
    tips = []

    # Heart rate advice
    if heart_rate > 100:
        advice.append("Your heart rate is elevated above the normal range of 60-100 BPM.")
        tips.append("Try relaxation techniques such as deep breathing or meditation to lower your heart rate.")
        tips.append("Avoid caffeine and strenuous activity until your heart rate normalizes.")
    elif heart_rate < 60:
        advice.append("Your heart rate is below the normal range.")
        tips.append("If you are not an athlete, consult a healthcare provider about your low heart rate.")
    else:
        advice.append("Your heart rate is within the normal range.")

    # SpO2 advice
    if spo2 < 92:
        advice.append("Your blood oxygen level is critically low and requires immediate attention.")
        tips.append("Seek medical attention immediately if you feel short of breath or dizzy.")
    elif spo2 < 95:
        advice.append("Your blood oxygen level is below the healthy range of 95-100%.")
        tips.append("Practice deep breathing exercises to improve oxygen intake.")
        tips.append("Ensure you are in a well-ventilated environment.")
    elif spo2 < 97:
        advice.append("Your blood oxygen level is slightly below the optimal range.")
        tips.append("Try some light breathing exercises and stay in a well-ventilated area.")
    else:
        advice.append("Your blood oxygen level is healthy.")

    # Sleep advice
    if sleep_hours < 5:
        advice.append(f"You are only getting {sleep_hours} hours of sleep which is critically low.")
        tips.append("Aim for at least 7-9 hours of sleep per night for optimal health.")
        tips.append("Establish a consistent sleep schedule by going to bed at the same time each night.")
        tips.append("Avoid screens and caffeine at least one hour before bedtime.")
    elif sleep_hours < 6:
        advice.append(f"You are getting {sleep_hours} hours of sleep which is below the recommended amount.")
        tips.append("Try to get at least 7 hours of sleep. Consider going to bed 30-60 minutes earlier.")
    elif sleep_hours > 10:
        advice.append(f"You are sleeping {sleep_hours} hours which is more than recommended.")
        tips.append("Oversleeping can sometimes indicate underlying health issues. Aim for 7-9 hours.")
    else:
        advice.append(f"Your sleep duration of {sleep_hours} hours is within the healthy range.")

    # Steps advice
    if steps < 2000:
        advice.append(f"Your step count of {steps:,} is very low.")
        tips.append("Try to take short walks throughout the day. Even 10-minute walks make a difference.")
        tips.append("Aim for at least 7,000-10,000 steps per day for cardiovascular health.")
    elif steps < 5000:
        advice.append(f"Your step count of {steps:,} is below the recommended daily target.")
        tips.append("Try to increase your daily movement. Take the stairs instead of the lift.")
        tips.append("A 30-minute walk can add approximately 3,000-4,000 steps to your daily count.")
    elif steps >= 10000:
        advice.append(f"Excellent! Your step count of {steps:,} exceeds the recommended daily target.")
    else:
        advice.append(f"Your step count of {steps:,} is good. Keep it up!")

    # Activity level advice
    if activity_level.lower() == 'sedentary':
        tips.append("Consider incorporating light physical activity into your daily routine such as walking or stretching.")

    # Overall summary
    if risk_level == "Normal":
        summary = "Your overall health indicators look good. Keep maintaining your healthy lifestyle."
    elif risk_level == "Mild Risk":
        summary = "Some of your health indicators need attention. Follow the tips below to improve your health."
    else:
        summary = "Several health indicators are concerning. Please consider consulting a healthcare provider and follow the recommendations below."

    return {
        "summary": summary,
        "observations": advice,
        "tips": tips
    }


def rule_based_risk(heart_rate, spo2, steps, sleep_hours,
                    hr_abnormal=False, spo2_abnormal=False,
                    has_baseline=False):
    risk_points = 0

    # Heart rate rules
    if has_baseline:
        if hr_abnormal:
            risk_points += 2
    else:
        if heart_rate < 60 or heart_rate > 100:
            risk_points += 2
        elif heart_rate < 65 or heart_rate > 90:
            risk_points += 1

    # Blood oxygen rules
    if has_baseline:
        if spo2_abnormal:
            risk_points += 2
    else:
        if spo2 < 92:
            risk_points += 3
        elif spo2 < 95:
            risk_points += 2
        elif spo2 < 97:
            risk_points += 1

    # Sleep rules
    if sleep_hours < 5 or sleep_hours > 10:
        risk_points += 2
    elif sleep_hours < 6 or sleep_hours > 9:
        risk_points += 1

    # Steps
    if steps < 2000:
        risk_points += 1

    if risk_points >= 4:
        return 2
    elif risk_points >= 2:
        return 1
    else:
        return 0


def predict_risk(heart_rate, spo2, steps, sleep_hours,
                 activity_level, hr_abnormal=None, spo2_abnormal=None):

    activity_map = {
        'sedentary': 0,
        'active': 1,
        'highly active': 2
    }
    activity_encoded = activity_map.get(activity_level.lower(), 1)

    # Determine if we have a baseline
    has_baseline = hr_abnormal is not None and spo2_abnormal is not None

    # Default to False if no baseline
    if hr_abnormal is None:
        hr_abnormal = False
    if spo2_abnormal is None:
        spo2_abnormal = False

    # Rule-based prediction
    rule_prediction = rule_based_risk(
        heart_rate, spo2, steps, sleep_hours,
        hr_abnormal, spo2_abnormal, has_baseline
    )

    # ML model prediction
    if MODEL_AVAILABLE and model is not None and scaler is not None:
        features = np.array([[heart_rate, spo2, steps, sleep_hours, activity_encoded]])
        features_scaled = scaler.transform(features)
        ml_prediction = model.predict(features_scaled)[0]
        ml_probability = model.predict_proba(features_scaled)[0]
        confidence_score = float(max(ml_probability)) * 100
    else:
        ml_prediction = rule_prediction
        confidence_score = 85.0

    # Smart combining logic
    if has_baseline and rule_prediction == 0:
        final_prediction = 0
    elif rule_prediction == 0 and confidence_score < 60:
        final_prediction = 0
    elif rule_prediction == 2:
        final_prediction = 2
    elif rule_prediction == 1:
        final_prediction = 1
    else:
        final_prediction = max(rule_prediction, int(ml_prediction))

    risk_label = RISK_LABELS[final_prediction]

    return {
        "risk_level": risk_label,
        "confidence": round(confidence_score, 2),
        "translations": TRANSLATIONS[risk_label],
        "advice": generate_advice(
            heart_rate, spo2, steps, sleep_hours,
            activity_level, risk_label
        )
    }