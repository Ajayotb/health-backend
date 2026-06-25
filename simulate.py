import requests
import pandas as pd
import time
import random

# Load your cleaned dataset
df = pd.read_csv(r'C:\Users\DELL 7420\cleaned_health_data.csv')

# Sample 50 rows for demo
df_sample = df.sample(50, random_state=42).reset_index(drop=True)

# Map activity level back to text
activity_map = {
    0: 'sedentary',
    1: 'active', 
    2: 'highly active'
}

# List of demo users
users = ['demo_001', 'demo_002', 'demo_003', 'demo_004', 'demo_005']

# Languages to cycle through
languages = ['english', 'yoruba', 'hausa', 'igbo', 'english']

print("=" * 50)
print("AI Health Monitoring — Live Simulation")
print("=" * 50)
print(f"Sending {len(df_sample)} readings...")
print("Open your dashboard at http://localhost:3000")
print("=" * 50)

for i, row in df_sample.iterrows():
    user_id = users[i % len(users)]
    language = languages[i % len(languages)]

    # Get activity level
    activity_val = row['Activity Level']
    if isinstance(activity_val, (int, float)):
        activity = activity_map.get(int(activity_val), 'active')
    else:
        activity = str(activity_val).lower()
        if activity not in ['sedentary', 'active', 'highly active']:
            activity = 'active'

    payload = {
        "user_id": user_id,
        "heart_rate": float(row['Heart Rate (BPM)']),
        "spo2": float(row['Blood Oxygen Level (%)']),
        "steps": int(row['Step Count']),
        "sleep_hours": float(row['Sleep Duration (hours)']),
        "activity_level": activity,
        "language": language
    }

    try:
        response = requests.post(
            'http://127.0.0.1:8000/api/health/predict',
            json=payload,
            timeout=5
        )
        result = response.json()

        risk = result.get('risk_level', 'Unknown')
        confidence = result.get('confidence', 0)
        message = result.get('message', '')

        # Color code the output
        if risk == 'Normal':
            symbol = '✅'
        elif risk == 'Mild Risk':
            symbol = '⚠️'
        else:
            symbol = '🚨'

        print(f"[{i+1:02d}/50] User: {user_id} | HR: {payload['heart_rate']:.0f} | "
              f"SpO₂: {payload['spo2']:.0f}% | "
              f"Risk: {symbol} {risk} ({confidence:.1f}%)")
        print(f"       Message: {message}")
        print()

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure server is running.")
        break
    except Exception as e:
        print(f"Error: {e}")

    # Wait 3 seconds between readings
    time.sleep(3)

print("=" * 50)
print("Simulation complete!")
print("=" * 50)