import pandas as pd
import random
from datetime import datetime, timedelta

NUM_PATIENTS = 100
DAYS_PER_PATIENT = 100

rows = []
start_date = datetime(2025, 1, 1)

for pid in range(1, NUM_PATIENTS + 1):
    mood = random.uniform(2.5, 4.0)
    base_hr = random.randint(65, 85)

    for day in range(DAYS_PER_PATIENT):
        date = start_date + timedelta(days=day)

        sleep_hours = max(0, min(10, random.gauss(6.5, 1.2)))
        activity_level = random.randint(1, 10)

        # Mood evolution
        mood += random.uniform(-0.25, 0.25)
        mood = min(max(mood, 0), 5)

        therapy_attended = random.choice([0, 1])

        # Stress inversely related to mood
        stress_level = int(min(10, max(1, 10 - mood + random.uniform(-1, 1))))

        # Heart rate affected by stress
        heart_rate = int(base_hr + stress_level * 2 + random.uniform(-5, 5))

        rows.append({
            "patient_id": pid,
            "date": date.strftime("%Y-%m-%d"),
            "sleep_hours": round(sleep_hours, 2),
            "activity_level": activity_level,
            "mood_score": round(mood, 2),
            "therapy_attended": therapy_attended,
            "heart_rate": heart_rate,
            "stress_level": stress_level
        })

df = pd.DataFrame(rows)
df.to_csv("patient_data.csv", index=False)

print(f"Generated {len(df)} rows → patient_data.csv")
