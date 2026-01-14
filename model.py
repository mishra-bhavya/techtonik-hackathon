import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
def load_data():
    return pd.read_csv("data/patient_data.csv")

# Process data
def process_data(df):
    df = df.fillna(df.mean(numeric_only=True))
    return df

# Train AI model
def train_model(df):
    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,
        random_state=42
    )
    model.fit(df)
    return model

# Detect concerning behavior
def detect_concern(model, df):
    scores = model.decision_function(df)
    risk_score = np.mean(scores)

    # Rule-based red flags
    low_sleep = df["sleep_hours"].iloc[-3:].mean() < 4
    low_mood = df["mood_score"].iloc[-3:].mean() < 2

    if low_sleep or low_mood:
        risk_score -= 0.5  # force concern signal

    return risk_score


# Check progress
def has_made_progress(df):
    if len(df) < 2:
        return False
    return df.iloc[-1]["mood_score"] > df.iloc[0]["mood_score"]

# Generate nurse insights
def generate_insight(risk_score, progress):
    if risk_score < -0.2 and not progress:
        return "High risk: Concerning behavior with no progress."
    elif risk_score < -0.2:
        return "Medium risk: Monitor patient closely."
    else:
        return "Stable: Continue monitoring."

# Test run
if __name__ == "__main__":
    df = load_data()
    df = process_data(df)
    model = train_model(df)
    risk = detect_concern(model, df)
    progress = has_made_progress(df)
    insight = generate_insight(risk, progress)

    print("Risk Score:", risk)
    print("Progress:", progress)
    print("Insight:", insight)
