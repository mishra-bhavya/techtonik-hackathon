import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ----------------------------------
# Data Loading & Processing
# ----------------------------------

def load_data(path="data/patient_data.csv"):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"])
    return df


def process_data(df):
    df = df.sort_values("date")
    df = df.fillna(df.mean(numeric_only=True))
    return df


# ----------------------------------
# AI Model Training
# ----------------------------------

def train_model(df):
    features = df[
        [
            "sleep_hours",
            "activity_level",
            "mood_score",
            "therapy_attended",
            "heart_rate",
            "stress_level",
        ]
    ]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.12,
        random_state=42,
    )
    model.fit(features)
    return model


# ----------------------------------
# Risk Detection (Hybrid AI + Rules)
# ----------------------------------

def detect_concern(model, df):
    features = df[
        [
            "sleep_hours",
            "activity_level",
            "mood_score",
            "therapy_attended",
            "heart_rate",
            "stress_level",
        ]
    ]

    # AI anomaly score
    scores = model.decision_function(features)
    risk_score = np.mean(scores)

    # Patient baselines
    baseline_sleep = df["sleep_hours"].mean()
    baseline_mood = df["mood_score"].mean()
    baseline_hr = df["heart_rate"].mean()

    recent_sleep = df["sleep_hours"].iloc[-3:].mean()
    recent_mood = df["mood_score"].iloc[-3:].mean()
    recent_hr = df["heart_rate"].iloc[-3:].mean()
    recent_stress = df["stress_level"].iloc[-3:].mean()
    recent_activity = df["activity_level"].iloc[-3:].mean()

    # 🔑 Escalation signals (MOOD IS CONTEXT ONLY)
    if recent_sleep < 0.7 * baseline_sleep:
        risk_score -= 0.30

    if recent_hr > 1.15 * baseline_hr:
        risk_score -= 0.25

    if recent_stress > 7:
        risk_score -= 0.30

    if recent_activity < df["activity_level"].quantile(0.25):
        risk_score -= 0.15

    # Mood contributes weakly (never escalates alone)
    if recent_mood < 0.7 * baseline_mood:
        risk_score -= 0.10

    return risk_score


# ----------------------------------
# Progress Tracking
# ----------------------------------

def has_made_progress(df):
    if len(df) < 8:
        return False

    recent = df.tail(4)["mood_score"].mean()
    previous = df.iloc[-8:-4]["mood_score"].mean()

    return recent > previous


# ----------------------------------
# Explainable Insight Generation
# ----------------------------------

def summarize_changes(df):
    recent = df.tail(3)

    return {
        "sleep_drop": recent["sleep_hours"].mean() < 4,
        "low_activity": recent["activity_level"].mean()
        < df["activity_level"].quantile(0.25),
        "low_mood": recent["mood_score"].mean() < 2,
        "high_hr": recent["heart_rate"].mean() > 100,
        "high_stress": recent["stress_level"].mean() > 7,
        "missed_therapy": recent["therapy_attended"].sum() < 2,
    }


def generate_insight(risk_score, progress, summary):
    reasons = []

    # 🔑 Escalation flags (EXCLUDES MOOD)

    hard_escalation_flags = (
    summary["high_hr"]
    + summary["high_stress"]
    + summary["missed_therapy"]
)


    if summary["sleep_drop"]:
        reasons.append("reduced sleep")
    if summary["low_activity"]:
        reasons.append("low activity levels")
    if summary["low_mood"]:
        reasons.append("persistently low mood")
    if summary["high_hr"]:
        reasons.append("elevated heart rate")
    if summary["high_stress"]:
        reasons.append("high stress levels")
    if summary["missed_therapy"]:
        reasons.append("missed therapy sessions")

    reason_text = ", ".join(reasons) if reasons else "no significant behavioral changes"

    # ✅ HIGH RISK ONLY IF REAL ESCALATION EXISTS
    if risk_score < -0.2 and hard_escalation_flags >= 1 and not progress:

        return (
            f"High risk: Patient shows {reason_text}. "
            f"Immediate nurse review recommended."
        )

    elif risk_score < -0.2:
        return (
            f"Medium risk: Behavioral changes detected ({reason_text}). "
            f"Close monitoring advised."
        )

    else:
        return (
            "Stable: No concerning behavioral changes detected. "
            "Continue routine monitoring."
        )


# ----------------------------------
# ALL PATIENTS ANALYSIS
# ----------------------------------

def analyze_all_patients(df):
    results = []

    for patient_id in df["patient_id"].unique():
        patient_df = df[df["patient_id"] == patient_id]

        if len(patient_df) < 5:
            continue

        model = train_model(patient_df)
        risk = detect_concern(model, patient_df)
        progress = has_made_progress(patient_df)
        summary = summarize_changes(patient_df)
        insight = generate_insight(risk, progress, summary)

        results.append(
            {
                "patient_id": patient_id,
                "risk_score": round(risk, 3),
                "progress": progress,
                "insight": insight,
            }
        )

    return pd.DataFrame(results)


# ----------------------------------
# STANDALONE RUN: HIGH-RISK TRIAGE
# ----------------------------------

if __name__ == "__main__":
    df = load_data()
    df = process_data(df)

    results_df = analyze_all_patients(df)

    high_risk_df = results_df[
        results_df["insight"].str.startswith("High risk")
    ].sort_values("risk_score")

    print("\n=== HIGH-RISK PATIENTS (IMMEDIATE ATTENTION REQUIRED) ===\n")
    print(high_risk_df)

    print(f"\nTotal High-Risk Patients: {len(high_risk_df)}")
