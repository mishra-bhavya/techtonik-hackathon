import streamlit as st
import pandas as pd

from model import (
    process_data,
    train_model,
    detect_concern,
    has_made_progress,
    generate_insight
)

st.set_page_config(page_title="CARE-AI Dashboard", layout="wide")

st.title("🧠 CARE-AI Nurse Dashboard")
st.caption("AI-Assisted Mental Health Monitoring for Rehabilitation Centers")

# Load data
df = pd.read_csv("data/patient_data.csv")
df = process_data(df)

# Patient selector
st.sidebar.header("👤 Select Patient")
patient_ids = df["patient_id"].unique()
selected_patient = st.sidebar.selectbox("Patient ID", patient_ids)

patient_df = df[df["patient_id"] == selected_patient]

# AI Analysis
model = train_model(patient_df[["sleep_hours", "activity_level", "mood_score", "therapy_attended"]])
risk_score = detect_concern(model, patient_df)
progress = has_made_progress(patient_df)
insight = generate_insight(risk_score, progress)

# Layout
col1, col2, col3 = st.columns(3)

col1.metric("Risk Score", round(risk_score, 3))
col2.metric("Progress", "Yes" if progress else "No")
col3.metric("Status", "High Risk" if "High" in insight else "Stable")

# Alert box
if "High risk" in insight:
    st.error(insight)
elif "Medium risk" in insight:
    st.warning(insight)
else:
    st.success(insight)

# Trends
st.subheader("📈 Behavioral Trends")
st.line_chart(
    patient_df[["sleep_hours", "activity_level", "mood_score"]]
)

# Raw data (optional)
with st.expander("📄 View Patient Records"):
    st.dataframe(patient_df)
