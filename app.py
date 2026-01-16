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
selected_patient = st.sidebar.selectbox("Patient ID", patient_ids, key="patient_selector")

patient_df = df[df["patient_id"] == selected_patient].reset_index(drop=True)

# AI Analysis
model = train_model(patient_df[["sleep_hours", "activity_level", "mood_score", "therapy_attended"]])
risk_score = detect_concern(model, patient_df)
progress = has_made_progress(patient_df)
insight = generate_insight(risk_score, progress)

# Main Dashboard Layout
st.markdown("---")
st.subheader(f"📊 Patient Overview: {selected_patient}")

# Key Metrics Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Risk Score", 
        value=f"{round(risk_score, 3)}",
        delta=None
    )

with col2:
    progress_emoji = "✅" if progress else "⏳"
    st.metric(
        label="📈 Progress Made", 
        value=f"{progress_emoji} {'Yes' if progress else 'No'}"
    )

with col3:
    status = "High Risk" if "High" in insight else "Stable"
    status_color = "🔴" if "High" in insight else "🟢"
    st.metric(
        label="🏥 Status", 
        value=f"{status_color} {status}"
    )

with col4:
    total_records = len(patient_df)
    st.metric(
        label="📋 Total Records", 
        value=total_records
    )

st.markdown("---")

# Alert box
if "High risk" in insight:
    st.error(insight)
elif "Medium risk" in insight:
    st.warning(insight)
else:
    st.success(insight)

st.markdown("---")

# Behavioral Trends Section
st.subheader("📈 Behavioral Trends Over Time")

# Create tabs for different visualizations
trend_tab1, trend_tab2, trend_tab3 = st.tabs(["📊 Combined View", "📉 Individual Metrics", "📅 Daily Summary"])

with trend_tab1:
    st.markdown("##### All Behavioral Metrics")
    chart_data = patient_df[["sleep_hours", "activity_level", "mood_score"]].copy()
    chart_data.columns = ["Sleep Hours", "Activity Level", "Mood Score"]
    st.line_chart(chart_data, use_container_width=True)

with trend_tab2:
    # Individual metric charts
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown("##### 😴 Sleep Hours")
        st.line_chart(patient_df["sleep_hours"], use_container_width=True, color="#1f77b4")
        
        st.markdown("##### 😊 Mood Score")
        st.line_chart(patient_df["mood_score"], use_container_width=True, color="#2ca02c")
    
    with metric_col2:
        st.markdown("##### 🏃 Activity Level")
        st.line_chart(patient_df["activity_level"], use_container_width=True, color="#ff7f0e")
        
        st.markdown("##### 🎭 Therapy Attendance")
        st.bar_chart(patient_df["therapy_attended"], use_container_width=True, color="#d62728")

with trend_tab3:
    st.markdown("##### Daily Summary Statistics")
    
    # Create summary statistics
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        avg_sleep = patient_df["sleep_hours"].mean()
        st.metric("Avg Sleep", f"{avg_sleep:.1f}h")
    
    with summary_cols[1]:
        avg_activity = patient_df["activity_level"].mean()
        st.metric("Avg Activity", f"{avg_activity:.0f}")
    
    with summary_cols[2]:
        avg_mood = patient_df["mood_score"].mean()
        st.metric("Avg Mood", f"{avg_mood:.1f}/5")
    
    with summary_cols[3]:
        therapy_rate = (patient_df["therapy_attended"].sum() / len(patient_df)) * 100
        st.metric("Therapy Rate", f"{therapy_rate:.0f}%")

st.markdown("---")

# Raw Patient Data Section
with st.expander("📄 View Complete Patient Records"):
    st.markdown("##### Detailed Patient Data")
    
    # Display dataframe with better formatting
    display_df = patient_df.copy()
    display_df.columns = ["Patient ID", "Sleep Hours", "Activity Level", "Mood Score", "Therapy Attended"]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button for data export
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Patient Data as CSV",
        data=csv,
        file_name=f"{selected_patient}_data.csv",
        mime="text/csv"
    )
