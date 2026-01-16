import streamlit as st
import pandas as pd

from model import (
    process_data,
    train_model,
    detect_concern,
    has_made_progress,
    summarize_changes,
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
model = train_model(patient_df[["sleep_hours", "activity_level", "mood_score", "therapy_attended", "heart_rate", "stress_level"]])
risk_score = detect_concern(model, patient_df)
progress = has_made_progress(patient_df)
summary = summarize_changes(patient_df)
insight = generate_insight(risk_score, progress, summary)

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

# Date range selector
col_filter1, col_filter2 = st.columns([3, 1])
with col_filter1:
    days_to_show = st.slider("Days to Display", min_value=7, max_value=len(patient_df), value=min(30, len(patient_df)), step=1)
with col_filter2:
    st.metric("Total Days", len(patient_df))

# Filter data based on selection
filtered_df = patient_df.tail(days_to_show).reset_index(drop=True)

# Create tabs for different visualizations
trend_tab1, trend_tab2, trend_tab3, trend_tab4 = st.tabs(["📊 Key Metrics", "💓 Health Indicators", "🎯 Therapy & Progress", "📅 Statistics"])

with trend_tab1:
    st.markdown("##### Sleep, Mood & Activity Patterns")
    
    # Normalize activity level for better visualization (scale to 0-10 range)
    viz_data = filtered_df[["sleep_hours", "mood_score"]].copy()
    viz_data["activity_level_scaled"] = (filtered_df["activity_level"] / filtered_df["activity_level"].max()) * 10
    viz_data.columns = ["Sleep Hours", "Mood Score", "Activity (Scaled)"]
    
    st.line_chart(viz_data, use_container_width=True, height=400)
    st.caption("📌 Activity Level is scaled 0-10 for visualization clarity")

with trend_tab2:
    st.markdown("##### Heart Rate & Stress Monitoring")
    
    health_col1, health_col2 = st.columns(2)
    
    with health_col1:
        st.markdown("**💓 Heart Rate (BPM)**")
        st.line_chart(filtered_df["heart_rate"], use_container_width=True, color="#e74c3c", height=300)
        avg_hr = filtered_df["heart_rate"].mean()
        st.caption(f"Average: {avg_hr:.1f} BPM")
    
    with health_col2:
        st.markdown("**😰 Stress Level**")
        st.line_chart(filtered_df["stress_level"], use_container_width=True, color="#9b59b6", height=300)
        avg_stress = filtered_df["stress_level"].mean()
        st.caption(f"Average: {avg_stress:.1f}/10")

with trend_tab3:
    st.markdown("##### Therapy Attendance & Engagement")
    
    therapy_col1, therapy_col2 = st.columns(2)
    
    with therapy_col1:
        st.markdown("**🎭 Therapy Sessions**")
        st.bar_chart(filtered_df["therapy_attended"], use_container_width=True, color="#3498db", height=300)
        attended = filtered_df["therapy_attended"].sum()
        st.caption(f"Attended: {attended}/{len(filtered_df)} sessions")
    
    with therapy_col2:
        st.markdown("**📊 Activity Levels (Steps)**")
        st.area_chart(filtered_df["activity_level"], use_container_width=True, color="#f39c12", height=300)
        avg_activity = filtered_df["activity_level"].mean()
        st.caption(f"Average: {avg_activity:.0f} steps/day")

with trend_tab4:
    st.markdown("##### Summary Statistics (Last {days_to_show} Days)".replace("{days_to_show}", str(days_to_show)))
    
    # Create summary statistics
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        avg_sleep = filtered_df["sleep_hours"].mean()
        min_sleep = filtered_df["sleep_hours"].min()
        max_sleep = filtered_df["sleep_hours"].max()
        st.metric("😴 Avg Sleep", f"{avg_sleep:.1f}h")
        st.caption(f"Range: {min_sleep:.1f}h - {max_sleep:.1f}h")
    
    with summary_cols[1]:
        avg_mood = filtered_df["mood_score"].mean()
        min_mood = filtered_df["mood_score"].min()
        max_mood = filtered_df["mood_score"].max()
        st.metric("😊 Avg Mood", f"{avg_mood:.1f}/5")
        st.caption(f"Range: {min_mood:.1f} - {max_mood:.1f}")
    
    with summary_cols[2]:
        avg_hr = filtered_df["heart_rate"].mean()
        min_hr = filtered_df["heart_rate"].min()
        max_hr = filtered_df["heart_rate"].max()
        st.metric("💓 Avg Heart Rate", f"{avg_hr:.0f}")
        st.caption(f"Range: {min_hr} - {max_hr} BPM")
    
    with summary_cols[3]:
        therapy_rate = (filtered_df["therapy_attended"].sum() / len(filtered_df)) * 100
        st.metric("🎭 Attendance", f"{therapy_rate:.0f}%")
        st.caption(f"{filtered_df['therapy_attended'].sum()}/{len(filtered_df)} sessions")

st.markdown("---")

# Raw Patient Data Section
with st.expander("📄 View Complete Patient Records"):
    st.markdown("##### Detailed Patient Data")
    
    # Display dataframe with better formatting
    display_df = patient_df.copy()
    display_df.columns = ["Patient ID", "Date", "Sleep Hours", "Activity Level", "Mood Score", "Therapy Attended", "Heart Rate", "Stress Level"]
    
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
