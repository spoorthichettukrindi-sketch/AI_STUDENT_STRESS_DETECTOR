import streamlit as st
import joblib
import pandas as pd

model = joblib.load("stress_model.joblib")

st.set_page_config(page_title="AI Student Stress Detector")
st.title("AI Student Stress Detector")
st.write("Enter student details to detect stress level")

sleep = st.slider("Sleep Hours", 3, 10, 6)
study = st.slider("Study Hours", 1, 12, 5)
screen = st.slider("Screen Time Hours", 1, 12, 6)
activity = st.slider("Physical Activity Days", 0, 7, 3)
workload = st.slider("Academic Workload", 1, 10, 6)
exam_freq = st.slider("Exam Frequency", 1, 10, 5)
mood = st.slider("Mood Score", 1, 10, 5)
support = st.slider("Social Support Score", 1, 10, 5)
attendance = st.slider("Attendance Percent", 50, 100, 80)

if st.button("Detect Stress"):
    data = {
        "sleep_hours": sleep,
        "study_hours": study,
        "screen_time_hours": screen,
        "physical_activity_days": activity,
        "academic_workload": workload,
        "exam_frequency": exam_freq,
        "mood_score": mood,
        "social_support_score": support,
        "attendance_percent": attendance
    }
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]

    if pred == "Low":
        st.success(f"Predicted Stress: {pred} - Student is relaxed!")
    elif pred == "Moderate":
        st.warning(f"Predicted Stress: {pred} - Needs attention")
    else:
        st.error(f"Predicted Stress: {pred} - High stress, counselling needed")