import streamlit as st
import pandas as pd
import numpy as np
# import pickle
import joblib

# -------------------------------------------------------------
# Load Trained Model Pipeline (Preprocessor + XGBoost)
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("stress_drop_model.pkl")

'''
def load_model():
    with open("stress_drop_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model
'''

model = load_model()

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="Stress-Drop Prediction App",
    layout="centered",
)

st.title("🧘‍♂️ Cruise Passenger Stress-Drop Prediction")
st.subheader("ML-powered Spa Treatment Effectiveness Estimator")

st.write("Enter passenger wellness and lifestyle details below:")

# -------------------------------------------------------------
# Input Form
# -------------------------------------------------------------
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.selectbox("Age Group", 
        ["20-30", "31-40", "41-50", "51-60", "61-70"])

    occupation = st.selectbox(
        "Occupation",
        ["Software Engineer", "Doctor", "Teacher", "Nurse", "Lawyer", "Salesperson", "Accountant",
         "Scientist", "Engineer", "Other"]
    )

    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
    sleep_quality = st.slider("Quality of Sleep (1–10)", min_value=1, max_value=10, value=7)

    physical_activity = st.selectbox(
        "Physical Activity Level",
        ["Low", "Moderate", "High"]
    )

    bmi_category = st.selectbox(
        "BMI Category",
        ["Normal", "Overweight", "Obese"]
    )

    blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)", value="120/80")

    heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=180, value=75)

    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)

    sleep_disorder = st.selectbox(
        "Sleep Disorder",
        ["None", "Insomnia", "Sleep Apnea"]
    )

    treatment = st.selectbox(
        "Spa Treatment",
        ["Aromatherapy", "Deep_Tissue_Massage", "Hot_Stone_Therapy"]
    )

    # StressBefore is an internal numeric feature (0–10 scale)
    stress_before = st.slider("Initial Stress Level (Before Treatment)", 0, 10, 5)

    submitted = st.form_submit_button("Predict Stress Drop")

# -------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------
if submitted:
    # Convert user input into DataFrame (single row)
    input_dict = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep_Duration": sleep_duration,
        "Quality_of_Sleep": sleep_quality,
        "Physical_Activity_Level": physical_activity,
        "BMI_Category": bmi_category,
        "Blood_Pressure": blood_pressure,
        "Heart_Rate": heart_rate,
        "Daily_Steps": daily_steps,
        "Sleep_Disorder": sleep_disorder,
        "Treatment": treatment,
        "StressBefore": stress_before
    }

    input_df = pd.DataFrame([input_dict])

    # Predict
    predicted_drop = model.predict(input_df)[0]

    # Clamp to 0–100 for safety
    predicted_drop = max(0, min(100, predicted_drop))

    # -------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------
    st.success(f"⭐ Predicted Stress-Drop Score: **{predicted_drop:.2f} / 100**")

    if predicted_drop >= 70:
        st.write("### 🟢 Excellent Response Expected! This treatment will significantly reduce passenger stress.")
    elif predicted_drop >= 40:
        st.write("### 🟡 Moderate Response. The passenger is likely to benefit from the treatment.")
    else:
        st.write("### 🔴 Low Expected Response. Consider an alternative treatment to improve outcomes.")




