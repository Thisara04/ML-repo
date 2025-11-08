import streamlit as st
import pandas as pd
import joblib

# Load model and scaler.
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💤 Sleep Quality Prediction App")

# --- INPUT FIELDS ---

# Dropdown for Gender
gender = st.selectbox("Gender", ["Male", "Female"])
gender_value = 0 if gender == "Male" else 1  # encode same as your training data

# Number input for Age
age = st.number_input("Age", 10, 100, 25)

# Slider for Sleep Duration
sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0)

# Slider for Stress Level
stress_level = st.slider("Stress Level", 1, 10, 5)

# Dropdown for BMI Category
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
bmi_map = {"Underweight": 18.0, "Normal": 23.0, "Overweight": 27.0, "Obese": 32.0}
bmi_value = bmi_map[bmi_category]

# Other inputs
heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 70)
daily_steps = st.number_input("Daily Steps", 1000, 20000, 8000)
systolic_bp = st.number_input("Systolic BP", 90, 180, 120)
diastolic_bp = st.number_input("Diastolic BP", 60, 120, 80)
quality_of_sleep = st.slider("Quality of Sleep", 1, 10, 7)
physical_activity = st.slider("Physical Activity Level", 0, 100, 50)

# --- CREATE DATAFRAME ---
input_df = pd.DataFrame([{
    'Gender': gender_value,
    'Age': age,
    'Sleep Duration': sleep_duration,
    'Quality of Sleep': quality_of_sleep,
    'Physical Activity Level': physical_activity,
    'Stress Level': stress_level,
    'BMI Category': bmi_value,
    'Heart Rate': heart_rate,
    'Daily Steps': daily_steps,
    'Systolic BP': systolic_bp,
    'Diastolic BP': diastolic_bp
}])

# --- PREDICTION ---
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

# --- OUTPUT ---
st.subheader("🧠 Prediction Result")
st.write(f"**Predicted Sleep Disorder:** {prediction}")
