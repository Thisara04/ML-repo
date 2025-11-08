import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD TRAINED MODEL AND SCALER ---
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💤 Sleep Quality Prediction App")
st.markdown("Predict your **Sleep Disorder** based on your daily lifestyle features.")

# --- INPUT FIELDS ---

# Dropdown for Gender
gender = st.selectbox("Gender", ["Male", "Female"])
gender_value = 0 if gender == "Male" else 1  # Encode same as training data

# Numeric inputs and sliders
age = st.number_input("Age", min_value=10, max_value=100, value=25)
sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0)
quality_of_sleep = st.slider("Quality of Sleep", 1, 10, 7)
physical_activity = st.slider("Physical Activity Level", 0, 100, 50)
stress_level = st.slider("Stress Level", 1, 10, 5)

# Dropdown for BMI Category
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
bmi_map = {"Underweight": 18.0, "Normal": 23.0, "Overweight": 27.0, "_
