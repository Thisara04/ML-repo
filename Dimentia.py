# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load calibrated model
@st.cache_data
def load_model():
    model = joblib.load("ensemble_calibrated.pkl")
    return model

calibrated_model = load_model()

# Load scaler if you want to scale inputs (optional for tree-based)
# scaler = joblib.load("scaler.pkl")

# Mapping for prediction
prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

st.title("Dementia Risk Prediction")
st.write("Enter the patient's details to estimate dementia risk.")

# Example input fields (add all required features)
def user_input_features():
    SEX = st.selectbox("Sex", [0, 1])
    HISPANIC = st.selectbox("Hispanic", [0, 1])
    HISPOR = st.number_input("Hispanic Origin (HISPOR)", min_value=0, max_value=8, value=0)
    RACE = st.number_input("Race", min_value=0, max_value=6, value=1)
    EDUC = st.number_input("Years of Education", min_value=0, max_value=30, value=16)
    MARISTAT = st.selectbox("Marital Status", [0, 1, 2])
    NACCAGE = st.number_input("Age", min_value=0, max_value=120, value=70)
    HEIGHT = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    WEIGHT = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70, step=0.1)
    
    # Add remaining features similarly...
    
    data = {
        "SEX": SEX,
        "HISPANIC": HISPANIC,
        "HISPOR": HISPOR,
        "RACE": RACE,
        "EDUC": EDUC,
        "MARISTAT": MARISTAT,
        "NACCAGE": NACCAGE,
        "HEIGHT": HEIGHT,
        "WEIGHT": WEIGHT,
        # add all remaining features here...
    }
    
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Predict button
if st.button("Predict"):
    # If using scaler: input_scaled = scaler.transform(input_df)
    prediction = calibrated_model.predict(input_df)[0]
    prediction_prob = calibrated_model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
