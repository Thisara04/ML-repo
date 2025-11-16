import streamlit as st
import pandas as pd
import numpy as np

st.title("Dementia Risk Prediction")
st.write("Enter patient details to estimate dementia risk.")

# Dummy prediction function
def dummy_predict(input_df):
    return 0, [0.7, 0.3]  # Always predicts Non-Dementia, 70% / 30% probability

# User input
def user_input_features():

    # Example categorical fields with readable options
    SEX = st.selectbox("Gender", ["Male", "Female", "Unknown"])
    SEX_value = 0 if SEX=="Male" else 1 if SEX=="Female" else np.nan

    HISPANIC = st.selectbox("Hispanic/Latino Ethnicity", ["No", "Yes", "Unknown"])
    HISPANIC_value = 0 if HISPANIC=="No" else 1 if HISPANIC=="Yes" else np.nan

    MARISTAT = st.selectbox("Marital Status", ["Single", "Married", "Other", "Unknown"])
    MARISTAT_value = 0 if MARISTAT=="Single" else 1 if MARISTAT=="Married" else 2 if MARISTAT=="Other" else np.nan

    INDEPEND = st.selectbox("Independent?", ["No", "Yes", "Unknown"])
    INDEPEND_value = 0 if INDEPEND=="No" else 1 if INDEPEND=="Yes" else np.nan

    HANDED = st.selectbox("Handedness", ["Right", "Left", "Unknown"])
    HANDED_value = 1 if HANDED=="Right" else 2 if HANDED=="Left" else np.nan

    ANYMEDS = st.selectbox("Taking any medications?", ["No", "Yes", "Unknown"])
    ANYMEDS_value = 0 if ANYMEDS=="No" else 1 if ANYMEDS=="Yes" else np.nan

    # Add more categorical fields similarly...

    # Numeric inputs
    NACCAGE = st.number_input("Age", min_value=0, max_value=120, value=70)
    NACCAGEB = st.number_input("Age at Baseline", min_value=0, max_value=120, value=70)
    HEIGHT = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    WEIGHT = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    NACCBMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    # Combine all into a dict, using mapped values
    data = {
        "SEX": SEX_value,
        "HISPANIC": HISPANIC_value,
        "MARISTAT": MARISTAT_value,
        "INDEPEND": INDEPEND_value,
        "HANDED": HANDED_value,
        "ANYMEDS": ANYMEDS_value,
        "NACCAGE": NACCAGE,
        "NACCAGEB": NACCAGEB,
        "HEIGHT": HEIGHT,
        "WEIGHT": WEIGHT,
        "NACCBMI": NACCBMI
        # add remaining numeric fields and mapped categorical fields here
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Predict button
if st.button("Predict"):
    prediction, prediction_prob = dummy_predict(input_df)
    prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
