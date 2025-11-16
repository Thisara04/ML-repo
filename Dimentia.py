import streamlit as st
import pandas as pd
import numpy as np

# Dummy prediction function for interface testing
def dummy_predict(input_df):
    # Always predicts Non-Dementia with 70% probability
    return 0, [0.7, 0.3]

st.title("Dementia Risk Prediction")
st.write("Enter the patient's details to estimate dementia risk.")

# Mapping for user-friendly categorical inputs
sex_options = {"Male": 1, "Female": 0}

# User input
SEX_label = st.selectbox("Sex", list(sex_options.keys()))
SEX = sex_options[SEX_label]  # Map to numeric for the model

AGE = st.number_input("Age", min_value=0, max_value=120, value=70)
HEIGHT = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
WEIGHT = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70, step=0.1)

# Add other features here similarly...

# Create input DataFrame
input_df = pd.DataFrame([{
    "SEX": SEX,
    "NACCAGE": AGE,
    "HEIGHT": HEIGHT,
    "WEIGHT": WEIGHT
    # Add other features here
}])

# Predict button
if st.button("Predict"):
    prediction, prediction_prob = dummy_predict(input_df)
    
    prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}
    
    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
