import streamlit as st
import pandas as pd
import numpy as np

# Dummy prediction function
def dummy_predict(input_df):
    return 0, [0.7, 0.3]  # Always predicts Non-Dementia, 70% / 30% probability

st.title("Dementia Risk Prediction")
st.write("Enter the patient's details to estimate dementia risk.")

# Example input fields
SEX = st.selectbox("Sex", [0, 1])
AGE = st.number_input("Age", min_value=0, max_value=120, value=70)

if st.button("Predict"):
    prediction, prediction_prob = dummy_predict(None)
    st.write(f"Predicted Dementia State: **{'Non-Dementia' if prediction==0 else 'Risk of Dementia'}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
