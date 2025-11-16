import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os 
'''
st.title("Dementia Risk Prediction")
st.write("Enter patient details to estimate dementia risk.")
st.write("Assist a co-participant to help the process!!.")

import streamlit as st
st.write("Hello, Streamlit is running!")

# --- Model download and loading ---
MODEL_URL = "https://huggingface.co/ThisaraAdhikari04/dementia-risk-model/resolve/main/Dementia_model.pkl"
MODEL_PATH = "Dementia_model.pkl"

def download_file(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner("Downloading model..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

download_file(MODEL_URL, MODEL_PATH)
model = joblib.load(MODEL_PATH)


def predict(input_df):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(prob))
    else:
        pred_class = int(model.predict(input_df)[0])
        prob = [1 - pred_class, pred_class]
    return pred_class, prob
'''

def user_input_features():
    SEX = st.selectbox("Gender", ["Male", "Female"])
    SEX_val = 1 if SEX=="Male" else 2 if SEX=="Female" else np.nan #ok

    HISPANIC = st.selectbox("Hispanic/Latino Ethnicity", ["No", "Yes", "Unknown"])
    HISPANIC_val = 0 if HISPANIC=="No" else 1 if HISPANIC=="Yes" else np.nan  #ok

    HISPOR = st.selectbox(
    "Hispanic Origin (HISPOR)",
    ["Mexican", "Puerto Rican", "Cuban", "Dominican", "Central American", "South American", "Other", "Unknown"])
    HISPOR_val = {
    "Mexican": 1,
    "Puerto Rican": 2,
    "Cuban": 3,
    "Dominican": 4,
    "Central American": 5,
    "South American": 7,
    "Other": 8,
    "Unknown": np.nan}[HISPOR]  #ok 

    RACE = st.selectbox("Race", ["White", "Black/African American", "American Indian","Native Hawaiian", "Asian", "Other", "Unknown"])
    RACE_val = 1 if RACE=="White" else 2 if RACE=="Black/African American" else 3 if RACE=="American Indian" else 4 if RACE=="Native Hawaiian" else 5 if RACE=="Asian" else 6 if RACE=="Other" else np.nan
    #ok
    
   

    data = {
    "SEX": SEX_val, "HISPANIC": HISPANIC_val, "HISPOR": HISPOR_val, "RACE": RACE_val,
    "PRIMLANG": PRIMLANG_val, "EDUC": EDUC_val, "MARISTAT": MARISTAT_val, "NACCLIVS": NACCLIVS_val,
    "INDEPEND": INDEPEND_val, "RESIDENC": RESIDENC_val, "HANDED": HANDED_val,
    "NACCAGE": NACCAGE, "NACCAGEB": NACCAGEB, "INBIRYR": INBIRYR_val, "NEWINF": NEWINF_val,
    "INRELTO": INRELTO_val, "INLIVWTH": INLIVWTH_val, "INRELY": INRELY_val, "NACCFAM": NACCFAM_val,
    "NACCMOM": NACCMOM_val, "NACCDAD": NACCDAD_val, "ANYMEDS": ANYMEDS_val, "NACCAMD": NACCAMD_val,
    "TOBAC100": TOBAC100_val, "SMOKYRS": SMOKYRS_val, "PACKSPER": PACKSPER_val,
    "CVHATT": CVHATT_val, "CVBYPASS": CVBYPASS_val, "CVPACE": CVPACE_val, "CVHVALVE": CVHVALVE_val,
    "CBSTROKE": CBSTROKE_val, "TBIBRIEF": TBIBRIEF_val, "TBIEXTEN": TBIEXTEN_val,
    "DEP2YRS": DEP2YRS_val, "DEPOTHR": DEPOTHR_val, "NACCTBI": NACCTBI_val,
    "HEIGHT": HEIGHT, "WEIGHT": WEIGHT, "NACCBMI": NACCBMI, "VISION": VISION_val,
    "VISCORR": VISCORR_val, "VISWCORR": VISWCORR_val, "HEARING": HEARING_val,
    "HEARAID": HEARAID_val, "HEARWAID": HEARWAID_val, "HXSTROKE": HXSTROKE_val,
    "HALL": HALL_val, "APP": APP_val,
    "BILLS": BILLS_val, "TAXES": TAXES_val, "SHOPPING": SHOPPING_val, "GAMES": GAMES_val,
    "STOVE": STOVE_val, "MEALPREP": MEALPREP_val, "EVENTS": EVENTS_val, "PAYATTN": PAYATTN_val,
    "REMDATES": REMDATES_val, "TRAVEL": TRAVEL_val}

    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict"):
    prediction, prediction_prob = dummy_predict(input_df)
    prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
