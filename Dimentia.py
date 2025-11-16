# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# -------------------------------
# Download the model if not exists
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=153VzcC2Ni-T2Pew5ne6e1zNBThNadJHV"
MODEL_PATH = "ensemble_calibrated.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------------
# Load calibrated model
# -------------------------------
@st.cache_data
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

try:
    calibrated_model = joblib.load("ensemble_calibrated.pkl")
except Exception as e:
    import streamlit as st
    st.error(f"Error loading model: {e}")


# -------------------------------
# Prediction mapping
# -------------------------------
prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Dementia Risk Prediction")
st.write("Enter the patient's details to estimate dementia risk.")

# Example input fields
def user_input_features():
    SEX = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    HISPANIC = st.selectbox("Hispanic (0=No, 1=Yes)", [0, 1])
    HISPOR = st.number_input("Hispanic Origin (HISPOR)", min_value=0, max_value=8, value=0)
    RACE = st.number_input("Race", min_value=0, max_value=6, value=1)
    EDUC = st.number_input("Years of Education", min_value=0, max_value=30, value=16)
    MARISTAT = st.selectbox("Marital Status", [0, 1, 2])
    NACCAGE = st.number_input("Age", min_value=0, max_value=120, value=70)
    HEIGHT = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    WEIGHT = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70, step=0.1)

    # Placeholder for remaining features (set default to 0 or np.nan)
    remaining_features = [
        "HANDED", "NACCLIVS", "INDEPEND", "RESIDENC", "NACCAGEB", "INBIRYR",
        "NEWINF", "INRELTO", "INLIVWTH", "INRELY", "NACCFAM", "NACCMOM",
        "NACCDAD", "ANYMEDS", "NACCAMD", "TOBAC100", "SMOKYRS", "PACKSPER",
        "CVHATT", "CVBYPASS", "CVPACE", "CVHVALVE", "CBSTROKE", "TBIBRIEF",
        "TBIEXTEN", "DEP2YRS", "DEPOTHR", "NACCTBI", "NACCBMI", "VISION",
        "VISCORR", "VISWCORR", "HEARING", "HEARAID", "HXSTROKE", "HALL",
        "APP", "BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP",
        "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"
    ]
    data = {
        "SEX": SEX,
        "HISPANIC": HISPANIC,
        "HISPOR": HISPOR,
        "RACE": RACE,
        "EDUC": EDUC,
        "MARISTAT": MARISTAT,
        "NACCAGE": NACCAGE,
        "HEIGHT": HEIGHT,
        "WEIGHT": WEIGHT
    }

    # Set remaining features to 0 by default
    for f in remaining_features:
        data[f] = 0 if f != "NACCBMI" else np.nan

    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    prediction = calibrated_model.predict(input_df)[0]
    prediction_prob = calibrated_model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
