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
    SEX = st.selectbox("Sex", [0, 1])
    HISPANIC = st.selectbox("Hispanic", [0, 1])
    HISPOR = st.number_input("Hispanic Origin (HISPOR)", min_value=0, max_value=8, value=0)
    RACE = st.number_input("Race", min_value=0, max_value=6, value=1)
    PRIMLANG = st.number_input("Primary Language", min_value=0, max_value=9, value=1)
    EDUC = st.number_input("Years of Education", min_value=0, max_value=30, value=16)
    MARISTAT = st.selectbox("Marital Status", [0, 1, 2])
    NACCLIVS = st.number_input("Living Situation", min_value=0, max_value=10, value=0)
    INDEPEND = st.selectbox("Independent?", [0, 1])
    RESIDENC = st.number_input("Residence Type", min_value=0, max_value=5, value=1)
    HANDED = st.selectbox("Handedness", [1, 2])
    NACCAGE = st.number_input("Age", min_value=0, max_value=120, value=70)
    NACCAGEB = st.number_input("Age at Baseline", min_value=0, max_value=120, value=70)
    INBIRYR = st.number_input("Birth Year", min_value=1900, max_value=2025, value=1950)
    NEWINF = st.number_input("New Info", min_value=-4, max_value=9, value=-4)
    INRELTO = st.number_input("Relationship", min_value=0, max_value=10, value=0)
    INLIVWTH = st.number_input("Lives With", min_value=0, max_value=10, value=0)
    INRELY = st.number_input("Dependent?", min_value=0, max_value=10, value=0)
    NACCFAM = st.number_input("Family History", min_value=0, max_value=10, value=0)
    NACCMOM = st.number_input("Mother Status", min_value=0, max_value=10, value=0)
    NACCDAD = st.number_input("Father Status", min_value=0, max_value=10, value=0)
    ANYMEDS = st.selectbox("Taking any medications?", [0, 1])
    NACCAMD = st.number_input("Number of Medications", min_value=0, max_value=50, value=0)
    TOBAC100 = st.selectbox("Smoker 100+", [0, 1])
    SMOKYRS = st.number_input("Smoking Years", min_value=0, max_value=100, value=0)
    PACKSPER = st.number_input("Packs per Day", min_value=0, max_value=10, value=0)
    CVHATT = st.selectbox("Hypertension?", [0, 1])
    CVBYPASS = st.selectbox("Bypass Surgery?", [0, 1])
    CVPACE = st.selectbox("Pacemaker?", [0, 1])
    CVHVALVE = st.selectbox("Heart Valve Issue?", [0, 1])
    CBSTROKE = st.selectbox("Stroke History?", [0, 1])
    TBIBRIEF = st.selectbox("TBI Brief?", [0, 1])
    TBIEXTEN = st.selectbox("TBI Extent?", [0, 1])
    DEP2YRS = st.selectbox("Depression 2 Yrs?", [0, 1])
    DEPOTHR = st.selectbox("Other Depression?", [0, 1])
    NACCTBI = st.selectbox("TBI History?", [0, 1])
    HEIGHT = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    WEIGHT = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    NACCBMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    VISION = st.selectbox("Vision Issue?", [0, 1])
    VISCORR = st.selectbox("Corrected Vision?", [0, 1])
    VISWCORR = st.selectbox("Worse Vision?", [0, 1])
    HEARING = st.selectbox("Hearing Issue?", [0, 1])
    HEARAID = st.selectbox("Hearing Aid?", [0, 1])
    HXSTROKE = st.selectbox("History Stroke?", [0, 1])
    HALL = st.selectbox("Hallucinations?", [0, 1])
    APP = st.selectbox("Apathy?", [0, 1])
    BILLS = st.selectbox("Can manage Bills?", [0, 1])
    TAXES = st.selectbox("Can manage Taxes?", [0, 1])
    SHOPPING = st.selectbox("Can go Shopping?", [0, 1])
    GAMES = st.selectbox("Can play Games?", [0, 1])
    STOVE = st.selectbox("Can use Stove?", [0, 1])
    MEALPREP = st.selectbox("Can prepare Meals?", [0, 1])
    EVENTS = st.selectbox("Can attend Events?", [0, 1])
    PAYATTN = st.selectbox("Can pay Attention?", [0, 1])
    REMDATES = st.selectbox("Can remember Dates?", [0, 1])
    TRAVEL = st.selectbox("Can Travel?", [0, 1])
    
    data = {
        "SEX": SEX, "HISPANIC": HISPANIC, "HISPOR": HISPOR, "RACE": RACE, "PRIMLANG": PRIMLANG,
        "EDUC": EDUC, "MARISTAT": MARISTAT, "NACCLIVS": NACCLIVS, "INDEPEND": INDEPEND,
        "RESIDENC": RESIDENC, "HANDED": HANDED, "NACCAGE": NACCAGE, "NACCAGEB": NACCAGEB,
        "INBIRYR": INBIRYR, "NEWINF": NEWINF, "INRELTO": INRELTO, "INLIVWTH": INLIVWTH,
        "INRELY": INRELY, "NACCFAM": NACCFAM, "NACCMOM": NACCMOM, "NACCDAD": NACCDAD,
        "ANYMEDS": ANYMEDS, "NACCAMD": NACCAMD, "TOBAC100": TOBAC100, "SMOKYRS": SMOKYRS,
        "PACKSPER": PACKSPER, "CVHATT": CVHATT, "CVBYPASS": CVBYPASS, "CVPACE": CVPACE,
        "CVHVALVE": CVHVALVE, "CBSTROKE": CBSTROKE, "TBIBRIEF": TBIBRIEF, "TBIEXTEN": TBIEXTEN,
        "DEP2YRS": DEP2YRS, "DEPOTHR": DEPOTHR, "NACCTBI": NACCTBI, "HEIGHT": HEIGHT, "WEIGHT": WEIGHT,
        "NACCBMI": NACCBMI, "VISION": VISION, "VISCORR": VISCORR, "VISWCORR": VISWCORR, "HEARING": HEARING,
        "HEARAID": HEARAID, "HXSTROKE": HXSTROKE, "HALL": HALL, "APP": APP, "BILLS": BILLS,
        "TAXES": TAXES, "SHOPPING": SHOPPING, "GAMES": GAMES, "STOVE": STOVE, "MEALPREP": MEALPREP,
        "EVENTS": EVENTS, "PAYATTN": PAYATTN, "REMDATES": REMDATES, "TRAVEL": TRAVEL
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
