import streamlit as st
import pandas as pd
import numpy as np

st.title("Dementia Risk Prediction")
st.write("Enter patient details to estimate dementia risk.")

# Dummy prediction function
def dummy_predict(input_df):
    return 0, [0.7, 0.3]  # Always predicts Non-Dementia

def user_input_features():
    # --- Categorical features with mappings ---
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
    
    PRIMLANG = st.selectbox(
    "Primary Language",
    ["English", "Spanish", "Mandarin", "Cantonese", "Russian", "Japanese", "Other", "Unknown"])
    PRIMLANG_val = {
    "English": 1,
    "Spanish": 2,
    "Mandarin": 3,
    "Cantonese": 4,
    "Russian": 5,
    "Japanese": 6,
    "Other": 7,
    "Unknown": np.nan}[PRIMLANG] #ok

    educ_options = [str(i) for i in range(0, 37)] + ["Unknown"]
    EDUC = st.selectbox("Years of Education", educ_options)
    EDUC_val = np.nan if EDUC == "Unknown" else int(EDUC) #ok


    MARISTAT = st.selectbox(
    "Marital Status",
    ["Married", "Widowed", "Divorced", "Separated", "Never married/Annulled", "Domestic partner", "Unknown"])
    MARISTAT_val = (
    1 if MARISTAT == "Married" else
    2 if MARISTAT == "Widowed" else
    3 if MARISTAT == "Divorced" else
    4 if MARISTAT == "Separated" else
    5 if MARISTAT == "Never married/Annulled" else
    6 if MARISTAT == "Domestic partner" else
    np.nan) #ok

    NACCLIVS = st.selectbox(
    "Living situation",
    ["Alone", "With spouce/partner", "With Relative/Friend", "With a Group", "Other","Unknown"])
    NACCLIVS_val = (
    1 if NACCLIVS == "Alone" else
    2 if NACCLIVS == "With spouce/partner" else
    3 if NACCLIVS == "With Relative/Friend" else
    4 if NACCLIVS == "With a Group" else
    5 if NACCLIVS == "Other" else
    np.nan) #ok

    INDEPEND = st.selectbox(
    "Level of Independance",
    ["able to live independantly", "Require assistance with complex activities", "Require assistance with basic activities", "Completely dependent","Unknown"])
    INDEPEND_val = (
    1 if INDEPEND == "able to live independantly" else
    2 if INDEPEND == "Require assistance with complex activities" else
    3 if INDEPEND == "Require assistance with basic activities" else
    4 if INDEPEND == "Completely dependent" else
    np.nan) #ok

    RESIDENC = st.number_input("Residence Type", min_value=0, max_value=5, value=1)

    HANDED = st.selectbox("Handedness", ["Left", "Right","Ambidextrous", "Unknown"])
    HANDED_val = 1 if HANDED=="Left" else 2 if HANDED=="Right" else 3 if HANDED=="Ambidextrous" else np.nan  #ok

    NACCAGE = st.number_input("Person's Age", min_value=18, max_value=120, value=70)
    NACCAGEB = st.number_input("Confirm Person's age", min_value=18, max_value=120, value=70)
    
    unknown_birth = st.checkbox("Birth Year Unknown")
    if unknown_birth:
        INBIRYR = np.nan
    else:
        INBIRYR = st.number_input(
        "Co-participant Birth Year",
        min_value=1875,
        max_value=2025,
        value=1950)  #ok
        
    NEWINF = st.number_input("Co-participant Familiar with data collecting process", min_value=-4, max_value=9, value=-4)
    INRELTO = st.number_input("Relationship", min_value=0, max_value=10, value=0)
    INLIVWTH = st.number_input("Lives With", min_value=0, max_value=10, value=0)
    INRELY = st.number_input("Dependent?", min_value=0, max_value=10, value=0)
    NACCFAM = st.number_input("Family History", min_value=0, max_value=10, value=0)
    NACCMOM = st.number_input("Mother Status", min_value=0, max_value=10, value=0)
    NACCDAD = st.number_input("Father Status", min_value=0, max_value=10, value=0)

    ANYMEDS = st.selectbox("Taking any medications?", ["No", "Yes", "Unknown"])
    ANYMEDS_val = 0 if ANYMEDS=="No" else 1 if ANYMEDS=="Yes" else np.nan

    NACCAMD = st.number_input("Number of Medications", min_value=0, max_value=50, value=0)

    TOBAC100 = st.selectbox("Smoker 100+?", ["No", "Yes", "Unknown"])
    TOBAC100_val = 0 if TOBAC100=="No" else 1 if TOBAC100=="Yes" else np.nan

    SMOKYRS = st.number_input("Smoking Years", min_value=0, max_value=100, value=0)
    PACKSPER = st.number_input("Packs per Day", min_value=0, max_value=10, value=0)

    CVHATT = st.selectbox("Hypertension?", ["No", "Yes", "Unknown"])
    CVHATT_val = 0 if CVHATT=="No" else 1 if CVHATT=="Yes" else np.nan

    CVBYPASS = st.selectbox("Bypass Surgery?", ["No", "Yes", "Unknown"])
    CVBYPASS_val = 0 if CVBYPASS=="No" else 1 if CVBYPASS=="Yes" else np.nan

    CVPACE = st.selectbox("Pacemaker?", ["No", "Yes", "Unknown"])
    CVPACE_val = 0 if CVPACE=="No" else 1 if CVPACE=="Yes" else np.nan

    CVHVALVE = st.selectbox("Heart Valve Issue?", ["No", "Yes", "Unknown"])
    CVHVALVE_val = 0 if CVHVALVE=="No" else 1 if CVHVALVE=="Yes" else np.nan

    CBSTROKE = st.selectbox("Stroke History?", ["No", "Yes", "Unknown"])
    CBSTROKE_val = 0 if CBSTROKE=="No" else 1 if CBSTROKE=="Yes" else np.nan

    TBIBRIEF = st.selectbox("TBI Brief?", ["No", "Yes", "Unknown"])
    TBIBRIEF_val = 0 if TBIBRIEF=="No" else 1 if TBIBRIEF=="Yes" else np.nan

    TBIEXTEN = st.selectbox("TBI Extent?", ["No", "Yes", "Unknown"])
    TBIEXTEN_val = 0 if TBIEXTEN=="No" else 1 if TBIEXTEN=="Yes" else np.nan

    DEP2YRS = st.selectbox("Depression 2 Yrs?", ["No", "Yes", "Unknown"])
    DEP2YRS_val = 0 if DEP2YRS=="No" else 1 if DEP2YRS=="Yes" else np.nan

    DEPOTHR = st.selectbox("Other Depression?", ["No", "Yes", "Unknown"])
    DEPOTHR_val = 0 if DEPOTHR=="No" else 1 if DEPOTHR=="Yes" else np.nan

    NACCTBI = st.selectbox("TBI History?", ["No", "Yes", "Unknown"])
    NACCTBI_val = 0 if NACCTBI=="No" else 1 if NACCTBI=="Yes" else np.nan

    HEIGHT = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    WEIGHT = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    NACCBMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    VISION = st.selectbox("Vision Issue?", ["No", "Yes", "Unknown"])
    VISION_val = 0 if VISION=="No" else 1 if VISION=="Yes" else np.nan

    VISCORR = st.selectbox("Corrected Vision?", ["No", "Yes", "Unknown"])
    VISCORR_val = 0 if VISCORR=="No" else 1 if VISCORR=="Yes" else np.nan

    VISWCORR = st.selectbox("Worse Vision?", ["No", "Yes", "Unknown"])
    VISWCORR_val = 0 if VISWCORR=="No" else 1 if VISWCORR=="Yes" else np.nan

    HEARING = st.selectbox("Hearing Issue?", ["No", "Yes", "Unknown"])
    HEARING_val = 0 if HEARING=="No" else 1 if HEARING=="Yes" else np.nan

    HEARAID = st.selectbox("Hearing Aid?", ["No", "Yes", "Unknown"])
    HEARAID_val = 0 if HEARAID=="No" else 1 if HEARAID=="Yes" else np.nan

    HXSTROKE = st.selectbox("History Stroke?", ["No", "Yes", "Unknown"])
    HXSTROKE_val = 0 if HXSTROKE=="No" else 1 if HXSTROKE=="Yes" else np.nan

    HALL = st.selectbox("Hallucinations?", ["No", "Yes", "Unknown"])
    HALL_val = 0 if HALL=="No" else 1 if HALL=="Yes" else np.nan

    APP = st.selectbox("Apathy?", ["No", "Yes", "Unknown"])
    APP_val = 0 if APP=="No" else 1 if APP=="Yes" else np.nan

    BILLS = st.selectbox("Can manage Bills?", ["No", "Yes", "Unknown"])
    BILLS_val = 0 if BILLS=="No" else 1 if BILLS=="Yes" else np.nan

    TAXES = st.selectbox("Can manage Taxes?", ["No", "Yes", "Unknown"])
    TAXES_val = 0 if TAXES=="No" else 1 if TAXES=="Yes" else np.nan

    SHOPPING = st.selectbox("Can go Shopping?", ["No", "Yes", "Unknown"])
    SHOPPING_val = 0 if SHOPPING=="No" else 1 if SHOPPING=="Yes" else np.nan

    GAMES = st.selectbox("Can play Games?", ["No", "Yes", "Unknown"])
    GAMES_val = 0 if GAMES=="No" else 1 if GAMES=="Yes" else np.nan

    STOVE = st.selectbox("Can use Stove?", ["No", "Yes", "Unknown"])
    STOVE_val = 0 if STOVE=="No" else 1 if STOVE=="Yes" else np.nan

    MEALPREP = st.selectbox("Can prepare Meals?", ["No", "Yes", "Unknown"])
    MEALPREP_val = 0 if MEALPREP=="No" else 1 if MEALPREP=="Yes" else np.nan

    EVENTS = st.selectbox("Can attend Events?", ["No", "Yes", "Unknown"])
    EVENTS_val = 0 if EVENTS=="No" else 1 if EVENTS=="Yes" else np.nan

    PAYATTN = st.selectbox("Can pay Attention?", ["No", "Yes", "Unknown"])
    PAYATTN_val = 0 if PAYATTN=="No" else 1 if PAYATTN=="Yes" else np.nan

    REMDATES = st.selectbox("Can remember Dates?", ["No", "Yes", "Unknown"])
    REMDATES_val = 0 if REMDATES=="No" else 1 if REMDATES=="Yes" else np.nan

    TRAVEL = st.selectbox("Can Travel?", ["No", "Yes", "Unknown"])
    TRAVEL_val = 0 if TRAVEL=="No" else 1 if TRAVEL=="Yes" else np.nan

    # --- Combine all into a dict ---
    data = {
        "SEX": SEX_val, "HISPANIC": HISPANIC_val, "HISPOR": HISPOR_val, "RACE": RACE_val,
        "PRIMLANG": PRIMLANG_val, "EDUC": EDUC, "MARISTAT": MARISTAT_val, "NACCLIVS": NACCLIVS,
        "INDEPEND": INDEPEND_val, "RESIDENC": RESIDENC, "HANDED": HANDED_val,
        "NACCAGE": NACCAGE, "NACCAGEB": NACCAGEB, "INBIRYR": INBIRYR, "NEWINF": NEWINF,
        "INRELTO": INRELTO, "INLIVWTH": INLIVWTH, "INRELY": INRELY, "NACCFAM": NACCFAM,
        "NACCMOM": NACCMOM, "NACCDAD": NACCDAD, "ANYMEDS": ANYMEDS_val, "NACCAMD": NACCAMD,
        "TOBAC100": TOBAC100_val, "SMOKYRS": SMOKYRS, "PACKSPER": PACKSPER,
        "CVHATT": CVHATT_val, "CVBYPASS": CVBYPASS_val, "CVPACE": CVPACE_val, "CVHVALVE": CVHVALVE_val,
        "CBSTROKE": CBSTROKE_val, "TBIBRIEF": TBIBRIEF_val, "TBIEXTEN": TBIEXTEN_val,
        "DEP2YRS": DEP2YRS_val, "DEPOTHR": DEPOTHR_val, "NACCTBI": NACCTBI_val,
        "HEIGHT": HEIGHT, "WEIGHT": WEIGHT, "NACCBMI": NACCBMI, "VISION": VISION_val,
        "VISCORR": VISCORR_val, "VISWCORR": VISWCORR_val, "HEARING": HEARING_val,
        "HEARAID": HEARAID_val, "HXSTROKE": HXSTROKE_val, "HALL": HALL_val, "APP": APP_val,
        "BILLS": BILLS_val, "TAXES": TAXES_val, "SHOPPING": SHOPPING_val, "GAMES": GAMES_val,
        "STOVE": STOVE_val, "MEALPREP": MEALPREP_val, "EVENTS": EVENTS_val, "PAYATTN": PAYATTN_val,
        "REMDATES": REMDATES_val, "TRAVEL": TRAVEL_val
    }

    return pd.DataFrame([data])

# --- Collect input ---
input_df = user_input_features()

# --- Predict ---
if st.button("Predict"):
    prediction, prediction_prob = dummy_predict(input_df)
    prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
