import streamlit as st
import pandas as pd
import joblib

# Load best model
model = joblib.load("heart_best_model.pkl")

# ---------------- UI ----------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction System by Kr. Roshan")
st.markdown("### AI-based Risk Detection System")

st.write("Fill patient details below:")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- ENCODING FUNCTION ----------------
def encode_input():
    mapping = {
        "Sex": {"M":1, "F":0},
        "ChestPainType": {"ATA":0, "NAP":1, "ASY":2, "TA":3},
        "RestingECG": {"Normal":0, "ST":1, "LVH":2},
        "ExerciseAngina": {"N":0, "Y":1},
        "ST_Slope": {"Up":0, "Flat":1, "Down":2}
    }

    data = {
        "Age": age,
        "Sex": mapping["Sex"][sex],
        "ChestPainType": mapping["ChestPainType"][chest_pain],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": mapping["RestingECG"][resting_ecg],
        "MaxHR": max_hr,
        "ExerciseAngina": mapping["ExerciseAngina"][exercise_angina],
        "Oldpeak": oldpeak,
        "ST_Slope": mapping["ST_Slope"][st_slope]
    }

    return pd.DataFrame([data])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    input_df = encode_input()

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {prob:.2f}")

    # Progress bar for probability
    st.progress(int(prob * 100))
