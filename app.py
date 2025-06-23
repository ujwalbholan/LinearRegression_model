import streamlit as st
import pickle
import numpy as np
import pandas as pd

try:
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Make sure 'linear_regression_model.pkl' is in the same directory.")
    st.stop()

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file not found. Make sure 'scaler.pkl' is in the same directory.")
    st.stop()

st.title("Stroke Risk Prediction (Linear Regression)")

st.markdown("Enter the patient details below:")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)

if st.button("🚀 Predict Stroke Risk"):
    input_df = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose_level, bmi]],
                            columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'])

    input_scaled = scaler.transform(input_df)

    risk_score = model.predict(input_scaled)[0]
    prediction = "Yes" if risk_score >= 0.5 else "No"

    st.success(f"🩺 Stroke Risk Score: `{risk_score:.3f}`")
    st.write(f"Prediction: **{'Yes' if prediction == 'Yes' else ' No'}**")
