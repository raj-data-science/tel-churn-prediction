import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

# Title
st.title("Telecom Customer Churn Prediction")

st.write("Enter Customer Details")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure", 1, 72)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

total_charges = st.number_input("Total Charges", min_value=0.0)

# Prediction Button
if st.button("Predict"):

    # Convert categorical values
    gender = 1 if gender == "Male" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0

    # Create dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Result
    if prediction[0] == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Stay")