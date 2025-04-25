import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.markdown("Predict whether a loan applicant is a **Good** or **Bad** credit risk.")

with open("credit_risk_app/model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

def user_input():
    Status = st.selectbox("Status", [0, 1, 2, 3])
    Duration = st.slider("Duration (months)", 4, 72, 24)
    CreditHistory = st.selectbox("Credit History", [0, 1, 2, 3, 4])
    Purpose = st.selectbox("Purpose", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10])
    CreditAmount = st.number_input("Credit Amount", 250, 20000, 1000)
    Savings = st.selectbox("Savings", [0, 1, 2, 3, 4])
    EmploymentSince = st.selectbox("Employment Since", [0, 1, 2, 3, 4])
    InstallmentRate = st.slider("Installment Rate", 1, 4, 2)
    PersonalStatusSex = st.selectbox("Personal Status & Sex", [0, 1, 2, 3])
    OtherDebtors = st.selectbox("Other Debtors", [0, 1, 2])
    ResidenceSince = st.slider("Residence Since (years)", 1, 4, 2)
    Property = st.selectbox("Property", [0, 1, 2, 3])
    Age = st.slider("Age", 18, 75, 30)
    OtherInstallmentPlans = st.selectbox("Other Installment Plans", [0, 1, 2])
    Housing = st.selectbox("Housing", [0, 1, 2])
    NumberCredits = st.slider("Number of Credits", 1, 4, 1)
    Job = st.selectbox("Job", [0, 1, 2, 3])
    PeopleLiable = st.slider("People Liable", 1, 2, 1)
    Telephone = st.selectbox("Telephone", [0, 1])
    ForeignWorker = st.selectbox("Foreign Worker", [0, 1])

    data = np.array([[Status, Duration, CreditHistory, Purpose, CreditAmount, Savings,
                      EmploymentSince, InstallmentRate, PersonalStatusSex, OtherDebtors,
                      ResidenceSince, Property, Age, OtherInstallmentPlans, Housing,
                      NumberCredits, Job, PeopleLiable, Telephone, ForeignWorker]])
    return data

input_data = user_input()

if st.button("Predict"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    result = "🔒 Good Credit Risk" if prediction[0] == 0 else "⚠️ Bad Credit Risk"
    st.subheader("Prediction Result:")
    st.success(result)
