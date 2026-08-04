import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------

model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏦 Customer Churn Prediction Dashboard")

st.markdown(
"""
This dashboard predicts whether a customer is likely to leave the bank
using the trained **Random Forest Machine Learning Model**.
"""
)

st.divider()

# --------------------------------------------------
# CUSTOMER INPUT
# --------------------------------------------------

st.header("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0
    )

with col2:

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    has_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

st.divider()
# --------------------------------------------------
# CONVERT INPUTS
# --------------------------------------------------

has_card_value = 1 if has_card == "Yes" else 0
is_active_value = 1 if is_active == "Yes" else 0
gender_male = 1 if gender == "Male" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame([
        {
            "Year": 2025,
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "EstimatedSalary": estimated_salary,
            "NumOfProducts": num_products,
            "HasCrCard": has_card_value,
            "IsActiveMember": is_active_value,
            "Gender_Male": gender_male,
            "Geography_Germany": geo_germany,
            "Geography_Spain": geo_spain,
        }
    ])

    expected_columns = list(scaler.feature_names_in_)
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")

    st.metric(
        label="Churn Probability",
        value=f"{probability * 100:.2f}%"
    )

    st.divider()

    st.subheader("Customer Risk Level")

    if probability >= 0.70:
        st.error("🔴 High Risk Customer")

    elif probability >= 0.40:
        st.warning("🟡 Medium Risk Customer")

    else:
        st.success("🟢 Low Risk Customer")
