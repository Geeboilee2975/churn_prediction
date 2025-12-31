import streamlit as st
import pandas as pd
import joblib

# Load the model you just created
model = joblib.load('churn_model.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📈")

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

# 1. User Inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)

# 2. Convert text to numbers (Matching your train.py logic)
# LabelEncoder for ['Month-to-month', 'One year', 'Two year'] becomes [0, 1, 2]
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_encoded = contract_mapping[contract]

if st.button("Predict Churn"):
    # The order MUST match the order in train.py: Contract, MonthlyCharges, tenure
    input_data = pd.DataFrame([[contract_encoded, monthly_charges, tenure]], 
                             columns=['Contract', 'MonthlyCharges', 'tenure'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("🚨 Result: This customer is likely to CHURN.")
    else:
        st.success("✅ Result: This customer is likely to STAY.")
        