import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = joblib.load('churn_model.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📈")

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

# 1. User Inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)

# 2. Convert text to numbers
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_encoded = contract_mapping[contract]

# 3. Prediction and Visualization Logic
if st.button("Predict Churn"):
    # Create the data for the model
    input_data = pd.DataFrame([[contract_encoded, monthly_charges, tenure]], 
                             columns=['Contract', 'MonthlyCharges', 'tenure'])
    
    # Run the prediction
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] # Probability of Churn
    
    # Display the result boxes
    if prediction[0] == 1:
        st.error(f"🚨 Result: This customer is likely to CHURN. (Risk: {prob*100:.1f}%)")
    else:
        st.success(f"✅ Result: This customer is likely to STAY. (Stay Probability: {(1-prob)*100:.1f}%)")

    # --- VISUALIZATIONS (Now inside the button block) ---
    st.divider()
    st.subheader("📊 Analytical Results")

    # 1. Risk Meter
    st.write(f"Churn Probability Gauge")
    st.progress(prob)

    # 2. Feature Importance
    importances = model.feature_importances_
    feature_names = ['Contract', 'MonthlyCharges', 'tenure']
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    st.write("### What influenced this result?")
    st.bar_chart(feature_df.set_index('Feature'))
    