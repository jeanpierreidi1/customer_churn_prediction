# app.py
# Customer Churn Prediction — Streamlit Web App

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon  = "📉",
    layout     = "centered"
)

# ── Load saved model ──────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── App Header ────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.markdown("""
    Enter customer details below to predict whether
    they are likely to churn.
""")
st.info(
    "ℹ️ **Demo only:** this form uses a simplified set of inputs and "
    "does not include every feature the model was trained on. "
    "Predictions are for demonstration purposes, not real business decisions.",
    icon="ℹ️"
)
st.divider()

# ── Input Form ────────────────────────────────────────────
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges   = monthly_charges * tenure

with col2:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    senior_citizen = st.checkbox("Senior Citizen")

st.divider()

# ── Predict Button ────────────────────────────────────────
if st.button("🔮 Predict Churn", type="primary"):

    # Build input — must match training features exactly
    # Using simplified feature set for demo
    charges_per_tenure = total_charges / (tenure + 1)
    is_new_customer    = 1 if tenure < 12 else 0
    is_long_term       = 1 if tenure > 24 else 0
    contract_encoded   = {"Month-to-month": 0,
                          "One year": 1,
                          "Two year": 2}[contract]
    internet_encoded   = {"DSL": 0,
                          "Fiber optic": 1,
                          "No": 2}[internet_service]

    # Create feature array
    features = np.array([[
        1 if senior_citizen else 0,
        tenure,
        monthly_charges,
        total_charges,
        charges_per_tenure,
        is_new_customer,
        is_long_term,
        contract_encoded,
        internet_encoded,
    ]])

    # Scale features
    # Note: in production, use full feature set from training
    # This is a simplified demo version
    churn_prob = model.predict_proba(
        scaler.transform(
            np.pad(features,
                   ((0,0),(0, scaler.n_features_in_ - features.shape[1])),
                   mode='constant')
        )
    )[0][1]

    # Display result
    st.subheader("Prediction Result")
    st.caption("⚠️ Simplified demo prediction — not based on full customer profile.")

    if churn_prob > 0.6:
        st.error(f"⚠️ HIGH CHURN RISK — {churn_prob*100:.1f}% probability")
        st.markdown("""
        **Recommended Actions:**
        - Offer loyalty discount
        - Upgrade to annual contract
        - Assign dedicated account manager
        """)
    elif churn_prob > 0.4:
        st.warning(f"🟡 MODERATE CHURN RISK — {churn_prob*100:.1f}% probability")
        st.markdown("""
        **Recommended Actions:**
        - Send satisfaction survey
        - Offer promotional upgrade
        """)
    else:
        st.success(f"✅ LOW CHURN RISK — {churn_prob*100:.1f}% probability")
        st.markdown("Customer appears satisfied — continue monitoring.")

    # Show input summary
    st.divider()
    st.subheader("Input Summary")
    st.dataframe(pd.DataFrame({
        "Feature":  ["Tenure", "Monthly Charges",
                     "Total Charges", "Contract",
                     "Internet Service", "Senior Citizen"],
        "Value":    [f"{tenure} months",
                     f"${monthly_charges:.2f}",
                     f"${total_charges:.2f}",
                     contract, internet_service,
                     "Yes" if senior_citizen else "No"]
    }))

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown("""
    **Model:** Logistic Regression | **Accuracy:** 80.4% |
    **ROC-AUC:** 0.845
    Built by Jean Pierre Idi |
    [GitHub](https://github.com/jeanpierreidi1)
""")
