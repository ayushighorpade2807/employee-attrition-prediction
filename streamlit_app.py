"""
Employee Attrition Risk Predictor — Streamlit app
---------------------------------------------------
Loads the pretrained XGBoost model (trained in your Jupyter notebook)
and predicts the probability that an employee will leave, with a SHAP-based
explanation of the top contributing factors for that specific prediction.

Files needed alongside this script (produced by the notebook):
  - attrition_model.pkl       (trained XGBClassifier)
  - attrition_scaler.pkl      (fitted StandardScaler)
  - attrition_features.pkl    (ordered list of feature column names used in training)

Run with:
  streamlit run streamlit_app.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Employee Attrition Risk Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("attrition_scaler.pkl")
    feature_columns = joblib.load("attrition_features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_columns, explainer


try:
    model, scaler, feature_columns, explainer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run the notebook first to generate "
        "attrition_model.pkl, attrition_scaler.pkl, and attrition_features.pkl, "
        "then place them in the same folder as this script."
    )
    st.stop()


def get_top_shap_features(shap_row, names, top_n=5):
    contributions = list(zip(names, shap_row))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


def predict_attrition(inputs: dict):
    input_dict = {col: 0 for col in feature_columns}

    input_dict["Age"] = inputs["age"]
    input_dict["MonthlyIncome"] = inputs["monthly_income"]
    input_dict["JobSatisfaction"] = inputs["job_satisfaction"]
    input_dict["YearsAtCompany"] = inputs["years_at_company"]
    input_dict["DistanceFromHome"] = inputs["distance_from_home"]
    input_dict["NumCompaniesWorked"] = inputs["num_companies_worked"]
    input_dict["WorkLifeBalance"] = inputs["work_life_balance"]
    input_dict["YearsSinceLastPromotion"] = inputs["years_since_last_promotion"]
    input_dict["IncomePerYearAtCompany"] = inputs["monthly_income"] / (inputs["years_at_company"] + 1)
    input_dict["PromotionGap"] = inputs["years_at_company"] - inputs["years_since_last_promotion"]

    if inputs["overtime"] == "Yes" and "OverTime_Yes" in input_dict:
        input_dict["OverTime_Yes"] = 1

    travel_col = f"BusinessTravel_{inputs['business_travel']}"
    if travel_col in input_dict:
        input_dict[travel_col] = 1

    role_col = f"JobRole_{inputs['job_role']}"
    if role_col in input_dict:
        input_dict[role_col] = 1

    dept_col = f"Department_{inputs['department']}"
    if dept_col in input_dict:
        input_dict[dept_col] = 1

    input_df = pd.DataFrame([input_dict])[feature_columns]
    input_scaled = scaler.transform(input_df)

    risk_prob = model.predict_proba(input_scaled)[0][1]

    shap_vals = explainer.shap_values(input_df)[0]
    top_feats = get_top_shap_features(shap_vals, feature_columns, top_n=5)

    return risk_prob, top_feats


st.title("📉 Employee Attrition Risk Predictor")
st.markdown(
    "Enter an employee's details to estimate their risk of leaving, with the top "
    "SHAP-based factors driving that specific prediction."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal")
    age = st.slider("Age", 18, 60, 30)
    distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 5)
    num_companies_worked = st.slider("Num Companies Worked", 0, 10, 1)

with col2:
    st.subheader("Role & Compensation")
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
    job_role = st.selectbox(
        "Job Role",
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative", "Manager",
         "Sales Representative", "Research Director", "Human Resources"],
    )
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])

with col3:
    st.subheader("Work Conditions")
    overtime = st.radio("OverTime", ["Yes", "No"], index=1, horizontal=True)
    business_travel = st.selectbox(
        "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=Very High)", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance (1=Bad, 4=Best)", 1, 4, 3)

st.subheader("Tenure")
tcol1, tcol2 = st.columns(2)
with tcol1:
    years_at_company = st.slider("Years at Company", 0, 40, 5)
with tcol2:
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 40, 1)

st.divider()

if st.button("Predict Attrition Risk", type="primary", use_container_width=True):
    inputs = dict(
        age=age,
        monthly_income=monthly_income,
        overtime=overtime,
        business_travel=business_travel,
        job_satisfaction=job_satisfaction,
        years_at_company=years_at_company,
        distance_from_home=distance_from_home,
        num_companies_worked=num_companies_worked,
        job_role=job_role,
        department=department,
        work_life_balance=work_life_balance,
        years_since_last_promotion=years_since_last_promotion,
    )

    risk_prob, top_feats = predict_attrition(inputs)

    result_col, chart_col = st.columns([1, 1.4])

    with result_col:
        if risk_prob >= 0.5:
            st.error(f"### ⚠️ Likely to leave\n**Attrition risk: {risk_prob*100:.1f}%**")
        else:
            st.success(f"### ✅ Likely to stay\n**Attrition risk: {risk_prob*100:.1f}%**")

        st.progress(min(int(risk_prob * 100), 100))

        st.markdown("**Top contributing factors:**")
        for name, val in top_feats:
            direction = "increases risk" if val > 0 else "decreases risk"
            clean_name = name.replace("_", " ")
            st.markdown(f"- **{clean_name}** — {direction} (impact: `{val:+.3f}`)")

    with chart_col:
        import matplotlib.pyplot as plt

        names = [f[0].replace("_", " ") for f in top_feats][::-1]
        values = [f[1] for f in top_feats][::-1]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on risk)")
        ax.set_title("Why this prediction")
        st.pyplot(fig)

st.divider()
st.caption(
    "Model: XGBoost trained on the IBM HR Analytics Employee Attrition dataset (1,470 employees), "
    "with SMOTE for class imbalance and SHAP for explainability. Not a substitute for HR judgment — "
    "use as a screening signal alongside manager input."
)