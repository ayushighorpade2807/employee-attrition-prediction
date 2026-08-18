# Employee Attrition Risk Prediction

An end-to-end machine learning project predicting employee attrition risk on the IBM HR Analytics dataset — covering EDA, feature engineering, class imbalance handling, model comparison, explainability, and deployment as an interactive Streamlit app.

## 🔍 Overview

Employee attrition costs companies significant time and money in re-hiring and lost productivity. This project builds a classification model to flag employees at high risk of leaving, and — just as importantly — explains *why* each prediction was made, so HR teams get an actionable signal instead of a black-box number.

## 📊 Dataset

**IBM HR Analytics Employee Attrition & Performance dataset**
- 1,470 employees, 35 original features
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Target: `Attrition` (Yes/No) — 16.1% of employees left, 83.9% stayed (imbalanced)

## 🧭 Project Workflow

1. **Data Cleaning** — dropped constant/identifier columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`); confirmed no missing values or duplicates.
2. **Exploratory Data Analysis** — analyzed attrition rate by OverTime, BusinessTravel, JobRole, and MaritalStatus; examined numeric distributions (Age, Income, Tenure) by attrition status; checked correlations to flag multicollinearity before modeling.
3. **Feature Engineering** — created `IncomePerYearAtCompany` (income normalized by tenure) and `PromotionGap` (years at company minus years since last promotion) as proxies for compensation fairness and promotion stagnation.
4. **Class Imbalance Handling** — applied SMOTE to the training set only, after the train/test split, to avoid leaking synthetic data into evaluation.
5. **Modeling** — trained and compared Logistic Regression (baseline, `class_weight='balanced'`) against XGBoost.
6. **Evaluation** — ROC-AUC, classification reports, confusion matrices, and ROC curves for both models.
7. **Explainability** — SHAP TreeExplainer for global feature importance and per-prediction breakdowns.
8. **Deployment** — Streamlit web app for real-time risk scoring with SHAP-based explanations.

## 📈 Key Findings

- **OverTime is the strongest single driver of attrition**: employees working overtime leave at ~30.5%, roughly 3x the rate of those who don't (~10.4%).
- **Frequent travelers churn more**: `Travel_Frequently` employees leave at ~24.9% vs ~15.0% for `Travel_Rarely` and ~8.0% for `Non-Travel`.
- **Sales Representatives have the highest attrition of any role** (~39.8%), while Research Director and Manager roles are the most stable (under 5%).
- **Leavers are younger, less tenured, and lower paid on average** than employees who stayed.
- **Logistic Regression slightly outperformed XGBoost** on this dataset (~0.79 vs ~0.77 test ROC-AUC) with better recall on the minority "Left" class — a reminder that more complex models don't automatically win on small tabular datasets.

## 🐛 A Bug I Found and Fixed

During cross-validation, applying SMOTE to the *entire* training set before running `cross_val_score` caused synthetic oversampled points to leak across folds — inflating the reported CV ROC-AUC to an unrealistic ~0.98. I identified this as a methodology error, not a real result, and fixed it by wrapping SMOTE inside an `imblearn` Pipeline so it's refit fresh within each fold's training split only. The corrected, honest CV ROC-AUC is ~0.80, consistent with the single train/test split result — a much more trustworthy number to report.

## 🧠 Model Performance

| Model                |                     Test ROC-AUC                    |     Notes |

| Logistic 
Regression             |                        ~0.79                        | `class_weight='balanced'`, better recall on leavers |
| XGBoost              |                         ~0.77                       | `n_estimators=200, max_depth=4, learning_rate=0.1` |
| XGBoost 
(5-fold CV,
leakage-corrected)      |                      ~0.80 (± 0.04)                 |     SMOTE applied inside each fold via pipeline |

XGBoost was kept as the deployed model for its SHAP compatibility and ability to capture feature interactions.

## 🖥️ Streamlit App

An interactive app where you input an employee's details and get:
- A risk probability (Likely to leave / Likely to stay)
- A visual risk gauge
- The top 5 SHAP-driven factors behind that specific prediction, with direction (increases/decreases risk)

## 📁 Repository Structure

```
├── Employee_Attrition_Risk_Corrected.ipynb   # Full analysis notebook (EDA → modeling → SHAP)
├── streamlit_app.py                          # Deployed prediction app
├── attrition_model.pkl                       # Trained XGBoost model
├── attrition_scaler.pkl                      # Fitted StandardScaler
├── attrition_features.pkl                    # Ordered feature column list
├── shap_summary_attrition.png                # Global SHAP feature importance plot
└── README.md
```

## ⚙️ How to Run

1. Clone this repo and download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), placing it in the project folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap joblib streamlit
   ```
3. Run the notebook top to bottom to reproduce the analysis and regenerate the model artifacts:
   ```bash
   jupyter notebook Employee_Attrition_Risk_Corrected.ipynb
   ```
4. Launch the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Matplotlib · Seaborn · Streamlit · Joblib

## 🔮 Possible Extensions

- Hyperparameter tuning (GridSearchCV) for both models
- Threshold tuning to weigh false negatives vs false positives based on business cost
- A written business-recommendations section translating findings into HR action items

## ⚠️ Disclaimer

This is a portfolio/learning project using a public, synthetic dataset. It is not validated for real-world HR decision-making and should not be used to make actual employment decisions.
