Employee Attrition Prediction

Attrition prediction using ML/DL (XGBoost, PyTorch), SHAP Explainability, Gradio app demo, and Power BI dashboard.

Overview

Attrition costs organizations money; this project aims to develop a pipeline that would predict which employees will attrit, based on structured HR data. In addition to making predictions, this project aims to explain the results (using SHAP), so that the HR team knows why the model thinks a particular employee will leave, rather than just whether or not he or she will.

This project includes:

- a classical ML model (XGBoost) and DL model (PyTorch) for comparison purposes;
- SHAP for explainability;
- Gradio app for real-time predictions;
- Power BI dashboard to visualize trends in employee attrition.

- Statement of Problem

In most cases, organizations have been found having difficulties identifying potential leavers. In this project, employee attrition will be viewed as a classification problem whereby historical human resources data will be used to detect individuals who are likely to leave. Also, it will be explained the reason why such employees are flagged as such by HR.

Data Set

Source: IBM HR Analytics Employee Attrition & Performance (Kaggle)
Size: 1,470 Employees, 35 Features (31 Features after dropping constant/ID variables)
Target Variable: Attrition (Yes/No) - Imbalanced (about 84% No / 16% Yes)
Key Variables: Age, Monthly Income, OverTime, Job Satisfaction, Years At Company, Distance From Home, Job Role, Department, Business Travel, Work Life Balance, Number Of Companies Worked, Stock Option Level

Approach

Data cleaning and preprocessing – removed constant / id columns, one hot encoded categorical features, scaled numeric features
Exploratory Data Analysis (EDA) – uncovered the factors driving attrition (overtime, job satisfaction, years with company, salary)
Feature engineering – calculated features like salary per year with company and promotion gaps
Class imbalance – used SMOTE only on the training dataset to prevent leaking information into the test set
Modeling – built, trained, and compared Logistic regression, Random Forest, XGBoost and PyTorch Neural Network models
Evaluation – evaluated model performance through ROC-AUC, precision and recall metrics (not accuracy because of class imbalance)
Explainability – applied SHAP to understand the importance of features and the reasoning behind predictions
Deployment – created a Gradio UI for interactive and explainable risk scoring
Business reporting – generated a Power BI visualization of historical attrition patterns for management reporting purposes

SHAP analysis key takeaways:

Overtime, stock option amount, and prior experience in how many firms played important roles in predicting employee attrition.
High job satisfaction always decreased predicted attrition, while low job satisfaction always increased it.
Employees with shorter tenure had higher predicted attrition risk than those who worked longer.

Repository Structure
employee-attrition-prediction/
│
├── employee_attrition_risk.ipynb                     # Main notebook: preprocessing, modeling, SHAP, Gradio app
├── Predicting & Visualizing Employee Attrition.pbix   # Power BI dashboard
├── README.md
└── .gitignore

How to Run

Clone the Repository
git clone https://github.com/ayushighorpade2807/employee-attrition-prediction.git
   cd employee-attrition-prediction

Install Dependencies
pip install -r requirements.txt

Download the dataset from Kaggle (link above) and place the CSV in the project folder, or upload it when prompted in the notebook.
Run the notebook
Open employee_attrition_risk.ipynb in Jupyter or Google Colab and run all cells — the final cells launch the Gradio app.
Open the Power BI dashboard
Open Predicting & Visualizing Employee Attrition.pbix in Power BI Desktop.

Future Improvements

 Hyperparameter tuning (GridSearchCV / Optuna) for improved model performance
 K-fold cross-validation instead of a single train/test split
 Deploy Gradio app permanently to Hugging Face Spaces
 Fairness audit across demographic groups (gender, age)
 Time-based train/test split to better simulate real-world deployment
