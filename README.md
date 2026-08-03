# 📉 Customer Churn Prediction ML Pipeline

> **This an end-to-end customer churn prediction pipeline using Scikit-learn — comparing Logistic Regression, Decision Tree, and Random Forest models with feature engineering on 7,043 real Telco customer records, achieving 80.4% accuracy and ROC-AUC of 0.845.**

---

## 📌 Project Overview

Customer churn — when a customer stops using a service — is one of the most costly problems in business. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a complete machine learning pipeline to predict which customers are likely to churn, enabling businesses to take proactive retention action.

```
Raw Telco Customer Data (7,043 records)
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Feature Engineering (3 new features)
        │
        ▼
Train/Test Split + Standard Scaling
        │
        ▼
Train 3 ML Models
├── Logistic Regression
├── Decision Tree
└── Random Forest
        │
        ▼
Model Comparison (Accuracy + ROC-AUC)
        │
        ▼
Feature Importance Analysis
└── Top churn predictors identified
```

---

## 🎯 Business Questions Answered

- Which customers are most likely to churn?
- What are the top factors that drive customer churn?
- Which machine learning model best predicts churn?
- How can the business prioritize retention efforts?

---

## 📂 Dataset

**Source:** IBM Telco Customer Churn Dataset
**Link:** [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Field | Description |
|---|---|
| customerID | Unique customer identifier |
| tenure | Months the customer has been with the company |
| MonthlyCharges | Monthly bill amount |
| TotalCharges | Total amount charged |
| Contract | Month-to-month, one year, two year |
| Churn | Yes / No — target variable |

**Churn Rate in Dataset: 26.5%** — 1 in 4 customers leaves.

---

## 📁 Repository Structure

```
customer-churn-prediction/
│
├── data/                               
├── plots/                              ← Generated visualizations
│   ├── churn_distribution.png
│   ├── monthly_charges_by_churn.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── churn_prediction.py                 ← Main ML pipeline script
├── requirements.txt                    ← Python dependencies
└── README.md
```

---

## 🔧 Feature Engineering

Three new features were engineered to improve model performance:

| Feature | Description | Business Meaning |
|---|---|---|
| `charges_per_tenure` | TotalCharges / (tenure + 1) | Revenue per month of relationship |
| `is_new_customer` | tenure < 12 months → 1 | New customers churn more |
| `is_long_term` | tenure > 24 months → 1 | Long-term customers churn less |

---

## 🤖 Models Trained & Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Logistic Regression** | **80.4%** ✅ Best | **0.845** ✅ Best |
| Random Forest | 79.4% | 0.840 |
| Decision Tree | 78.6% | 0.823 |

**Best Model: Logistic Regression** — 80.4% accuracy with ROC-AUC of 0.845

---

## 📊 Classification Report (Best Model)

```
              precision    recall  f1-score   support

    No Churn       0.84      0.90      0.87      1035
       Churn       0.66      0.53      0.59       374

    accuracy                           0.80      1409
   macro avg       0.75      0.72      0.73      1409
weighted avg       0.79      0.80      0.80      1409
```

---

## 🔍 Key Findings

- **Churn rate is 26.5%** — significant business problem worth solving
- **Monthly charges** is the strongest predictor of churn — higher bills = more churn
- **Tenure** is the second strongest predictor — newer customers churn more
- **Contract type** matters — month-to-month customers churn significantly more than annual contract customers
- **Logistic Regression outperforms** tree-based models on this dataset — linear decision boundaries work well for this problem

---

## 📈 Visualizations Generated

| Plot | Description |
|---|---|
| `churn_distribution.png` | Bar chart of churn vs no churn counts |
| `monthly_charges_by_churn.png` | Distribution of monthly charges by churn status |
| `model_comparison.png` | Side-by-side accuracy and ROC-AUC comparison |
| `feature_importance.png` | Top 10 most important features from Random Forest |

---

## 🚀 How to Run

### Prerequisites
```cmd
pip install -r requirements.txt
```

### Step 1 — Download the dataset
👉 [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place in: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Step 2 — Run the pipeline
```cmd
python churn_prediction.py
```

### Expected output
```
============================================================
  CUSTOMER CHURN PREDICTION PIPELINE
============================================================

📊 STEP 1: Loading and exploring data...
  Shape:  (7043, 21)
  Churn rate: 26.5%

🔧 STEP 2: Feature Engineering...
  3 new features created

✂️  STEP 3: Splitting data...
  Training: 5,634 | Test: 1,409

🤖 STEP 4: Training ML models...
  Logistic Regression: 80.4% | ROC-AUC: 0.845
  Decision Tree:       78.6% | ROC-AUC: 0.823
  Random Forest:       79.4% | ROC-AUC: 0.840

🏆 STEP 6: Best model: Logistic Regression

============================================================
  ✅ PIPELINE COMPLETE
============================================================
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git / GitHub |

---

## 📈 Skills Demonstrated

- End-to-end machine learning pipeline development
- Exploratory data analysis (EDA) on real customer data
- Feature engineering — creating business-meaningful derived features
- Multi-model training and comparison (Logistic Regression, Decision Tree, Random Forest)
- Model evaluation — accuracy, ROC-AUC, precision, recall, F1-score
- Feature importance analysis and business insight generation
- Data visualization — distribution plots, model comparison charts

---

## 👤 Author

**Jean Pierre Idi**
M.S. Business Informatics — Northern Kentucky University (2025)
📧 jeanpierreidi1@gmail.com | 🔗 github.com/jeanpierreidi1
