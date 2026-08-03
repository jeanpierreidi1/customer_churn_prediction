# churn_prediction.py
# Customer Churn Prediction — End to End ML Pipeline
# Scikit-learn | Pandas | NumPy | Matplotlib


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r'C:\Users\jeanp\customer_churn_prediction')
print(f"Working directory: {os.getcwd()}")

print("="*60)
print("  CUSTOMER CHURN PREDICTION PIPELINE")
print("  Scikit-learn | Pandas | NumPy | Matplotlib")
print("="*60)


# ════════════════════════════════════════════════════════════
# STEP 1: Load & Explore Data (EDA)
# ════════════════════════════════════════════════════════════
print("\n📊 STEP 1: Loading and exploring data...")

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"\n  Shape:          {df.shape}")
print(f"  Columns:        {df.columns.tolist()}")
print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n  Churn distribution:")
print(df['Churn'].value_counts())
print(f"\n  Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")

# Plot churn distribution
plt.figure(figsize=(6, 4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue','tomato'])
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/churn_distribution.png')
plt.close()
print("\n  ✅ Churn distribution plot saved")

# Plot monthly charges by churn
plt.figure(figsize=(8, 4))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
for label, group in df.groupby('Churn')['MonthlyCharges']:
    plt.hist(group, alpha=0.5, label=label, bins=30)
plt.title('Monthly Charges by Churn Status')
plt.xlabel('Monthly Charges')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plots/monthly_charges_by_churn.png')
plt.close()
print("  ✅ Monthly charges plot saved")


# ════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering
# ════════════════════════════════════════════════════════════
print("\n🔧 STEP 2: Feature Engineering...")

# Drop customer ID — not useful for prediction
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Engineer new features
# Charges per month of tenure
df['charges_per_tenure'] = (df['TotalCharges'] /
                             (df['tenure'] + 1))

# Is customer new (less than 12 months)
df['is_new_customer'] = (df['tenure'] < 12).astype(int)

# Is customer long-term (more than 24 months)
df['is_long_term'] = (df['tenure'] > 24).astype(int)

print(f"  New features created:")
print(f"  - charges_per_tenure")
print(f"  - is_new_customer")
print(f"  - is_long_term")

# Encode categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Don't encode target yet

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Encode target
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

print(f"\n  Encoded {len(categorical_cols)} categorical columns")
print(f"  Final feature count: {df.shape[1] - 1}")
print("  ✅ Feature engineering complete")


# ════════════════════════════════════════════════════════════
# STEP 3: Train / Test Split
# ════════════════════════════════════════════════════════════
print("\n✂️  STEP 3: Splitting data...")

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"  Training set:   {X_train.shape[0]:,} records")
print(f"  Test set:       {X_test.shape[0]:,} records")
print(f"  Features:       {X_train.shape[1]}")


# ════════════════════════════════════════════════════════════
# STEP 4: Train 3 ML Models
# ════════════════════════════════════════════════════════════
print("\n🤖 STEP 4: Training ML models...")

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000
    ),
    'Decision Tree':       DecisionTreeClassifier(
        random_state=42, max_depth=5
    ),
    'Random Forest':       RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=5
    )
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)

    results[name] = {
        'model':    model,
        'accuracy': acc,
        'auc':      auc,
        'y_pred':   y_pred
    }

    print(f"\n  {name}:")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    ROC-AUC:   {auc:.3f}")


# ════════════════════════════════════════════════════════════
# STEP 5: Compare Models
# ════════════════════════════════════════════════════════════
print("\n📊 STEP 5: Comparing model performance...")

# Bar chart comparison
model_names = list(results.keys())
accuracies  = [results[m]['accuracy']*100 for m in model_names]
aucs        = [results[m]['auc'] for m in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(model_names, accuracies, color=['steelblue','orange','green'])
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim([70, 100])
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

ax2.bar(model_names, aucs, color=['steelblue','orange','green'])
ax2.set_title('ROC-AUC Score Comparison')
ax2.set_ylabel('AUC Score')
ax2.set_ylim([0.5, 1.0])
for i, v in enumerate(aucs):
    ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.close()
print("  ✅ Model comparison plot saved")


# ════════════════════════════════════════════════════════════
# STEP 6: Best Model — Feature Importance
# ════════════════════════════════════════════════════════════
print("\n🏆 STEP 6: Analyzing best model...")

# Find best model by AUC
best_name  = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']

print(f"\n  Best model: {best_name}")
print(f"  Accuracy:   {results[best_name]['accuracy']*100:.1f}%")
print(f"  ROC-AUC:    {results[best_name]['auc']:.3f}")

# Classification report
print(f"\n  Classification Report:")
print(classification_report(
    y_test, results[best_name]['y_pred'],
    target_names=['No Churn', 'Churn']
))

# Feature importance (Random Forest)
rf_model = results['Random Forest']['model']
feature_names    = df.drop('Churn', axis=1).columns
feature_importance = pd.DataFrame({
    'feature':   feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'],
         feature_importance['importance'],
         color='steelblue')
plt.title('Top 10 Most Important Features — Random Forest')
plt.xlabel('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()
print("\n  ✅ Feature importance plot saved")

print("\n" + "="*60)
print("  ✅ PIPELINE COMPLETE")
print("  Plots saved to: plots/")
print("="*60)