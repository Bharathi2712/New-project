import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# Upload CSV
print("Please upload the customer churn dataset CSV file:")
uploaded = files.upload()

# Read CSV
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Show first few rows
print("Dataset Preview:")
display(df.head())

# Data Cleaning
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with NaN values in the target 'Churn' column
df.dropna(subset=['Churn'], inplace=True)

# Drop rows with any remaining NaN values
df.dropna(inplace=True)

# Encode target 'Churn'
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
else:
    raise ValueError("The dataset must contain a 'Churn' column.")

# One-hot encoding for categorical features
df = pd.get_dummies(df, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in num_cols:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Check if y contains any NaN values
if y.isnull().sum() > 0:
    print("Target column 'Churn' contains NaN values. Dropping rows with NaN values in 'Churn'.")
    df.dropna(subset=['Churn'], inplace=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Plot ROC curves
plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature importance (Random Forest)
importances = models["Random Forest"].feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances - Random Forest")
plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# 1. Churn Distribution with Percentage
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.xlabel('Churn (0: No, 1: Yes)')
plt.ylabel('Count')

# Adding percentage labels
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_y() + p.get_height() + 5
    ax.annotate(percentage, (x, y), size=12)

plt.show()

# 2. Churn by Tenure with Hue for Contract Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', hue='Contract_Two year', data=df)  # Example hue: Contract type
plt.title('Churn by Tenure and Contract Type')
plt.xlabel('Churn (0: No, 1: Yes)')
plt.ylabel('Tenure (Months)')
plt.show()


# 3. Churn by Monthly Charges with Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Churn by Monthly Charges')
plt.xlabel('Churn (0: No, 1: Yes)')
plt.ylabel('Monthly Charges')
plt.show()


# 4. Churn by Total Charges with Distribution (Kernel Density Estimation)
plt.figure(figsize=(8, 6))
sns.kdeplot(df[df['Churn'] == 0]['TotalCharges'], label='Churn: No', shade=True)
sns.kdeplot(df[df['Churn'] == 1]['TotalCharges'], label='Churn: Yes', shade=True)
plt.title('Distribution of Total Charges by Churn')
plt.xlabel('Total Charges')
plt.ylabel('Density')
plt.legend()
plt.show()


# 5. Correlation Matrix with Improved Readability
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels vertical
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
