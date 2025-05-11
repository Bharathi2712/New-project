import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

st.title("Customer Churn Prediction Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your customer churn CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Drop customerID if present
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['Churn'], inplace=True)
    df.dropna(inplace=True)

    # Encode Churn column
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        st.error("Dataset must contain a 'Churn' column.")
        st.stop()

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    # Features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select a model", 
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader(f"Model: {model_choice}")
    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    st.metric("ROC AUC Score", f"{roc_auc:.2f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_choice} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Feature importances (if tree-based model)
    if model_choice in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        top_features = X.columns[indices]
        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices], color="skyblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importances")
        st.pyplot(fig)

    # Additional Visuals
    st.subheader("Churn Visualizations")

    # 1. Churn Distribution
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    ax.set_title("Churn Distribution")
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height() + 5
        ax.annotate(percentage, (x, y))
    st.pyplot(fig)

    # 2. Boxplot of Tenure
    contract_col = [col for col in df.columns if 'Contract_' in col]
    if contract_col:
        fig, ax = plt.subplots()
        sns.boxplot(x='Churn', y='tenure', hue=contract_col[0], data=df, ax=ax)
        ax.set_title('Churn by Tenure and Contract Type')
        st.pyplot(fig)

    # 3. Violin plot for Monthly Charges
    fig, ax = plt.subplots()
    sns.violinplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
    ax.set_title("Churn by Monthly Charges")
    st.pyplot(fig)

    # 4. KDE for Total Charges
    fig, ax = plt.subplots()
    sns.kdeplot(df[df['Churn'] == 0]['TotalCharges'], label='Churn: No', shade=True, ax=ax)
    sns.kdeplot(df[df['Churn'] == 1]['TotalCharges'], label='Churn: Yes', shade=True, ax=ax)
    ax.set_title("Distribution of Total Charges by Churn")
    ax.legend()
    st.pyplot(fig)

    # 5. Correlation Matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file to begin.")
