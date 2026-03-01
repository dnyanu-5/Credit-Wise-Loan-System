"""
Train Model Script for CreditWise Loan Approval System
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=" * 60)
print("CreditWise Loan Approval - Model Training")
print("=" * 60)

# Load data
print("\n Loading dataset...")
data = pd.read_csv("data/loan_approval_data.csv")
print(f" Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# Drop missing values
print("\n Cleaning data...")
original_rows = data.shape[0]
data = data.dropna()
print(f" Removed {original_rows - data.shape[0]} rows with missing values")
print(f" Remaining records: {data.shape[0]}")

# Feature Engineering
print("\n Creating new features...")
data['DTI_Ratio_sq'] = data['DTI_Ratio'] ** 2
data['Credit_Score_sq'] = data['Credit_Score'] ** 2
print(" Added DTI_Ratio_sq and Credit_Score_sq")

# Encode categorical variables
print("\n Encoding categorical variables...")
categorical_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 
                   'Property_Area', 'Education_Level', 'Gender', 'Employer_Category']

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print(f" Encoding complete: {data_encoded.shape[1]} total features")

# Prepare features and target
print("\n Preparing features and target...")
X = data_encoded.drop(['Applicant_ID', 'Loan_Approved'], axis=1)
y = data_encoded['Loan_Approved'].map({'Yes': 1, 'No': 0})

print(f" Features: {X.shape[1]} columns")
print(f" Target distribution:")
print(f"   - Approved: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"   - Rejected: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

# Train-test split
print("\n Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Training set: {X_train.shape[0]} samples")
print(f" Test set: {X_test.shape[0]} samples")

# Scale features
print("\n Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Features scaled using StandardScaler")

# Train model
print("\n Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print(" Model training complete!")

# Evaluate model
print("\n Evaluating model performance...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

print("\n" + "=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall:    {recall:.2f}%")
print(f"F1-Score:  {f1:.2f}%")
print("=" * 60)

# Save model and scaler
print("\n Saving model and scaler...")
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(" Model saved as 'loan_model.pkl'")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(" Scaler saved as 'scaler.pkl'")

print("\n" + "=" * 60)
print(" Training Complete! Model ready for deployment.")
print("=" * 60)
print("\n Next: Run 'streamlit run app.py' to start the web app!")