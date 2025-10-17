# -----------------------------------------------------------
# Section 1: Setup and Data Loading
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
# Make sure your advertising.csv file is in the same folder as this script.
try:
    data = pd.read_csv('advertising.csv')
    print("Data loaded successfully.")
    print("\nSample Data:")
    print(data.head())
    print("\nDataFrame Info:")
    data.info()
except FileNotFoundError:
    print("Error: The 'advertising.csv' file was not found. Please make sure it is in the same directory as the script.")
    exit()

# -----------------------------------------------------------
# Section 2: Data Preprocessing
# -----------------------------------------------------------

# Define features (X) and target (y)
# The 'User_ID' column is not useful for modeling, so we drop it.
# New, correct line
X = data.drop(columns=['Clicked on Ad', 'Timestamp'])# New, correct line
y = data['Clicked on Ad']
# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nNumerical features: {list(numerical_features)}")
print(f"Categorical features: {list(categorical_features)}")

# Create preprocessing pipelines for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Use ColumnTransformer to apply the correct transformations to the right columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} records")
print(f"Testing set size: {len(X_test)} records")

# -----------------------------------------------------------
# Section 3: Model Building and Training
# -----------------------------------------------------------

print("\nBuilding and training the Logistic Regression model...")

# Create the full pipeline including preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete.")

# -----------------------------------------------------------
# Section 4: Model Evaluation
# -----------------------------------------------------------

print("\nEvaluating the model performance on the test set...")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate key evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

# Plot the distribution of predicted probabilities
plt.figure(figsize=(10, 6))
sns.histplot(y_proba, kde=True, bins=50)
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability of Click')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('probability_distribution.png') 
plt.show()

# -----------------------------------------------------------
# Section 5: Model Deployment & Interpretation
# -----------------------------------------------------------

print("\nSaving the trained model...")
joblib.dump(model_pipeline, 'ctr_prediction_model.joblib')
print("Model saved as 'ctr_prediction_model.joblib'.")

print("\nEnd of project script.")