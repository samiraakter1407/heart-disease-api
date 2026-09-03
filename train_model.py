import os

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("heart.csv")


# Features used by the model
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Input features
X = df[FEATURES]

# Target
y = df["target"]


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Create the classifier
model = LogisticRegression(max_iter=1000)


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)


# Save the trained model
model_path = "model/heart_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved successfully to: {model_path}")