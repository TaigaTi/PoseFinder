import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from export_data import classification_report_export
from joblib import dump
import os


# Load dataset
df = pd.read_csv("pose_dataset.csv")

# Split features and labels
X = df.iloc[:, :-1].values  # keypoints
y = df.iloc[:, -1].values   # labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"\nTest Accuracy: {acc:.2f}")

# Classification report
y_pred_labels = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# Export to Excel
report_path = 'results/results.xlsx'
classification_report_export(classification_report(y_test, y_pred_labels, output_dict=True), report_path)

# Save the Model
os.makedirs('model', exist_ok=True)
dump(model, 'model/rf_pose_classifier.joblib')
