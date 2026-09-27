import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("student_performance.csv")

# Features used for prediction
X = data[
    [
        "attendance",
        "internal_marks",
        "assignment_score",
        "study_hours",
        "gpa",
        "previous_failures"
    ]
]

# Target variable
y = data["risk"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Save the trained model
joblib.dump(model, "student_risk_model.pkl")

print("\nModel saved successfully!")