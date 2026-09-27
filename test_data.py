import pandas as pd

# Load the student dataset
data = pd.read_csv("student_performance.csv")

# Display the first 5 rows
print(data.head())

# Display dataset information
print("\nDataset Shape:", data.shape)
print("\nRisk Distribution:")
print(data["risk"].value_counts())