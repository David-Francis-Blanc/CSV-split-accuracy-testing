import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("study_data.csv")
X = data[["Hours"]]
y = data["Score"]

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Show results
print("Actual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {pred:.2f}")

# Accuracy metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
