# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    "Advertising": [100, 200, 300, 400, 500, 600],
    "Sales": [10, 20, 25, 35, 45, 55]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["Advertising"]]
y = df["Sales"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Display results
print("Actual Sales:", list(y_test))
print("Predicted Sales:", y_pred)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))