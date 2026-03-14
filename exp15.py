# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Area": [1000, 1500, 1800, 2000, 2500, 3000],
    "Bedrooms": [2, 3, 3, 4, 4, 5],
    "Price": [200000, 300000, 350000, 400000, 450000, 500000]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house price
predicted_price = model.predict(X_test)

# Display results
print("Actual Price:", list(y_test))
print("Predicted Price:", predicted_price)