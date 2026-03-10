# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020],
    "Mileage": [50000, 40000, 30000, 20000, 15000, 10000],
    "Price": [500000, 550000, 600000, 650000, 700000, 750000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["Year", "Mileage"]]
y = df["Price"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict price
predicted_price = model.predict(X_test)

# Display results
print("Actual Price:", list(y_test))
print("Predicted Price:", predicted_price)