# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'RAM': [2, 3, 4, 6, 8, 12, 16],
    'Storage': [16, 32, 64, 64, 128, 256, 512],
    'Battery': [3000, 3200, 3500, 4000, 4500, 5000, 6000],
    'Price': [7000, 9000, 12000, 18000, 25000, 35000, 45000]
}

# Convert dataset into DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[['RAM', 'Storage', 'Battery']]
y = df['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Display results
print("Actual Prices:", list(y_test))
print("Predicted Prices:", y_pred)

# Model evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))