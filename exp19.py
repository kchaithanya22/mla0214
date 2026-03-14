# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'Income': [25000, 30000, 40000, 50000, 60000, 70000, 80000],
    'CreditScore': [600, 650, 700, 720, 750, 780, 800],
    'LoanApproved': [0, 0, 1, 1, 1, 1, 1]   # 0 = No, 1 = Yes
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Income', 'CreditScore']]
y = df['LoanApproved']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("Actual:", list(y_test))
print("Predicted:", y_pred)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))