import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'passed':        [0, 0, 0, 0, 1, 1, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['hours_studied']]
y = df['passed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[5]])

print("Prediction for 5 study hours:")
print("Pass" if prediction[0] == 1 else "Fail")