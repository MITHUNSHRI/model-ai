import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load data
data = pd.read_csv("customers.csv")

# Encode text to numbers
data["industry"] = data["industry"].astype("category").cat.codes
data["role"] = data["role"].astype("category").cat.codes
data["interest"] = data["interest"].astype("category").cat.codes


# Input and Output
X = data[["industry", "role", "interest"]]
y = data["response"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Check accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy * 100, "%")


# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
