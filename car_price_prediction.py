import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")
print("Dataset loaded successfully!")
print(data.head())

# Preprocess data
data = pd.get_dummies(data, columns=['Brand','Fuel'], drop_first=True)

X = data.drop('Price', axis=1)
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature importance visualization
importance = model.coef_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance in Car Price Prediction")
plt.show()

# User input prediction
def predict_price():
    print("\n--- Car Price Prediction ---")
    year = int(input("Enter car year: "))
    mileage = int(input("Enter mileage (km): "))
    fuel = input("Enter fuel type (Petrol/Diesel): ")
    brand = input("Enter brand (e.g., Toyota, Honda): ")

    input_data = pd.DataFrame([[year, mileage, fuel, brand]], 
                              columns=['Year','Mileage','Fuel','Brand'])

    input_data = pd.get_dummies(input_data, columns=['Brand','Fuel'])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    predicted_price = model.predict(input_data)[0]
    print(f"Estimated Price: ₹{predicted_price:,.2f}")

predict_price()