# STEP 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from xgboost import XGBRegressor


# STEP 2: Load Dataset

data = pd.read_csv(
    "Skyserver_CrossID5_5_2026 11_07_03 AM.csv",
    comment='#',
    skiprows=1
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

print("Columns:", data.columns)
print(data.head())


# STEP 3: Select Required Columns

data = data[["u", "g", "r", "i", "z", "redshift"]]


# STEP 4: Clean Data

data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()


# STEP 5: Remove Extreme Outliers

data = data[data["redshift"] < 1.5]


# STEP 6: Feature Engineering

data["u_g"] = data["u"] - data["g"]
data["g_r"] = data["g"] - data["r"]
data["r_i"] = data["r"] - data["i"]
data["i_z"] = data["i"] - data["z"]

data["u_r"] = data["u"] - data["r"]
data["g_i"] = data["g"] - data["i"]
data["r_z"] = data["r"] - data["z"]
data["u_i"] = data["u"] - data["i"]
data["g_z"] = data["g"] - data["z"]


# STEP 7: Define Features & Target

X = data.drop("redshift", axis=1)
y = np.log1p(data["redshift"])   # log transform


# STEP 8: Polynomial Features

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


# STEP 9: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)


# STEP 10: Train Model

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1,
    random_state=42
)

model.fit(X_train, y_train)


# STEP 11: Predictions

y_pred = model.predict(X_test)

# Convert back to original scale
y_pred = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

# Ensure no negative predictions
y_pred = np.clip(y_pred, 0, None)


# STEP 12: Evaluation

mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)


# STEP 13: Visualization

plt.figure()
plt.scatter(y_test_actual, y_pred)
plt.plot([0, 1.5], [0, 1.5])  # ideal line
plt.xlabel("Actual Redshift")
plt.ylabel("Predicted Redshift")
plt.title("Actual vs Predicted")
plt.show()


# STEP 14: Sample Predictions

comparison = pd.DataFrame({
    "Actual": y_test_actual.values,
    "Predicted": y_pred
})

print("\nSample Predictions:")
print(comparison.head())