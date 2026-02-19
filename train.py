import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales_data_sample.csv", encoding="latin1")

print("Total rows:", len(df))

# =========================
# SELECT FEATURES
# =========================

features = [
    "QUANTITYORDERED",
    "PRICEEACH",
    "QTR_ID",
    "MONTH_ID",
    "YEAR_ID",
    "PRODUCTLINE",
    "COUNTRY",
    "DEALSIZE"
]

target = "SALES"

df = df[features + [target]]

# =========================
# HANDLE CATEGORICAL DATA
# =========================

categorical_cols = ["PRODUCTLINE", "COUNTRY", "DEALSIZE"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =========================
# TRAIN TEST SPLIT
# =========================

X = df.drop("SALES", axis=1)
y = df["SALES"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================

# =========================
# TRAIN MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# EVALUATION
# =========================

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# =========================
# SAVE MODEL
# =========================

# =========================
# SAVE MODEL
# =========================

import os

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/model.pkl")

# Save column order
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("\nModel and columns saved successfully!")


import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance_df.head(10))

plt.figure(figsize=(8,5))
plt.barh(feature_importance_df["feature"][:10], 
         feature_importance_df["importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.show()
