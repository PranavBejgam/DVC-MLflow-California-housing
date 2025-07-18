from sklearn.datasets import fetch_california_housing
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


os.makedirs("data", exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("data/raw_data.csv", index=False)

preprocess_version = os.getenv("PREPROCESS_VERSION", "v2")

if preprocess_version == "v1":
    df_processed = df.copy()
    scaler = StandardScaler()
    df_processed[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
elif preprocess_version == "v2":
    df_processed = df.copy()
    scaler = MinMaxScaler()
    df_processed[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
else:
    raise ValueError("Invalid PREPROCESS_VERSION. Use 'v1' or 'v2'.")

df_processed.to_csv(f"data/processed_{preprocess_version}.csv", index=False)

X = df_processed.drop("MedHouseVal", axis=1)
y = df_processed["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mlflow.set_experiment("CaliforniaHousing_Regression")

with mlflow.start_run():
    mlflow.log_param("preprocessing_version", preprocess_version)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact(f"data/processed_{preprocess_version}.csv")

