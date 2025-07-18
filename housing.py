from sklearn.datasets import fetch_california_housing
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
