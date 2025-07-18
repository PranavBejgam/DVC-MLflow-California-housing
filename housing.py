from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("data/raw_data.csv", index=False)
