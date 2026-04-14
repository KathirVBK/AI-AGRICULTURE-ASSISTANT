"""
AgriSense-AI — test_data.py
"""

import pandas as pd

df = pd.read_csv("data/sensor_Crop_Dataset (1).csv")

print("=== ALL COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== FIRST 3 ROWS ===")
print(df.head(3))

print("\n=== CROP COLUMN VALUES ===")
print(df.iloc[:, -1].value_counts())   # last column
