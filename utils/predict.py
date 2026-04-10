"""
AgriSense-AI — utils/predict.py
Utility script for standalone crop prediction testing.
"""

import pickle
import pandas as pd

# Load model and encoder using 'with' to ensure proper closure
with open("models/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Input must match TRAINING COLUMN NAMES
input_data = pd.DataFrame([{
    'Nitrogen': 70,
    'Phosphorus': 20,
    'Potassium': 40,
    'Temperature': 5,
    'Humidity': 70,
    'pH_Value': 6.5,
    'Rainfall': 200
}])

# Predict probabilities
probs = model.predict_proba(input_data)[0]

# Get top 3 predictions
top_indices = probs.argsort()[-3:][::-1]

crops = le.inverse_transform(top_indices)
confidences = probs[top_indices]

print("\n🌱 Top Crop Recommendations:\n")
for i, (crop, conf) in enumerate(zip(crops, confidences), 1):
    print(f"{i}. {crop.capitalize()} ({conf*100:.1f}%)")
