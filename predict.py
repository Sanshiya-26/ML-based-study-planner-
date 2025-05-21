import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("model/model.pkl")
style_encoder, topic_encoder = joblib.load("model/encoders.pkl")

# Example user input
hours_per_day = 1
days_per_week = 1
preferred_style = "article"

# Encode style
style_encoded = style_encoder.transform([preferred_style])[0]

# Prepare input
X_new = pd.DataFrame([{
    "hours_per_day": hours_per_day,
    "days_per_week": days_per_week,
    "style_encoded": style_encoded
}])

# Predict
predicted_index = model.predict(X_new)[0]
predicted_topic = topic_encoder.inverse_transform([predicted_index])[0]

print("🧠 Recommended Topic:", predicted_topic)

# predict.py
print("Model Prediction Input Data:", X_new)
predicted_index = model.predict(X_new)[0]
print("Model Predicted Index:", predicted_index)

predicted_topic = topic_encoder.inverse_transform([predicted_index])[0]
print("Model Predicted Topic:", predicted_topic)
