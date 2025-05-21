import pickle

import pandas as pd

# Load the saved model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    style_encoder, topic_encoder = pickle.load(f)

# ====== New User Input ======
# For example: someone studies 3 hours/day, 5 days/week, prefers 'reading'
hours_per_day = 3
days_per_week = 5
preferred_style = "reading"

# Encode the style input (convert text to number)
style_encoded = style_encoder.transform([preferred_style])[0]

# Combine inputs into a list for prediction
X_new = pd.DataFrame([{"hours_per_day": hours_per_day, "days_per_week": days_per_week, "style_encoded": style_encoded}])

# Make prediction
predicted_index = model.predict(X_new)[0]

# Decode the prediction (convert number back to label)
predicted_topic = topic_encoder.inverse_transform([predicted_index])[0]

print("🧠 Recommended Topic:", predicted_topic)

