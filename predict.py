import joblib
import pandas as pd
import streamlit as st

model = joblib.load("model/model.pkl")
style_encoder, topic_encoder = joblib.load("model/encoders.pkl")

hours_per_day = 1
days_per_week = 1
preferred_style = "article"

style_encoded = style_encoder.transform([preferred_style])[0]

X_new = pd.DataFrame([{
    "hours_per_day": hours_per_day,
    "days_per_week": days_per_week,
    "style_encoded": style_encoded
}])

predicted_index = model.predict(X_new)[0]
predicted_topic = topic_encoder.inverse_transform([predicted_index])[0]

st.title("🧠 Recommended Topic")
st.write(predicted_topic)
st.subheader("Model Prediction Input Data")
st.dataframe(X_new)
st.write("Model Predicted Index:", predicted_index)
st.write("Model Predicted Topic:", predicted_topic)
