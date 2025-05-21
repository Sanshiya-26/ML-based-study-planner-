import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv(r'/Users/sanshiyajagroo/PycharmProjects/ml_study_planner/dataa/study_logs.csv')

# Encode text
le_style = LabelEncoder()
le_topic = LabelEncoder()

df['style_encoded'] = le_style.fit_transform(df['preferred_style'])
df['target'] = le_topic.fit_transform(df['recommended_topic'])

# Features + target
X = df[['hours_per_day', 'days_per_week', 'style_encoded']]
y = df['target']

# Train a model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump((le_style, le_topic), f)

print("✅ Model trained and saved.")
