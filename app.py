# Tuberculosis Diagnosis Web App with Streamlit (Updated for Your Dataset)
# =====================================================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load your uploaded dataset
df = pd.read_csv("tb_datset.csv")

# Drop Patient_ID since it's not useful for prediction
df.drop(columns=["Patient_ID"], inplace=True)

# Encode categorical columns
categorical_cols = ['Gender', 'Chest_Pain', 'Fever', 'Night_Sweats', 'Sputum_Production',
                    'Blood_in_Sputum', 'Smoking_History', 'Previous_TB_History', 'Class']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
features = ['Age', 'Gender', 'Chest_Pain', 'Cough_Severity', 'Breathlessness',
            'Fatigue', 'Weight_Loss', 'Fever', 'Night_Sweats', 'Sputum_Production',
            'Blood_in_Sputum', 'Smoking_History', 'Previous_TB_History']
target = 'Class'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit Web App
st.title("Tuberculosis Diagnosis Predictor")
st.write("This app predicts the likelihood of TB based on patient symptoms and history.")

# User input widgets
def user_input_features():
    age = st.slider('Age', 0, 100, 25)
    gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
    chest_pain = st.selectbox('Chest Pain', label_encoders['Chest_Pain'].classes_)
    cough_severity = st.slider('Cough Severity (0-10)', 0, 10, 5)
    breathlessness = st.slider('Breathlessness (0-10)', 0, 10, 5)
    fatigue = st.slider('Fatigue (0-10)', 0, 10, 5)
    weight_loss = st.slider('Weight Loss (0-20)', 0.0, 20.0, 5.0)
    fever = st.selectbox('Fever', label_encoders['Fever'].classes_)
    night_sweats = st.selectbox('Night Sweats', label_encoders['Night_Sweats'].classes_)
    sputum_production = st.selectbox('Sputum Production', label_encoders['Sputum_Production'].classes_)
    blood_in_sputum = st.selectbox('Blood in Sputum', label_encoders['Blood_in_Sputum'].classes_)
    smoking_history = st.selectbox('Smoking History', label_encoders['Smoking_History'].classes_)
    previous_tb = st.selectbox('Previous TB History', label_encoders['Previous_TB_History'].classes_)

    data = {
        'Age': age,
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'Chest_Pain': label_encoders['Chest_Pain'].transform([chest_pain])[0],
        'Cough_Severity': cough_severity,
        'Breathlessness': breathlessness,
        'Fatigue': fatigue,
        'Weight_Loss': weight_loss,
        'Fever': label_encoders['Fever'].transform([fever])[0],
        'Night_Sweats': label_encoders['Night_Sweats'].transform([night_sweats])[0],
        'Sputum_Production': label_encoders['Sputum_Production'].transform([sputum_production])[0],
        'Blood_in_Sputum': label_encoders['Blood_in_Sputum'].transform([blood_in_sputum])[0],
        'Smoking_History': label_encoders['Smoking_History'].transform([smoking_history])[0],
        'Previous_TB_History': label_encoders['Previous_TB_History'].transform([previous_tb])[0]
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Prediction")
st.write(label_encoders['Class'].inverse_transform(prediction)[0])

st.subheader("Prediction Probability")
st.write({
    label_encoders['Class'].inverse_transform([0])[0]: prediction_proba[0][0],
    label_encoders['Class'].inverse_transform([1])[0]: prediction_proba[0][1]
})

st.markdown(f"### Model Accuracy: {accuracy:.2%}")
