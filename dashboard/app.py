import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 AI-Based Disease Prediction")
st.write(
    "Select your symptoms and the machine learning model "
    "will predict the possible disease."
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("../models/disease_prediction_model.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("../dataset/Training.csv")

symptoms = data.drop(
    columns=["prognosis", "Unnamed: 133"]
).columns.tolist()

# -----------------------------
# Symptom Selection
# -----------------------------
st.subheader("🔍 Select Symptoms")

selected_symptoms = st.multiselect(
    "Choose your symptoms:",
    symptoms
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Disease"):

    if not selected_symptoms:

        st.warning("Please select at least one symptom.")

    else:

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=model.feature_names_in_
        )

        for symptom in selected_symptoms:
            input_data[symptom] = 1

        prediction = model.predict(input_data)

        predicted_disease = prediction[0]

        st.success(
            f"Predicted Disease: {predicted_disease}"
        )

        # Show selected symptoms
        st.subheader("Selected Symptoms")

        for symptom in selected_symptoms:
            st.write("✓", symptom)

# -----------------------------
# Disclaimer
# -----------------------------
st.warning(
    "⚠️ This project is for educational purposes only "
    "and should not be used as a medical diagnosis."
)