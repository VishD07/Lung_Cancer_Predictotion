import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load trained model using relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
model = joblib.load(MODEL_PATH)

# Add custom CSS for sci-fi theme
st.markdown(
    """<style>
    body {
        background-color: #121212;
        color: #DADADA;
        font-family: 'Courier New', Courier, monospace;
    }
    .stButton>button {
        background-color: #6C63FF;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-family: 'Courier New', Courier, monospace;
    }
    .stButton>button:hover {
        background-color: #4E4BFF;
        cursor: pointer;
    }
    .stSidebar {
        background-color: #202030;
        color: #DADADA;
    }
    .stHeader {
        background-color: #202030;
        color: white;
    }
    .css-18e3th9 {
        background-color: #1E1E2E;
    }
    h1, h2, h3 {
        color: #6C63FF;
    }
    </style>""",
    unsafe_allow_html=True
)

# Set up label encoders (same as in model training)
smoking_encoder = LabelEncoder()
alcohol_encoder = LabelEncoder()
smoking_encoder.classes_ = np.array(['Non-Smoker', 'Smoker'])
alcohol_encoder.classes_ = np.array(['No', 'Yes'])

# Streamlit App
def main():
    st.title("Lung Cancer Risk Predictor")
    st.write("Predict your risk for lung cancer based on your lifestyle.")

    # User inputs
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
    smoking_history = st.selectbox("Smoking History", ["Non-Smoker", "Smoker"])
    packs_per_day = st.number_input("Packs Consumed Per Day (if smoker)", min_value=0.0, value=0.0, step=0.1)
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["No", "Yes"])

    # Preprocessing inputs
    smoking_encoded = smoking_encoder.transform([smoking_history])[0]
    alcohol_encoded = alcohol_encoder.transform([alcohol_consumption])[0]

    # Create input array
    input_data = pd.DataFrame({
        "Age": [age],
        "Smoking_History": [smoking_encoded],
        "Packs_consumed_day": [packs_per_day],
        "Alcohol_Consumption": [alcohol_encoded]
    })

    # Prediction
    if st.button("Predict Health Risk"):
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("High Risk for Lung Cancer")
        else:
            st.success("Low Risk for Lung Cancer")

        # Explanation (Optional)
        st.write("This prediction is based on your inputs: Age, Smoking History, Packs Consumed Per Day, and Alcohol Consumption.")

# Run the app
if __name__ == "__main__":
    main()
