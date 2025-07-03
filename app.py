import joblib
import streamlit as st
import pandas as pd
import gdown
import os

# File name to save locally
model_path = 'pollution_model.pkl'

# Download the model from Google Drive if not already present
if not os.path.exists(model_path):
    file_id = '12GH455E2HXYiFVcph6l_upto35eMa1yv'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# Load model and columns
model = joblib.load(model_path)
model_cols = joblib.load('model_columns.pkl')

# Custom styling (optional)
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 32px;
            color: #0066cc;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💧 Water Pollutant Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict water quality indicators by station and year</div>", unsafe_allow_html=True)
st.markdown("---")

# Input UI
year_input = st.number_input("📅 Select Year", min_value=2000, max_value=2100, value=2025)

# Optional: Create dropdown for known station IDs (replace with actual list if needed)
station_list = [str(i) for i in range(1, 23)]  # '1' to '22'
station_id = st.selectbox("🏭 Select Station ID", options=station_list)

# Predict button
if st.button("🔍 Predict Pollutants"):
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # Add missing columns
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_cols]

    # Predict
    predicted_pollutants = model.predict(input_encoded)[0]
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

    st.markdown(f"### 📊 Predicted Pollutant Levels for Station **{station_id}** in **{year_input}**")
    st.markdown("---")

    # Display pollutants in 2-column layout
    col1, col2 = st.columns(2)
    for i, (p, val) in enumerate(zip(pollutants, predicted_pollutants)):
        display = f"{val:.2f}"
        if i % 2 == 0:
            col1.metric(label=p, value=display)
        else:
            col2.metric(label=p, value=display)
