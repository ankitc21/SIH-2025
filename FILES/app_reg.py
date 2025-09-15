import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Set a title and description for the app
st.set_page_config(page_title="Crop Production Prediction", layout="wide")
st.title("🌾 Crop Production Prediction")
st.markdown("Enter the details below to predict the crop production.")
st.markdown("---")

# Load the trained model and the label encoders
try:
    with open(r'D:\Crop\rf_reg_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(r'D:\Crop\label_encoders_regressor.json', 'r') as json_file:
        encoders = json.load(json_file)
except FileNotFoundError:
    st.error("Model or encoders file not found. Please make sure 'model.pkl' and 'label_encoders.json' are in the same directory.")
    st.stop()
    
# Get the unique values from the encoders to populate the dropdown menus
states = encoders['State_Name']['classes']
seasons = encoders['Season']['classes']
crops = encoders['Crop']['classes']

# Create the input fields
col1, col2 = st.columns(2)
with col1:
    state_name = st.selectbox('Select State', states)
    crop = st.selectbox('Select Crop', crops)

with col2:
    season = st.selectbox('Select Season', seasons)
    crop_year = st.number_input('Crop Year', min_value=1997, max_value=2015, value=2010, step=1)

area = st.number_input('Area (in hectares)', min_value=0.0, value=100.0)

# Create a button to trigger the prediction
if st.button('Predict Production'):
    # Prepare the input data for the model
    # The order of the features must match the training data: ['State_Name', 'Crop_Year', 'Season', 'Crop', 'Area']
    
    # Get the integer-encoded value for each categorical input
    state_encoded = encoders['State_Name']['mapping'][state_name]
    season_encoded = encoders['Season']['mapping'][season]
    crop_encoded = encoders['Crop']['mapping'][crop]
    
    # Create a DataFrame with the same column names as the training data
    input_data = pd.DataFrame([[state_encoded, crop_year, season_encoded, crop_encoded, area]],
                              columns=['State_Name', 'Crop_Year', 'Season', 'Crop', 'Area'])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result
    st.markdown("### Prediction Result")
    st.info(f"The predicted production is **{prediction:,.2f}** tonnes")
    st.warning("Disclaimer: This prediction is based on the provided data and model. Factors not included in the model (e.g., weather, pests) can affect actual production.")
    
st.markdown("---")
st.markdown("This app uses a Random Forest Regressor model to predict crop production based on state, crop year, season, crop type, and area.")
st.markdown("Model metrics: R-squared: 0.9224, Mean Absolute Error: 145903.3903")