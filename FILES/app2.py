import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Crop Production Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the trained model, scaler, and label encoders
try:
    with open(r'D:\Crop\xgb_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(r'D:\Crop\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(r'D:\Crop\label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'xgb_best_model.pkl', 'scaler.pkl', and 'label_encoders.pkl' are in the same directory.")
    st.stop()

# Application title and description
st.title("🌾 Crop Production Prediction")
st.write("""
This app predicts crop production based on various agricultural parameters. 
Adjust the input values on the sidebar to get a prediction.
""")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    """Gathers user input from the sidebar widgets."""
    state_name = st.sidebar.selectbox("State Name", options=list(encoders['State_Name'].classes_))
    district_name = st.sidebar.selectbox("District Name", options=list(encoders['District_Name'].classes_))
    crop_year = st.sidebar.slider("Crop Year", 2000, 2014, 2012)
    season = st.sidebar.selectbox("Season", options=list(encoders['Season'].classes_))
    crop = st.sidebar.selectbox("Crop", options=list(encoders['Crop'].classes_))
    area = st.sidebar.number_input("Area (in hectares)", min_value=0.0, value=600.0)

    data = {
        'State_Name_encoded': encoders['State_Name'].transform([state_name])[0],
        'District_Name_encoded': encoders['District_Name'].transform([district_name])[0],
        'Crop_Year': crop_year,
        'Season_encoded': encoders['Season'].transform([season])[0],
        'Crop_encoded': encoders['Crop'].transform([crop])[0],
        'Area': area
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Display user input
input_df = user_input_features()
st.subheader("User Input Parameters")
st.dataframe(input_df)

# Prediction section
st.subheader("Prediction")
if st.sidebar.button("Predict Production"):
    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction using the loaded model
    prediction = model.predict(input_scaled)
    
    # Display the prediction
    st.success(f"Predicted Production: {prediction[0]:,.2f} tonnes")
    st.info("Note: The prediction is based on a simulated dataset used for demonstration.")

# Optional: Display model performance metrics
st.sidebar.markdown("---")
st.sidebar.header("Model Performance")
st.sidebar.write("The model achieved the following scores on a test set:")
st.sidebar.write("- **R² Score:** 0.9899")
st.sidebar.write("- **Root Mean Squared Error (RMSE):** 1181.24")
st.sidebar.write("- **Mean Absolute Error (MAE):** 492.29")

# Optional: Display feature importance
st.sidebar.markdown("---")
st.sidebar.header("Feature Importance")
st.sidebar.write("The most important features for the model are:")
st.sidebar.write("""
- **Area**: 0.7420
- **Season_encoded**: 0.0808
- **Crop_Year**: 0.0634
- **Crop_encoded**: 0.0512
- **State_Name_encoded**: 0.0500
- **District_Name_encoded**: 0.0125
""")

# A small visual touch at the end
st.markdown("---")
