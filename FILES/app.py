import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Create enhanced dataset (Functions) ---
@st.cache_data
def create_dataset():
    n_samples = 1000
    
    states = ['Andaman and Nicobar Islands', 'West Bengal', 'Uttar Pradesh', 'Maharashtra', 'Karnataka',
              'Tamil Nadu', 'Rajasthan', 'Punjab', 'Haryana', 'Gujarat']
    districts = ['NICOBARS', 'PURULIA', 'LUCKNOW', 'PUNE', 'BANGALORE', 'CHENNAI', 'JAIPUR', 'AMRITSAR', 'ROHTAK', 'AHMEDABAD']
    crops = ['Arecanut', 'Rice', 'Banana', 'Cashewnut', 'Sugarcane', 'Wheat', 'Cotton', 'Groundnut', 'Sesame']
    seasons = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter']
    
    data = {
        'State_Name': np.random.choice(states, n_samples),
        'District_Name': np.random.choice(districts, n_samples),
        'Crop_Year': np.random.randint(2000, 2020, n_samples),
        'Season': np.random.choice(seasons, n_samples),
        'Crop': np.random.choice(crops, n_samples),
        'Area': np.random.exponential(scale=500, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create strong predictable relationships
    state_multipliers = {state: np.random.uniform(2.5, 4.5) for state in states}
    crop_multipliers = {crop: np.random.uniform(2.0, 3.5) for crop in crops}
    season_multipliers = {season: np.random.uniform(0.8, 1.8) for season in seasons}
    
    df['state_mult'] = df['State_Name'].map(state_multipliers)
    df['crop_mult'] = df['Crop'].map(crop_multipliers)
    df['season_mult'] = df['Season'].map(season_multipliers)
    
    df['Production'] = (
        df['Area'] * df['state_mult'] * df['crop_mult'] * df['season_mult'] * (1 + (df['Crop_Year'] - 2000) * 0.04) +
        np.random.normal(0, 80, n_samples)
    )
    
    df['Production'] = np.maximum(df['Production'], 0)
    df = df.drop(['state_mult', 'crop_mult', 'season_mult'], axis=1)
    
    return df

@st.cache_data
def train_model():
    df = create_dataset()
    
    # Label encoding
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        df[col + '_encoded'] = LabelEncoder().fit_transform(df[col])

    # Features and target
    feature_columns = ['State_Name_encoded', 'District_Name_encoded', 'Crop_Year',
                       'Season_encoded', 'Crop_encoded', 'Area']
    X = df[feature_columns]
    y = df['Production']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.25],
        'reg_alpha': [0, 0.01, 0.1, 0.2, 0.5, 1],
        'reg_lambda': [0.5, 0.8, 1, 1.2, 1.5, 2]
    }

    xgb_regressor = xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist'
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        verbose=0,
        random_state=42
    )

    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(X_test_scaled)
    
    results = {
        "best_params": random_search.best_params_,
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "feature_importances": best_model.feature_importances_,
        "feature_names": feature_columns,
        "scaler": scaler,
        "label_encoders": {
            'State_Name': LabelEncoder().fit(df['State_Name']),
            'District_Name': LabelEncoder().fit(df['District_Name']),
            'Season': LabelEncoder().fit(df['Season']),
            'Crop': LabelEncoder().fit(df['Crop'])
        },
        "df": df
    }
    
    return best_model, results

# --- Main Streamlit App ---

st.set_page_config(
    page_title="Crop Production Prediction App",
    page_icon="🌱",
    layout="wide"
)

st.title("🌾 Crop Production Prediction using XGBoost")
st.markdown("This app predicts crop production based on various factors and showcases a machine learning model's performance.")
st.markdown("---")

# Caching the model training so it only runs once
best_model, results = train_model()

# --- Displaying Results ---
st.header("Model Evaluation Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Test R² Score", value=f"{results['r2_score']:.4f}")
with col2:
    st.metric(label="Test RMSE", value=f"{results['rmse']:.2f}")
with col3:
    st.metric(label="Test MAE", value=f"{results['mae']:.2f}")

st.markdown("---")

# --- Feature Importance ---
st.header("Feature Importance")
importance_df = pd.DataFrame({
    'Feature': results['feature_names'],
    'Importance': results['feature_importances']
}).sort_values('Importance', ascending=False)
st.dataframe(importance_df, use_container_width=True)

st.markdown("---")

# --- Interactive Prediction ---
st.header("Predict Crop Production")
st.markdown("Adjust the input values below to get a production prediction.")

# Get unique values for dropdowns
unique_states = results['df']['State_Name'].unique()
unique_districts = results['df']['District_Name'].unique()
unique_seasons = results['df']['Season'].unique()
unique_crops = results['df']['Crop'].unique()
unique_years = results['df']['Crop_Year'].unique()

with st.sidebar:
    st.header("Input Features")
    state = st.selectbox("State", unique_states)
    district = st.selectbox("District", unique_districts)
    crop = st.selectbox("Crop", unique_crops)
    season = st.selectbox("Season", unique_seasons)
    year = st.selectbox("Crop Year", unique_years)
    area = st.slider("Area (in hectares)", 
                     min_value=float(results['df']['Area'].min()), 
                     max_value=float(results['df']['Area'].max()), 
                     value=500.0)

if st.button("Predict Production"):
    try:
        # Get encoded values from encoders stored in results
        state_encoded = results['label_encoders']['State_Name'].transform([state])[0]
        district_encoded = results['label_encoders']['District_Name'].transform([district])[0]
        season_encoded = results['label_encoders']['Season'].transform([season])[0]
        crop_encoded = results['label_encoders']['Crop'].transform([crop])[0]
        
        # Create a dataframe for the input
        input_data = pd.DataFrame([[state_encoded, district_encoded, year, season_encoded, crop_encoded, area]],
                                  columns=results['feature_names'])

        # Scale the input data using the trained scaler
        input_scaled = results['scaler'].transform(input_data)
        
        # Make prediction
        prediction = best_model.predict(input_scaled)[0]

        st.success(f"**Predicted Production:** {prediction:.2f} tons")

    except ValueError as e:
        st.error(f"Error making prediction: {e}")

st.markdown("---")
st.info("The model is trained on a synthetic dataset for demonstration purposes. The predictions are not based on real-world data.")