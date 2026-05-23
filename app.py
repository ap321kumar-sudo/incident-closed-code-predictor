import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# --- Define paths for loading saved artifacts --- 
# Assuming model and encoders are in the same directory as app.py for Streamlit Cloud deployment
BASE_DIR = Path(__file__).resolve().parent
model_load_path = BASE_DIR / 'best_decision_tree_model.joblib'
label_encoder_load_path = BASE_DIR / 'label_encoder.joblib'
one_hot_encoder_load_path = BASE_DIR / 'one_hot_encoder.joblib'

# --- Global lists for consistent preprocessing (derived from training data) ---
# These are hardcoded based on the analysis performed in the notebook
FINAL_NUMERICAL_COLS_BEFORE_OHE = ['reassignment_count', 'reopen_count', 'sys_mod_count', 'opened_at_year', 'opened_at_month', 'opened_at_day', 'opened_at_dayofweek', 'opened_at_hour', 'sys_updated_at_year', 'sys_updated_at_month', 'sys_updated_at_day', 'sys_updated_at_dayofweek', 'sys_updated_at_hour', 'resolved_at_year', 'resolved_at_month', 'resolved_at_day', 'resolved_at_dayofweek', 'resolved_at_hour', 'closed_at_year', 'closed_at_month', 'closed_at_day', 'closed_at_dayofweek', 'closed_at_hour']
FINAL_CATEGORICAL_COLS_BEFORE_OHE = ['incident_state', 'sys_updated_by', 'contact_type', 'location', 'category', 'subcategory', 'symptom', 'impact', 'urgency', 'priority', 'assignment_group', 'notify', 'resolved_by', 'isParent']
DATE_COLS_FOR_PREPROCESSING = ['opened_at', 'sys_created_at', 'sys_updated_at', 'resolved_at', 'closed_at']
COLS_TO_DROP_FOR_PREPROCESSING = ['caller_id', 'opened_by', 'assigned_to', 'problem_id', 'change', 'vendor', 'caused_by', 'cmdb_ci', 'sys_created_at', 'sys_created_by', 'number', 'closed_code'] # Added 'number' and 'closed_code' as these are not features

# --- Load saved artifacts ---
@st.cache_resource
def load_model_and_encoders():
    loaded_model = joblib.load(model_load_path)
    loaded_label_encoder = joblib.load(label_encoder_load_path)
    loaded_ohe = joblib.load(one_hot_encoder_load_path)
    return loaded_model, loaded_label_encoder, loaded_ohe

loaded_model, loaded_label_encoder, loaded_ohe = load_model_and_encoders()

# --- Prediction function ---
def predict_incident_closed_code(raw_incident_data: dict) -> str:
    """
    Takes raw incident data as a dictionary, preprocesses it, and predicts
    the closed code using the loaded model.
    """
    # 1. Convert raw input to DataFrame
    input_df = pd.DataFrame([raw_incident_data])

    # 2. Replace '?' with NaN
    input_df.replace("?", np.nan, inplace=True)

    # 3. Convert Date Columns to Datetime Objects
    for col in DATE_COLS_FOR_PREPROCESSING:
        if col in input_df.columns:
            # Ensure the format matches the training data format
            input_df[col] = pd.to_datetime(input_df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 4. Drop specified columns (ensure 'closed_code' and 'number' are also dropped if present)
    existing_cols_to_drop = [col for col in COLS_TO_DROP_FOR_PREPROCESSING if col in input_df.columns]
    processed_X = input_df.drop(columns=existing_cols_to_drop, errors='ignore')

    # 5. Impute remaining missing values with mode (for object/datetime) or median (for numeric)
    for col in processed_X.columns:
        if processed_X[col].isnull().any():
            if processed_X[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(processed_X[col]):
                # For new data, if a value is missing, use a placeholder 'unknown' or a consistent mode from training.
                # For this deployment, we'll use a simple fillna with a placeholder for new unseen data.
                processed_X[col] = processed_X[col].fillna('unknown')
            elif pd.api.types.is_numeric_dtype(processed_X[col]):
                processed_X[col] = processed_X[col].fillna(0) # Default to 0 for numerical

    # 6. Feature Engineering: Extracting Time-Based Features
    date_feature_cols_in_X = [col for col in ['opened_at', 'sys_updated_at', 'resolved_at', 'closed_at'] if col in processed_X.columns]
    for col in date_feature_cols_in_X:
        if pd.api.types.is_datetime64_any_dtype(processed_X[col]):
            processed_X[f'{col}_year'] = processed_X[col].dt.year
            processed_X[f'{col}_month'] = processed_X[col].dt.month
            processed_X[f'{col}_day'] = processed_X[col].dt.day
            processed_X[f'{col}_dayofweek'] = processed_X[col].dt.dayofweek
            processed_X[f'{col}_hour'] = processed_X[col].dt.hour

    # 7. Removing Original Datetime Columns (from processed_X)
    processed_X = processed_X.drop(columns=date_feature_cols_in_X, errors='ignore')

    # 8. Cleaning Inconsistent Categorical Data (lowercase, strip)
    for col in processed_X.select_dtypes(include='object').columns:
        processed_X[col] = processed_X[col].astype(str).str.lower().str.strip()

    # Ensure all numerical columns are present and in the correct order
    for col in FINAL_NUMERICAL_COLS_BEFORE_OHE:
        if col not in processed_X.columns:
            processed_X[col] = 0 # Default numerical value for missing expected numerical features
    processed_X_numerical = processed_X[FINAL_NUMERICAL_COLS_BEFORE_OHE]

    # Ensure all categorical columns are present for OHE and in the correct order
    for col in FINAL_CATEGORICAL_COLS_BEFORE_OHE:
        if col not in processed_X.columns:
            processed_X[col] = 'unknown' # Default categorical value for missing expected categorical features
    processed_X_categorical = processed_X[FINAL_CATEGORICAL_COLS_BEFORE_OHE]

    # 9. One-Hot Encoding and Stacking
    encoded_features_sparse = loaded_ohe.transform(processed_X_categorical)

    # Combine numerical features and sparse encoded categorical features
    final_features = hstack([processed_X_numerical.values, encoded_features_sparse]).tocsr()

    # 10. Make prediction
    prediction_encoded = loaded_model.predict(final_features)

    # 11. Inverse transform to get original label
    prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)

    return prediction_label[0]

# --- Streamlit UI --- 
st.title('Incident Closed Code Prediction')
st.write('Enter incident details to predict the closed code.')

# Input fields for key features

# Using a selectbox for incident_state, assuming common states
incident_state_options = ['new', 'active', 'resolved', 'closed', 'awaiting user info', 'awaiting vendor', 'awaiting problem', 'in progress', 'on hold', 'pending', 'canceled', 'rejected']
incident_state_input = st.selectbox('Incident State', options=incident_state_options)

# Numerical inputs
reassignment_count_input = st.number_input('Reassignment Count', min_value=0, value=0)
sys_mod_count_input = st.number_input('System Modification Count', min_value=0, value=0)

# Date/Time input for 'opened_at' - this will derive several time features
opened_at_date = st.date_input('Opened Date', pd.to_datetime('today'))
opened_at_time = st.time_input('Opened Time', pd.to_datetime('09:00').time())
opened_at_input = f"{opened_at_date} {opened_at_time}"

# Categorical inputs (example subset)
contact_type_options = ['phone', 'email', 'self-service', 'chat']
contact_type_input = st.selectbox('Contact Type', options=contact_type_options)

category_options = ['category 55', 'category 22', 'category 4', 'software', 'network', 'hardware'] # Example options
category_input = st.selectbox('Category', options=category_options)

impact_options = ['1 - high', '2 - medium', '3 - low']
impact_input = st.selectbox('Impact', options=impact_options)

urgency_options = ['1 - high', '2 - medium', '3 - low']
urgency_input = st.selectbox('Urgency', options=urgency_options)

priority_options = ['1 - critical', '2 - high', '3 - moderate', '4 - low']
priority_input = st.selectbox('Priority', options=priority_options)

# Assume default values for other less critical features for the UI
# These values should ideally come from common modes/medians of the training data
# or be made configurable if they are important for specific use cases.
default_values = {
    'number': 'INC_STREAMLIT_TEST', # Placeholder, will be dropped anyway
    'active': True,
    'made_sla': True,
    'caller_id': 'caller default',
    'opened_by': 'opened by default',
    'sys_created_by': 'created by default',
    'sys_created_at': pd.to_datetime(opened_at_input).strftime('%Y-%m-%d %H:%M:%S'), # Use opened_at for created_at default
    'sys_updated_by': 'updated by default',
    'sys_updated_at': pd.to_datetime(opened_at_input).strftime('%Y-%m-%d %H:%M:%S'), # Use opened_at for updated_at default
    'location': 'location default',
    'subcategory': 'subcategory default',
    'symptom': 'symptom default',
    'cmdb_ci': np.nan, # Will be dropped
    'assignment_group': 'group default',
    'assigned_to': np.nan, # Will be dropped
    'knowledge': True,
    'u_priority_confirmation': False,
    'notify': 'do not notify',
    'problem_id': np.nan, # Will be dropped
    'change': np.nan, # Will be dropped
    'vendor': np.nan, # Will be dropped
    'caused_by': np.nan, # Will be dropped
    'closed_code': np.nan, # Target, not input
    'resolved_by': 'resolved by default',
    'resolved_at': pd.to_datetime(opened_at_input).strftime('%Y-%m-%d %H:%M:%S'), # Use opened_at for resolved_at default
    'closed_at': pd.to_datetime(opened_at_input).strftime('%Y-%m-%d %H:%M:%S'), # Use opened_at for closed_at default
    'isParent': 'no'
}

# Populate raw_incident_data with user inputs and defaults
raw_incident_data = {
    **default_values,
    'incident_state': incident_state_input,
    'reassignment_count': reassignment_count_input,
    'sys_mod_count': sys_mod_count_input,
    'opened_at': opened_at_input, # This is the full datetime string
    'contact_type': contact_type_input,
    'category': category_input,
    'impact': impact_input,
    'urgency': urgency_input,
    'priority': priority_input,
}

# When the user clicks the predict button
if st.button('Predict Closed Code'):
    try:
        predicted_code = predict_incident_closed_code(raw_incident_data)
        st.success(f'The Predicted Incident Closed Code is: {predicted_code}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
        st.write('Please check the input values and ensure all necessary files are deployed correctly.')

