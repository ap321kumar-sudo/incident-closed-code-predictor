
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import json # for handling JSON input

# --- Define paths for loading saved artifacts ---
# Assuming the model and encoders are in the same directory as app.py
model_load_path = 'best_decision_tree_model.joblib'
label_encoder_load_path = 'label_encoder.joblib'
one_hot_encoder_load_path = 'one_hot_encoder.joblib'

# --- Global lists for consistent preprocessing (derived from training data) ---
# These lists are hardcoded based on the preprocessing steps performed during training.
FINAL_NUMERICAL_COLS_BEFORE_OHE = ['reassignment_count', 'reopen_count', 'sys_mod_count', 'opened_at_year', 'opened_at_month', 'opened_at_day', 'opened_at_dayofweek', 'opened_at_hour', 'sys_updated_at_year', 'sys_updated_at_month', 'sys_updated_at_day', 'sys_updated_at_dayofweek', 'sys_updated_at_hour', 'resolved_at_year', 'resolved_at_month', 'resolved_at_day', 'resolved_at_dayofweek', 'resolved_at_hour', 'closed_at_year', 'closed_at_month', 'closed_at_day', 'closed_at_dayofweek', 'closed_at_hour']
FINAL_CATEGORICAL_COLS_BEFORE_OHE = ['incident_state', 'sys_updated_by', 'contact_type', 'location', 'category', 'subcategory', 'symptom', 'impact', 'urgency', 'priority', 'assignment_group', 'notify', 'resolved_by', 'isParent']
DATE_COLS_FOR_PREPROCESSING = ['opened_at', 'sys_created_at', 'sys_updated_at', 'resolved_at', 'closed_at']
COLS_TO_DROP_FOR_PREPROCESSING = ['caller_id', 'opened_by', 'assigned_to', 'problem_id', 'change', 'vendor', 'caused_by', 'cmdb_ci', 'sys_created_at', 'sys_created_by']

# --- Define all original columns for input schema consistency ---
# This list is critical to ensure the input_df always has the correct schema,
# even if some fields are not provided by the user in the Streamlit app.
original_df_cols = ['number', 'incident_state', 'active', 'reassignment_count', 'reopen_count',
                        'sys_mod_count', 'made_sla', 'caller_id', 'opened_by', 'opened_at',
                        'sys_created_by', 'sys_created_at', 'sys_updated_by', 'sys_updated_at',
                        'contact_type', 'location', 'category', 'subcategory', 'symptom',
                        'cmdb_ci', 'impact', 'urgency', 'priority', 'assignment_group',
                        'assigned_to', 'knowledge', 'u_priority_confirmation', 'notify',
                        'problem_id', 'change', 'vendor', 'caused_by', 'closed_code',
                        'resolved_by', 'resolved_at', 'closed_at', 'isParent']

# --- Load saved artifacts ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model_and_encoders():
    try:
        loaded_model = joblib.load(model_load_path)
        loaded_label_encoder = joblib.load(label_encoder_load_path)
        loaded_ohe = joblib.load(one_hot_encoder_load_path)
        return loaded_model, loaded_label_encoder, loaded_ohe
    except FileNotFoundError:
        st.error("Error: Model or encoder files not found. Please ensure they are in the same directory as app.py.")
        st.stop()

loaded_model, loaded_label_encoder, loaded_ohe = load_model_and_encoders()

# --- Prediction function ---
def predict_incident_closed_code(raw_incident_data: dict) -> str:
    """
    Takes raw incident data as a dictionary, preprocesses it, and predicts
    the closed code using the loaded model.
    """
    # 1. Convert raw input to DataFrame, ensuring all original columns are present
    input_df = pd.DataFrame([raw_incident_data])

    for col in original_df_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[original_df_cols]

    # 2. Replace '?' with NaN
    input_df.replace("?", np.nan, inplace=True)

    # 3. Convert Date Columns to Datetime Objects
    for col in DATE_COLS_FOR_PREPROCESSING:
        if col in input_df.columns:
            # Ensure it's string for format to work, then convert
            input_df[col] = input_df[col].astype(str)
            input_df[col] = pd.to_datetime(input_df[col], format='%d-%m-%Y %H:%M', errors='coerce')

    # 4. Drop specified columns (this includes 'closed_code' and 'number' for X)
    cols_to_drop_for_prediction_X = COLS_TO_DROP_FOR_PREPROCESSING + ['closed_code', 'number']
    existing_cols_to_drop_X = [col for col in cols_to_drop_for_prediction_X if col in input_df.columns]
    processed_X_temp = input_df.drop(columns=existing_cols_to_drop_X, errors='ignore')

    # 5. Impute remaining missing values with defaults for consistency
    for col in processed_X_temp.columns:
        if processed_X_temp[col].isnull().any():
            if processed_X_temp[col].dtype == 'object':
                processed_X_temp[col] = processed_X_temp[col].fillna('missing_category')
            elif pd.api.types.is_datetime64_any_dtype(processed_X_temp[col]):
                processed_X_temp[col] = processed_X_temp[col].fillna(pd.to_datetime('1970-01-01')) # Placeholder date
            elif pd.api.types.is_numeric_dtype(processed_X_temp[col]):
                processed_X_temp[col] = processed_X_temp[col].fillna(0)

    # 6. Feature Engineering: Extracting Time-Based Features
    date_feature_cols_in_X = ['opened_at', 'sys_updated_at', 'resolved_at', 'closed_at']
    for col in date_feature_cols_in_X:
        if col in processed_X_temp.columns and pd.api.types.is_datetime64_any_dtype(processed_X_temp[col]):
            processed_X_temp[f'{col}_year'] = processed_X_temp[col].dt.year
            processed_X_temp[f'{col}_month'] = processed_X_temp[col].dt.month
            processed_X_temp[f'{col}_day'] = processed_X_temp[col].dt.day
            processed_X_temp[f'{col}_dayofweek'] = processed_X_temp[col].dt.dayofweek
            processed_X_temp[f'{col}_hour'] = processed_X_temp[col].dt.hour

    # 7. Removing Original Datetime Columns (from processed_X_temp)
    processed_X = processed_X_temp.drop(columns=date_feature_cols_in_X, errors='ignore')

    # 8. Cleaning Inconsistent Categorical Data (lowercase, strip)
    for col in processed_X.select_dtypes(include='object').columns:
        processed_X[col] = processed_X[col].astype(str).str.lower().str.strip()

    # Ensure all expected numerical and categorical columns (before OHE) are present
    all_expected_features_before_ohe = FINAL_NUMERICAL_COLS_BEFORE_OHE + FINAL_CATEGORICAL_COLS_BEFORE_OHE

    for col in all_expected_features_before_ohe:
        if col not in processed_X.columns:
            if col in FINAL_NUMERICAL_COLS_BEFORE_OHE:
                processed_X[col] = 0
            else: # Categorical
                processed_X[col] = 'missing_category'

    # Reorder columns to match the training data's structure
    processed_X = processed_X[all_expected_features_before_ohe]

    processed_X_numerical = processed_X[FINAL_NUMERICAL_COLS_BEFORE_OHE]
    processed_X_categorical = processed_X[FINAL_CATEGORICAL_COLS_BEFORE_OHE]

    # 9. One-Hot Encoding and Stacking
    # Use the loaded OneHotEncoder to transform new data
    encoded_features_sparse = loaded_ohe.transform(processed_X_categorical)

    # Combine numerical features and sparse encoded categorical features
    final_features = hstack([processed_X_numerical.values, encoded_features_sparse]).tocsr()

    # 10. Make prediction
    prediction_encoded = loaded_model.predict(final_features)

    # 11. Inverse transform to get original label
    prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)

    return prediction_label[0]

# --- Streamlit UI ---
st.title("Incident Closed Code Predictor")
st.write("Predict the closed code based on incident details.")

st.header("Enter Incident Details")

# Simple user-friendly inputs
incident_state = st.selectbox(
    "Incident State",
    ["New", "Active", "Resolved", "Closed"]
)

category = st.selectbox(
    "Category",
    [
        "Software",
        "Hardware",
        "Network",
        "Email",
        "Access Issue"
    ]
)

impact = st.selectbox(
    "Impact",
    ["1 - High", "2 - Medium", "3 - Low"]
)

urgency = st.selectbox(
    "Urgency",
    ["1 - High", "2 - Medium", "3 - Low"]
)

priority = st.selectbox(
    "Priority",
    ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]
)

notify = st.selectbox(
    "Notify",
    ["Do Not Notify", "Send Notification"]
)
assignment_group = st.selectbox(
    "Assignment Group",
    [
        "Group 10",
        "Group 25",
        "Group 56",
        "Group 70"
    ]
)
# Predict button
  # Predict button
if st.button("Predict Closed Code"):

    category_mapping = {
        "Software": "Category 55",
        "Hardware": "Category 23",
        "Network": "Category 10",
        "Email": "Category 15",
        "Access Issue": "Category 40"
    }

    mapped_category = category_mapping[category]

    # Create input dictionary
    incident_input_dict = {
        "number": "INC_TEST_001",
        "incident_state": incident_state,
        "active": True,
        "reassignment_count": 0,
        "reopen_count": 0,
        "sys_mod_count": 0,
        "made_sla": True,
        "caller_id": "Caller 2403",
        "opened_by": "Opened by 8",
        "opened_at": "29-02-2016 01:16",
        "sys_created_by": "Created by 6",
        "sys_created_at": "29-02-2016 01:23",
        "sys_updated_by": "Updated by 21",
        "sys_updated_at": "29-02-2016 01:23",
        "contact_type": "Phone",
        "location": "Location 143",
        "category": mapped_category,
        "subcategory": "Subcategory 170",
        "symptom": "Symptom 72",
        "cmdb_ci": None,
        "impact": impact,
        "urgency": urgency,
        "priority": priority,
        "assignment_group": assignment_group,
        "assigned_to": None,
        "knowledge": True,
        "u_priority_confirmation": False,
        "notify": notify,
        "problem_id": None,
        "change": None,
        "vendor": None,
        "caused_by": None,
        "closed_code": "code 5",
        "resolved_by": "Resolved by 149",
        "resolved_at": "29-02-2016 11:29",
        "closed_at": "05-03-2016 12:00",
        "isParent": "No"
    }

    with st.spinner('Predicting...'):
        prediction = predict_incident_closed_code(incident_input_dict)

        closed_code_mapping = {
            "code 1": "Resolved Remotely",
            "code 2": "Duplicate Ticket",
            "code 3": "User Error",
            "code 4": "Network Issue Fixed",
            "code 5": "Hardware Fixed",
            "code 6": "Software Issue Resolved"
        }

        final_output = closed_code_mapping.get(prediction, prediction)

        st.success(f"Predicted Resolution: {final_output}")

st.markdown("---")
st.markdown("Developed as part of the ML Project")