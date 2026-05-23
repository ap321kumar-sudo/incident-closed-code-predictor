import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack


MODEL_PATH = "best_decision_tree_model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
ONE_HOT_ENCODER_PATH = "one_hot_encoder.joblib"

FINAL_NUMERICAL_COLS_BEFORE_OHE = [
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
    "opened_at_year",
    "opened_at_month",
    "opened_at_day",
    "opened_at_dayofweek",
    "opened_at_hour",
    "sys_updated_at_year",
    "sys_updated_at_month",
    "sys_updated_at_day",
    "sys_updated_at_dayofweek",
    "sys_updated_at_hour",
    "resolved_at_year",
    "resolved_at_month",
    "resolved_at_day",
    "resolved_at_dayofweek",
    "resolved_at_hour",
    "closed_at_year",
    "closed_at_month",
    "closed_at_day",
    "closed_at_dayofweek",
    "closed_at_hour",
]

FINAL_CATEGORICAL_COLS_BEFORE_OHE = [
    "incident_state",
    "sys_updated_by",
    "contact_type",
    "location",
    "category",
    "subcategory",
    "symptom",
    "impact",
    "urgency",
    "priority",
    "assignment_group",
    "notify",
    "resolved_by",
    "isParent",
]

DATE_COLS_FOR_PREPROCESSING = [
    "opened_at",
    "sys_created_at",
    "sys_updated_at",
    "resolved_at",
    "closed_at",
]

COLS_TO_DROP_FOR_PREPROCESSING = [
    "caller_id",
    "opened_by",
    "assigned_to",
    "problem_id",
    "change",
    "vendor",
    "caused_by",
    "cmdb_ci",
    "sys_created_at",
    "sys_created_by",
    "number",
    "closed_code",
]


@st.cache_resource
def load_model_and_encoders():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    one_hot_encoder = joblib.load(ONE_HOT_ENCODER_PATH)
    return model, label_encoder, one_hot_encoder


def format_datetime(date_value, time_value):
    return (
        f"{date_value.day:02d}-{date_value.month:02d}-{date_value.year} "
        f"{time_value.hour:02d}:{time_value.minute:02d}"
    )


def predict_incident_closed_code(raw_incident_data, model, label_encoder, one_hot_encoder):
    input_df = pd.DataFrame([raw_incident_data])
    input_df.replace("?", np.nan, inplace=True)

    for col in DATE_COLS_FOR_PREPROCESSING:
        if col in input_df.columns:
            input_df[col] = pd.to_datetime(
                input_df[col],
                format="%d-%m-%Y %H:%M",
                errors="coerce",
            )

    processed_x = input_df.drop(columns=COLS_TO_DROP_FOR_PREPROCESSING, errors="ignore")

    for col in processed_x.columns:
        if processed_x[col].isnull().any():
            if pd.api.types.is_numeric_dtype(processed_x[col]):
                processed_x[col] = processed_x[col].fillna(0)
            else:
                processed_x[col] = processed_x[col].fillna("unknown")

    date_feature_cols = [
        col
        for col in ["opened_at", "sys_updated_at", "resolved_at", "closed_at"]
        if col in processed_x.columns
    ]

    for col in date_feature_cols:
        if pd.api.types.is_datetime64_any_dtype(processed_x[col]):
            processed_x[f"{col}_year"] = processed_x[col].dt.year
            processed_x[f"{col}_month"] = processed_x[col].dt.month
            processed_x[f"{col}_day"] = processed_x[col].dt.day
            processed_x[f"{col}_dayofweek"] = processed_x[col].dt.dayofweek
            processed_x[f"{col}_hour"] = processed_x[col].dt.hour

    processed_x = processed_x.drop(columns=date_feature_cols, errors="ignore")

    for col in processed_x.select_dtypes(include=["object", "string"]).columns:
        processed_x[col] = processed_x[col].astype(str).str.lower().str.strip()

    for col in FINAL_NUMERICAL_COLS_BEFORE_OHE:
        if col not in processed_x.columns:
            processed_x[col] = 0

    for col in FINAL_CATEGORICAL_COLS_BEFORE_OHE:
        if col not in processed_x.columns:
            processed_x[col] = "unknown"

    numerical_features = processed_x[FINAL_NUMERICAL_COLS_BEFORE_OHE]
    categorical_features = processed_x[FINAL_CATEGORICAL_COLS_BEFORE_OHE]
    encoded_features = one_hot_encoder.transform(categorical_features)
    final_features = hstack([numerical_features.values, encoded_features]).tocsr()

    prediction_encoded = model.predict(final_features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    return str(prediction_label[0])


st.set_page_config(page_title="Incident Closed Code Prediction", layout="wide")
st.title("Incident Closed Code Prediction")
st.markdown("Enter incident details to predict the closed code.")

try:
    loaded_model, loaded_label_encoder, loaded_ohe = load_model_and_encoders()
except Exception as exc:
    st.error(f"Model files could not be loaded: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Core Incident Details")
    incident_state_input = st.selectbox(
        "Incident State*",
        ["new", "active", "resolved", "closed", "in progress", "on hold", "pending", "canceled"],
    )
    contact_type_input = st.selectbox(
        "Contact Type*",
        ["phone", "email", "self-service", "chat", "direct call", "web"],
    )
    reassignment_count_input = st.number_input("Reassignment Count*", min_value=0, value=0)
    reopen_count_input = st.number_input("Reopen Count*", min_value=0, value=0)
    sys_mod_count_input = st.number_input("System Modification Count*", min_value=0, value=0)

with col2:
    st.header("Categorization & Priority")
    category_input = st.text_input("Category*", value="category 55")
    subcategory_input = st.text_input("Subcategory*", value="subcategory default")
    symptom_input = st.text_area("Symptom Description", value="user reports slow system performance")
    impact_input = st.selectbox("Impact*", ["1 - high", "2 - medium", "3 - low"])
    urgency_input = st.selectbox("Urgency*", ["1 - high", "2 - medium", "3 - low"])
    priority_input = st.selectbox(
        "Priority*",
        ["1 - critical", "2 - high", "3 - moderate", "4 - low"],
    )

with col3:
    st.header("Time & Other Details")
    opened_at_date = st.date_input("Opened Date*", pd.Timestamp.today())
    opened_at_time = st.time_input("Opened Time*", pd.Timestamp("09:00").time())
    opened_at_input = format_datetime(opened_at_date, opened_at_time)

    location_input = st.text_input("Location", value="location default")
    assignment_group_input = st.text_input("Assignment Group", value="group default")
    notify_input = st.selectbox("Notify", ["do not notify", "notify caller", "notify group"])
    is_parent_input = st.selectbox("Is Parent Incident?", ["no", "yes"])

default_values = {
    "number": "INC_STREAMLIT_TEST",
    "active": True,
    "made_sla": True,
    "caller_id": "caller default",
    "opened_by": "opened by default",
    "sys_created_by": "created by default",
    "sys_created_at": opened_at_input,
    "sys_updated_by": "updated by default",
    "sys_updated_at": opened_at_input,
    "resolved_by": "resolved by default",
    "resolved_at": opened_at_input,
    "closed_at": opened_at_input,
    "problem_id": np.nan,
    "change": np.nan,
    "vendor": np.nan,
    "caused_by": np.nan,
    "cmdb_ci": np.nan,
    "closed_code": np.nan,
    "assigned_to": np.nan,
    "knowledge": True,
    "u_priority_confirmation": False,
}

raw_incident_data = {
    **default_values,
    "incident_state": incident_state_input,
    "reassignment_count": reassignment_count_input,
    "reopen_count": reopen_count_input,
    "sys_mod_count": sys_mod_count_input,
    "opened_at": opened_at_input,
    "contact_type": contact_type_input,
    "category": category_input,
    "subcategory": subcategory_input,
    "symptom": symptom_input,
    "impact": impact_input,
    "urgency": urgency_input,
    "priority": priority_input,
    "location": location_input,
    "assignment_group": assignment_group_input,
    "notify": notify_input,
    "isParent": is_parent_input,
}

st.markdown("---")

if st.button("Predict Closed Code", type="primary"):
    with st.spinner("Predicting..."):
        try:
            predicted_code = predict_incident_closed_code(
                raw_incident_data,
                loaded_model,
                loaded_label_encoder,
                loaded_ohe,
            )
            st.success(f"Predicted Incident Closed Code: `{predicted_code}`")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
