# app_streamlit.py - robust replacement (copy & overwrite)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from data_preprocessing import clean_and_engineer

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")
st.title("✈️ Flight Fare Prediction")

# --- Session state defaults ---
defaults = {
    "duration": 180,
    "day": 1,
    "month": 1,
    "airline": "IndiGo",
    "source": "Delhi",
    "destination": "Cochin",
    "stops": "non-stop"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Load model ---
MODEL_FILENAME = "model.joblib"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        return None, None
    obj = joblib.load(MODEL_FILENAME)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("columns", None)
    else:
        return obj, None

model, model_columns = load_model()

# --- Sidebar ---
st.sidebar.header("Flight Details")
st.sidebar.number_input("Duration (minutes)", min_value=0, max_value=2000, key="duration")
st.sidebar.slider("Journey Day", 1, 31, key="day")
st.sidebar.slider("Journey Month", 1, 12, key="month")
st.sidebar.selectbox("Airline",
    ["IndiGo","Air India","Jet Airways","SpiceJet","Vistara","Multiple carriers","Unknown"], key="airline")
st.sidebar.selectbox("Source",
    ["Delhi","Kolkata","Mumbai","Chennai","Bengaluru","Other"], key="source")
st.sidebar.selectbox("Destination",
    ["Cochin","Banglore","New Delhi","Hyderabad","Kolkata","Other"], key="destination")
st.sidebar.selectbox("Total Stops",
    ["non-stop","1 stop","2 stops","3 stops","Unknown"], key="stops")

def make_user_df():
    mins = int(st.session_state.duration)
    hrs = mins // 60
    rem = mins % 60
    duration_str = f"{hrs}h {rem}m" if hrs>0 else f"{rem}m"
    d = {
        "Date_of_Journey": None,
        "Journey_Day": st.session_state.day,
        "Journey_Month": st.session_state.month,
        "Duration": duration_str,
        "Airline": st.session_state.airline,
        "Source": st.session_state.source,
        "Destination": st.session_state.destination,
        "Total_Stops": st.session_state.stops
    }
    return pd.DataFrame([d])

def prepare_X_for_model(df_user, model_columns):
    df_user = clean_and_engineer(df_user)
    df_user = pd.get_dummies(df_user)
    if model_columns:
        for c in model_columns:
            if c not in df_user.columns:
                df_user[c] = 0
        X = df_user[model_columns]
    else:
        X = df_user.select_dtypes(include=[np.number]).fillna(0)
    return X

if st.button("Predict Fare"):
    if model is None:
        st.error("Model file not found. Please run training (python train_model.py) in the project folder.")
    else:
        user_df = make_user_df()
        X = prepare_X_for_model(user_df, model_columns)
        try:
            pred = model.predict(X)[0]
            st.success(f"Estimated Fare: ₹{pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with st.expander("Debug info"):
    st.write("model loaded:", None if model is None else type(model))
    st.write("model columns (if present):", model_columns)
    st.write("session state:", dict(st.session_state))
