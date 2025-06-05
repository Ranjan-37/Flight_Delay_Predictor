import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
le_airline = joblib.load("le_airline.pkl")
le_airport_from = joblib.load("le_airport_from.pkl")
le_airport_to = joblib.load("le_airport_to.pkl")

st.set_page_config(page_title="Ranjan's Flight Delay Predictor", page_icon="🛫")
st.title("🛫 Flight Delay Prediction App")
st.write("Fill in the flight details to predict whether your flight will be delayed.")

# Input fields
airline_input = st.selectbox("Select Airline", le_airline.classes_)
flight_input = st.number_input("Enter Flight Number", min_value=1, value=100)
airport_from_input = st.selectbox("Departure Airport", le_airport_from.classes_)
airport_to_input = st.selectbox("Arrival Airport", le_airport_to.classes_)
day_input = st.selectbox("Day of the Week (1 = Mon, 7 = Sun)", list(range(1, 8)))
time_input = st.number_input("Scheduled Departure Time (HHMM)", min_value=0, max_value=2359, value=900)
length_input = st.number_input("Flight Duration (in minutes)", min_value=0, value=60)

# Predict button
if st.button("Predict Delay"):
    # Encode categorical features
    airline = le_airline.transform([airline_input])[0]
    airport_from = le_airport_from.transform([airport_from_input])[0]
    airport_to = le_airport_to.transform([airport_to_input])[0]

    # Create input array
    input_data = np.array([[airline, flight_input, airport_from, airport_to, day_input, time_input, length_input]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"✈️ Flight is likely to be **delayed**. Probability: {pred_proba:.2f}")
    else:
        st.success(f"✅ Flight is likely to be **on time**. Probability: {1 - pred_proba:.2f}")
