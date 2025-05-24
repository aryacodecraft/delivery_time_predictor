import streamlit as st
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
labels = pickle.load(open("label_encoders.pkl", "rb"))

# App title
st.title(" Food Delivery Time Predictor")

# User inputs
distance = st.number_input("Distance (in km)", min_value=0.0)

# Hardcoded dropdown options (must match training data)
weather = st.selectbox("Weather", ["Clear", "Rainy", "Windy", "Foggy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])

prep_time = st.number_input("Preparation Time (minutes)", min_value=0)
experience = st.number_input("Courier Experience (years)", min_value=0.0)

# Predict button
if st.button("Predict Delivery Time"):
    try:
        # Convert dropdowns using label encoders
        weather_encoded = labels['Weather'].transform([weather])[0]
        traffic_encoded = labels['Traffic_Level'].transform([traffic])[0]
        time_encoded = labels['Time_of_Day'].transform([time_of_day])[0]
        vehicle_encoded = labels['Vehicle_Type'].transform([vehicle])[0]

        # Form input array
        input_data = np.array([[
            distance,
            weather_encoded,
            traffic_encoded,
            time_encoded,
            vehicle_encoded,
            prep_time,
            experience
        ]])

        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"📦 Predicted Delivery Time: {prediction:.2f} minutes")


    except Exception as e:
        st.error("Error: " + str(e))

 