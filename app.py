# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained model and encoders
model = joblib.load('car_price_predictor.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit title
st.title("Car Price Prediction")

# User input fields
vehicle_condition = st.selectbox("Vehicle Condition", ["USED", "NEW"])
standard_make = st.text_input("Car Make")
standard_colour = st.text_input("Car Colour")
standard_model = st.text_input("Car Model")
body_type = st.text_input("Body Type")
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
mileage = st.number_input("Mileage", min_value=0)
year_of_registration = st.number_input("Year of Registration", min_value=1900, max_value=2024)

# Preprocess input data for prediction
input_data = pd.DataFrame({
    'vehicle_condition': [vehicle_condition],
    'standard_make': [standard_make],
    'standard_colour': [standard_colour],
    'standard_model': [standard_model],
    'body_type': [body_type],
    'fuel_type': [fuel_type],
    'mileage': [mileage],
    'year_of_registration': [year_of_registration],
    'crossover_car_and_van': [0]  # Assuming 0 for this example; add a field if necessary
})

# Encode categorical columns
input_data['vehicle_condition'] = label_encoders['vehicle_condition'].transform(input_data['vehicle_condition'])
input_data['standard_make'] = label_encoders['standard_make'].transform(input_data['standard_make'])
input_data['standard_colour'] = label_encoders['standard_colour'].transform(input_data['standard_colour'])
input_data['standard_model'] = label_encoders['standard_model'].transform(input_data['standard_model'])
input_data['body_type'] = label_encoders['body_type'].transform(input_data['body_type'])
input_data['fuel_type'] = label_encoders['fuel_type'].transform(input_data['fuel_type'])

# Normalize numerical columns
input_data[['mileage', 'year_of_registration']] = scaler.transform(input_data[['mileage', 'year_of_registration']])

# Make prediction
predicted_price = model.predict(input_data)

# Display prediction result
st.subheader(f"Predicted Car Price: ${predicted_price[0]:.2f}")
