import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model, scaler, and encoders
model = joblib.load('car_price_predictor.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
ordinal_encoder = joblib.load('ordinal_encoders.pkl')
standardize = joblib.load('price_standardize.pkl')

# Function to encode and scale the input data
def preprocess_input(features):
    # Convert input features into a DataFrame
    features = pd.DataFrame([features])

    # Apply the ordinal encoding for 'vehicle_condition'
    features['vehicle_condition'] = ordinal_encoder.transform([[features['vehicle_condition'][0]]])[0]

    # Apply label encoding for categorical variables
    for col in ['standard_make', 'standard_colour', 'standard_model', 'body_type', 'fuel_type', 'reg_code', 'crossover_car_and_van']:
        features[col] = label_encoders[col].transform([features[col][0]])[0]

    # Apply scaling for 'mileage' and 'year_of_registration'
    features[['mileage', 'year_of_registration']] = scaler.transform(features[['mileage', 'year_of_registration']])

    # Reorder the features based on the model's expected input order
    feature_columns = ['mileage', 'reg_code', 'standard_colour', 'standard_make',
                       'standard_model', 'vehicle_condition', 'year_of_registration',
                       'body_type', 'crossover_car_and_van', 'fuel_type']
    
    # Ensure the input features are in the correct order
    features = features[feature_columns]
    
    # Debug: print the processed features
    st.write("Processed Features:", features)

    return features



# Streamlit app
def app():
    st.title('Car Price Prediction')

    # Input fields for features
    vehicle_condition = st.selectbox('Vehicle Condition', ['USED', 'NEW'])
    standard_make = st.text_input('Car Make','Toyota')
    standard_colour = st.text_input('Car Colour','White')
    standard_model = st.text_input('Car Model','Yaris')
    body_type = st.text_input('Body Type','Hatchback')
    fuel_type = st.text_input('Fuel Type','Petrol')
    reg_code = st.text_input('Registration Code','19')
    crossover_car_and_van = st.selectbox('Crossover Car and Van', ['False', 'True'])
    mileage = st.number_input('Mileage', min_value=0)
    year_of_registration = st.number_input('Year of Registration', min_value=1900, max_value=2024)

    # Create input data dictionary
    input_data = {
        'mileage': mileage,
        'reg_code': reg_code,
        'standard_colour': standard_colour,
        'standard_make': standard_make,
        'standard_model': standard_model,
        'vehicle_condition': vehicle_condition,
        'year_of_registration': year_of_registration,
        'body_type': body_type,
        'crossover_car_and_van': crossover_car_and_van,
        'fuel_type': fuel_type
    }

    # Process input data
    processed_input = preprocess_input(input_data)

    # Predict the price
    predicted_price_scaled = model.predict(processed_input)[0]

    predicted_price = standardize.inverse_transform([[predicted_price_scaled]])[0][0]

    # Display the predicted price
    st.write(f'Predicted Price: ${predicted_price:.2f}')

if __name__ == '__main__':
    app()
