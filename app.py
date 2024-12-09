import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

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
    for col in ['standard_make', 'standard_colour', 'standard_model', 'body_type', 'fuel_type']:
        features[col] = label_encoders[col].transform([features[col][0]])[0]

    # Apply scaling for 'mileage' and 'year_of_registration'
    features[['mileage', 'year_of_registration']] = scaler.transform(features[['mileage', 'year_of_registration']])

    # Reorder the features based on the model's expected input order
    feature_columns = ['mileage', 'standard_colour', 'standard_make',
                       'standard_model', 'vehicle_condition', 'year_of_registration',
                       'body_type', 'fuel_type']
    
    return features

# Streamlit app
def app():
    st.title('Car Price Prediction')

    # Add introductory text
    st.markdown("""
    ## Welcome to the Car Price Prediction App!
                
    This app predicts the price of a car based on various features such as the make, model, mileage, and more.
    Please fill in the details below to get the predicted price.
                
    Please note that, the prediction has an accuracy of 81%, therefore the price shown might not be the actual price of the car.
    
    ### How does the model work?
    The model predicts the price of a car based on features such as:
    - Vehicle condition (new or used)
    - Car make, model, color, and body type
    - Fuel type and mileage
    - Year of registration

    If you have any questions, feel free to reach out!
    """)

    # Load the make-model mapping
    with open('make_model_mapping.json', 'r') as json_file:
        make_model_mapping = json.load(json_file)

    # Dropdown for Car Make and Model
    selected_make = st.selectbox('Select Car Make', list(make_model_mapping.keys()))
    selected_model = st.selectbox('Select Car Model', make_model_mapping[selected_make] if selected_make else [])

    # Other input fields
    mileage = st.number_input('Mileage', min_value=0)
    year_of_registration = st.number_input('Year of Registration', min_value=1900, max_value=2024)
    standard_colour = st.selectbox('Car Colour', ['Grey', 'Blue', 'Brown', 'Red', 'Bronze', 'Black', 'White','Silver', 'Purple', 'Green', 'Orange', 
                                    'Yellow', 'Turquoise','Gold', 'Multicolour', 'Beige', 'Burgundy', 'Pink', 'Maroon', 'Magenta', 'Navy', 'Indigo'])
    body_type = st.selectbox('Body Type', ['SUV', 'Saloon', 'Hatchback', 'Convertible', 'Limousine', 'Estate', 'MPV', 'Coupe',
                              'Pickup', 'Combi Van', 'Panel Van', 'Minibus','Window Van', 'Camper', 'Car Derived Van', 'Chassis Cab'])
    vehicle_condition = st.selectbox('Vehicle Condition', ['USED', 'NEW'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol Plug-in Hybrid', 'Diesel', 'Petrol', 'Diesel Hybrid',
                              'Petrol Hybrid', 'Electric', 'Diesel Plug-in Hybrid', 'Bi Fuel','Natural Gas'])

    # Validate inputs
    if not selected_make or not selected_model:
        st.warning("Please select both a car make and model.")
        return

    # Create input data dictionary
    input_data = {
        'mileage': mileage,
        'standard_colour': standard_colour,
        'standard_make': selected_make,
        'standard_model': selected_model,
        'vehicle_condition': vehicle_condition,
        'year_of_registration': year_of_registration,
        'body_type': body_type,
        'fuel_type': fuel_type
    }

    # Process input data
    processed_input = preprocess_input(input_data)

    # Predict the price
    with st.spinner('Predicting the price...'):
        predicted_price_scaled = model.predict(processed_input)[0]
        predicted_price = standardize.inverse_transform([[predicted_price_scaled]])[0][0]

    st.write(f'### Predicted Price: ${predicted_price:.2f}')


if __name__ == '__main__':
    app()
