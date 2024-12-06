#Import all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

#-------------------------Data Reading and Understanding------------------------------------

#Read csv file
dt = pd.read_csv('adverts.csv')

#-------------------------Cleaning Missing Data------------------------------------
"""
mileage (Numerical) = Mean/Median

reg_code (Categorical) = Mode

year_of_registration (Numerical) = Mean/Median

standard_colour (Categorical) = Mode

body_type (Categorical) = Mode

fuel_type (Categorical) = Mode
"""

#Filling it missing values for categorical data
dt['reg_code'] = dt['reg_code'].fillna(dt['reg_code'].mode()[0])
dt['standard_colour'] = dt['standard_colour'].fillna(dt['standard_colour'].mode()[0])
dt['body_type'] = dt['body_type'].fillna(dt['body_type'].mode()[0])
dt['fuel_type'] = dt['fuel_type'].fillna(dt['fuel_type'].mode()[0])

#Fill in missing data will be median
dt['mileage'] = dt['mileage'].fillna(dt['mileage'].median())
dt['year_of_registration'] = dt['year_of_registration'].fillna(dt['year_of_registration'].median())

#-------------------------Feature Selection------------------------------------
"""
Not all features affect the price of the car
*public_reference
"""

#Dropping columns and save into 'dt_select'
dt_select = dt.drop(columns=['public_reference'],inplace=False)

"""Selecting 10K data sample from the total 400k"""
dt_sampled = dt_select.sample(n=10000, random_state=23)

#-------------------------Data Encoding------------------------------------
"""
Categorical Data
*   standard_colour (label)
*   standard_make (label)
*   standard_model (label)
*   vehicle_condition (ordinal)
*   body_type (label)
*   crossover_car_and_van (label)
*   fuel_type (label)
"""

#Ordinal Encoding for vehicle_condition
condition_order = ['USED', 'NEW']
encoder	=	OrdinalEncoder(categories=[condition_order])
dt_sampled['vehicle_condition']	=	encoder.fit_transform(dt_sampled[['vehicle_condition']])


#Label Encoder for the rest of the categorical columns
label_encoders = LabelEncoder()
categorical_columns = ['standard_make','standard_colour','standard_model','body_type','fuel_type','reg_code','crossover_car_and_van']
for col in categorical_columns:
    dt_sampled[col] = label_encoders.fit_transform(dt_sampled[col])

#Normalization
normalizer = MinMaxScaler()
dt_sampled[['mileage','year_of_registration','price']] = normalizer.fit_transform(dt_sampled[['mileage','year_of_registration','price']])
dt_sampled.head()


#Spliting data into training and testing sets
X = dt_sampled.drop(columns=['price'])
y = dt_sampled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

#Random Forest Regression
model1 = RandomForestRegressor(random_state=23)
model1.fit(X_train, y_train)

joblib.dump(model, 'car_price_predictor.pkl')
joblib.dump(normalizer, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

y_predicted1 = model1.predict(X_test)
mae = mean_absolute_error(y_test, y_predicted1)
mse = mean_squared_error(y_test, y_predicted1)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
