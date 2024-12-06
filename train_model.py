# Import all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Data Reading
dt = pd.read_csv('adverts.csv')

# Cleaning Missing Data
dt['reg_code'] = dt['reg_code'].fillna(dt['reg_code'].mode()[0])
dt['standard_colour'] = dt['standard_colour'].fillna(dt['standard_colour'].mode()[0])
dt['body_type'] = dt['body_type'].fillna(dt['body_type'].mode()[0])
dt['fuel_type'] = dt['fuel_type'].fillna(dt['fuel_type'].mode()[0])
dt['mileage'] = dt['mileage'].fillna(dt['mileage'].median())
dt['year_of_registration'] = dt['year_of_registration'].fillna(dt['year_of_registration'].median())

# Feature Selection
dt_select = dt.drop(columns=['public_reference'], inplace=False)
dt_sampled = dt_select.sample(n=10000, random_state=23)

# Data Encoding
condition_order = ['USED', 'NEW']
ordinal_encoder = OrdinalEncoder(categories=[condition_order])
dt_sampled['vehicle_condition'] = ordinal_encoder.fit_transform(dt_sampled[['vehicle_condition']])

label_encoders = {}
categorical_columns = ['standard_make', 'standard_colour', 'standard_model', 'body_type', 'fuel_type', 'reg_code', 'crossover_car_and_van']
for col in categorical_columns:
    le = LabelEncoder()
    dt_sampled[col] = le.fit_transform(dt_sampled[col])
    label_encoders[col] = le  # Store each encoder in a dictionary

# Standardization
scaler = MinMaxScaler()
dt_sampled[['mileage', 'year_of_registration']] = scaler.fit_transform(dt_sampled[['mileage', 'year_of_registration']])

standardize = MinMaxScaler()
dt_sampled[['price']]=standardize.fit_transform(dt_sampled[['price']])

# Define a function to remove outliers using IQR
def remove_outliers_iqr(dt, columns):
    for col in columns:
        Q1 = dt[col].quantile(0.25)
        Q3 = dt[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dt = dt[(dt[col] >= lower_bound) & (dt[col] <= upper_bound)]
    return dt

# Remove outliers for specified columns
dt_sampled_cleaned = remove_outliers_iqr(dt_sampled, ['mileage', 'year_of_registration', 'price'])


# Splitting Data
X = dt_sampled_cleaned.drop(columns=['price'])
y = dt_sampled_cleaned['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Model Training
model = RandomForestRegressor(random_state=23)
model.fit(X_train, y_train)

# Save Model, Scaler, and Encoders
joblib.dump(model, 'car_price_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(ordinal_encoder, 'ordinal_encoders.pkl')
joblib.dump(standardize, 'price_standardize.pkl')

# Evaluate Model
y_pred = model.predict(X_test)
scores1 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R^2 Value: {scores1}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
