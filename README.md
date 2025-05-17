#🚗 Car Price Prediction App
Bento Motors presents a machine learning-powered web application that predicts the resale price of a car based on various features such as make, model, mileage, and more. Built using Streamlit, this app offers an intuitive interface for users to input car details and receive an estimated price.

📘 University Project Disclaimer
⚠️ Disclaimer:
This project was developed as part of a university assignment for academic purposes. While it demonstrates core concepts in machine learning and web deployment, it is not intended for commercial use and may contain simplifications for educational clarity.

📊 Features
Interactive Web Interface: User-friendly inputs for car features.

Machine Learning Model: Predicts car prices with approximately 81% accuracy.

Data Preprocessing: Handles missing values using median imputation.

Feature Encoding: Implements label and ordinal encoding for categorical variables.

Data Scaling: Applies standardization to numerical features like mileage and year of registration.

Model Explainability: Utilizes SHAP values to interpret model predictions.

🛠️ Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/Harith02/CarPricePredict.git
cd CarPricePredict
Create a Virtual Environment (Optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure Model and Encoder Files are Present:

car_price_predictor.pkl

scaler.pkl

label_encoders.pkl

ordinal_encoders.pkl

price_standardize.pkl

make_model_mapping.json

Note: These files should be placed in the root directory of the project.

🚀 Usage
Run the Streamlit application:

bash
Copy
Edit
streamlit run app.py
This will launch the app in your default web browser. Input the required car details, and the app will display the predicted resale price.

🧠 Model Details
Algorithm: Random Forest Regressor

Evaluation Metrics:

R² Score: 86%

MAE: 0

RMSE: 0 
