import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost

# Load the trained XGBoost model
model = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
model.load_model("xgboost_model.bin")  # Replace with your model filename

# Define the features required for prediction
features = ['survival', 'wallmotion-index', 'wallmotion-score', 'pericardialeffusion',
            'epss', 'lvdd', 'mult', 'age', 'fractionalshortening']

# Function to preprocess user-entered data
def preprocess_user_data(user_data):
    # Convert 'pericardialeffusion' to binary
    user_data['pericardialeffusion'] = user_data['pericardialeffusion'].apply(
        lambda x: 1 if x == "Present" else 0)
    # You can add additional preprocessing steps here as needed
    return user_data

# Function to generate random input values within the corrected ranges
def generate_input_values():
    wallmotion_score = np.random.randint(5, 25)  # Integer between 5 and 24 (inclusive)
    pericardial_effusion = np.random.choice(["None", "Present"])
    epss = np.random.uniform(0, 27)
    lvdd = np.random.uniform(3, 7)
    mult = np.random.uniform(0.5, 1)
    age = np.random.randint(45, 82)  # Integer between 45 and 81 (inclusive)
    fractional_shortening = np.random.uniform(0, 0.5)
    survival = np.random.randint(9, 32)  # Integer between 9 and 31 (inclusive)
    wallmotion_index = np.random.uniform(1, 1.5)
    return [survival, wallmotion_index, wallmotion_score, pericardial_effusion, epss, lvdd, mult, age, fractional_shortening]

# Standardize input data
scaler = StandardScaler()

while True:
    # Generate random input values
    input_values = generate_input_values()

    # Create a DataFrame from input values
    user_df = pd.DataFrame([input_values], columns=features)

    # Preprocess user-entered data
    user_df = preprocess_user_data(user_df)

    # Standardize the input data
    user_data_scaled = scaler.fit_transform(user_df)

    # Make prediction
    prediction = model.predict(user_data_scaled)

    # Check if the prediction is that the patient will survive
    if prediction[0] == 1:
        print("Patient will survive with input values:", input_values)
        break
