import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost  # Assuming you're using xgboost for prediction

# Load the pre-processed data (assuming it's saved as a CSV)
processed_data = pd.read_csv('processed_echocardiogram.csv')

# Load the trained model (assuming it's saved as a pickle file)
with open('heart_disease_model.pkl', 'rb') as file:
    model = xgboost.XGBClassifier()  # Assuming XGBoost model
    model.load_model(file)

# Define features and target variable based on pre-processed data
features = processed_data.drop('alive', axis=1).columns
label = ['alive']

# Function to preprocess user data
def preprocess_data(user_data):
  # Convert pericardial effusion to binary (0 or 1)
  user_data["pericardialeffusion"] = 1 if user_data["pericardialeffusion"] == "Present" else 0

  # Create a DataFrame from user input
  user_df = pd.DataFrame([user_data], columns=features)

  # Standardize user data using the scaler from pre-processing (assuming it's saved)
  # You'll need to load the scaler object from the pre-processing step
  # scaler = ... (load scaler)
  # user_df[continuous_features] = scaler.transform(user_df[continuous_features])

  return user_df

# Create the Streamlit web app
def main():
  st.title("Heart Disease Prediction")

  # Collect user input
  survival = st.number_input("Enter Survival", value=20, min_value=9, max_value=31, step=1)
  age = st.number_input("Enter Age", value=50, min_value=45, max_value=81, step=1)
  pericardial_effusion = st.selectbox("Pericardial Effusion", ["None", "Present"])
  fractional_shortening = st.number_input("Enter Fractional Shortening", value=0.15, min_value=0.0, max_value=0.5, step=0.01)
  epss = st.number_input("Enter EPSS Value", value=12.0, min_value=0.0, max_value=27.0, step=0.1)
  lvdd = st.number_input("Enter LVDD Value", value=5.0, min_value=3.0, max_value=7.0, step=0.1)
  wallmotion_score = st.slider("Enter Wallmotion Score", min_value=5, max_value=24, value=14, step=1)
  wallmotion_index = st.slider("Enter Wallmotion Index Value", min_value=1.0, max_value=1.5, value=1.0, step=0.01)
  mult = st.slider("Enter Mult Value", min_value=0.5, max_value=1.0, value=0.5, step=0.01)

  if st.button("Submit"):
    # Preprocess user data
    user_data = preprocess_data({'survival': survival, 'age': age, 'pericardialeffusion': pericardial_effusion,
                                 'fractionalshortening': fractional_shortening, 'epss': epss, 'lvdd': lvdd,
                                 'wallmotion-score': wallmotion_score, 'wallmotion-index': wallmotion_index, 'mult': mult})

    # Make prediction
    prediction = model.predict(user_data)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 1:
      st.write("The model predicts that the patient is alive.")
    else:
      st.write("The model predicts that the patient is not alive.")

if __name__ == "__main__":
  main()
