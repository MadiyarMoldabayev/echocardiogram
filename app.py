import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost

# Load the trained XGBoost model
model = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
model.load_model("xgboost_model.bin")  # Replace "xgboost_model.bin" with the actual file name of your model

# Define the features required for prediction
features = ['wallmotion-score', 'pericardialeffusion', 'epss', 'lvdd', 'mult', 'age', 'fractionalshortening', 'survival', 'alive']

# Create the Streamlit web app
def main():
    st.title("Heart Disease Prediction")

    # Collect user input
    wallmotion_score = st.number_input("Enter Wallmotion Score", value=14.0)
    pericardial_effusion = st.selectbox("Pericardial Effusion", ["None", "Present"])
    epss = st.number_input("Enter EPSS Value", value=12.0)
    lvdd = st.number_input("Enter LVDD Value", value=5.0)
    mult = st.number_input("Enter Mult Value", value=2.0)
    age = st.number_input("Enter Age", value=50)
    fractional_shortening = st.number_input("Enter Fractional Shortening", value=0.15)
    survival = st.number_input("Enter Survival", value=0)
    alive = st.number_input("Enter Alive Value", value=0)

    # Convert pericardial effusion to binary (0 or 1)
    pericardial_effusion = 1 if pericardial_effusion == "Present" else 0

    # Create a DataFrame from user input
    user_data = pd.DataFrame([[wallmotion_score, pericardial_effusion, epss, lvdd, mult, age, fractional_shortening, survival, alive]], columns=features)

    # Standardize the user input data
    scaler = StandardScaler()
    user_data_scaled = scaler.fit_transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The model predicts that the patient is alive.")
    else:
        st.write("The model predicts that the patient is not alive.")

if __name__ == "__main__":
    main()
