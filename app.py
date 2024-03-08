import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost

# Load the trained XGBoost model
model = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
model.load_model("xgboost_model.bin")  # Replace "xgboost_model.bin" with the actual file name of your model

# Define the features required for prediction
features = ['survival', 'age', 'pericardialeffusion', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score', 'wallmotion-index', 'mult']

# Create the Streamlit web app
def main():
    st.title("Heart Disease Prediction")

    # Collect user input
    survival = st.number_input("Enter Survival", value=0, min_value=9, max_value=31, step=1)
    age = st.number_input("Enter Age", value=50, min_value=45, max_value=81, step=1)
    pericardial_effusion = st.selectbox("Pericardial Effusion", ["None", "Present"])
    fractional_shortening = st.number_input("Enter Fractional Shortening", value=0.15, min_value=0.0, max_value=0.5, step=0.01)
    epss = st.number_input("Enter EPSS Value", value=12.0, min_value=0.0, max_value=27.0, step=0.1)
    lvdd = st.number_input("Enter LVDD Value", value=5.0, min_value=3.0, max_value=7.0, step=0.1)
    wallmotion_score = st.slider("Enter Wallmotion Score", min_value=5, max_value=24, value=14, step=1)
    wallmotion_index = st.slider("Enter Wallmotion Index Value", min_value=1.0, max_value=1.5, value=1.0, step=0.01)
    mult = st.slider("Enter Mult Value", min_value=0.5, max_value=1.0, value=0.5, step=0.01)

    if st.button("Submit"):
        # Convert pericardial effusion to binary (0 or 1)
        pericardial_effusion = 1 if pericardial_effusion == "Present" else 0

        # Create a DataFrame from user input
        user_data = pd.DataFrame([[survival, age, pericardial_effusion, fractional_shortening, epss, lvdd, wallmotion_score, wallmotion_index, mult]], columns=features)

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
