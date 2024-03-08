import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost

# Load the trained XGBoost model
model = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
model.load_model("xgboost_model.bin")  # Replace "xgboost_model.bin" with the actual file name of your model

# Define the features required for prediction
features = ['survival', 'age', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score', 'wallmotion-index', 'mult']

# Create the Streamlit web app
def main():
    st.title("Heart Disease Prediction")

    # Collect user input
    survival = st.number_input("Enter Survival", min_value=9, max_value=31, value=20)
    age = st.slider("Enter Age", min_value=45, max_value=81, value=50)
    fractional_shortening = st.slider("Enter Fractional Shortening", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    epss = st.slider("Enter EPSS Value", min_value=0.0, max_value=27.0, value=12.0, step=0.1)
    lvdd = st.slider("Enter LVDD Value", min_value=3.0, max_value=7.0, value=5.0, step=0.1)
    wallmotion_score = st.slider("Enter Wallmotion Score", min_value=5, max_value=24, value=14)
    wallmotion_index = st.slider("Enter Wallmotion Index Value", min_value=1.0, max_value=1.5, value=1.0, step=0.1)
    mult = st.slider("Enter Mult Value", min_value=0.5, max_value=1.0, value=0.75, step=0.01)

    if st.button("Submit"):
        # Create a DataFrame from user input
        user_data = pd.DataFrame([[survival, age, fractional_shortening, epss, lvdd, wallmotion_score, wallmotion_index, mult]], columns=features)

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
