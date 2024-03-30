import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost
import matplotlib.pyplot as plt


# Load the original data
data = pd.read_csv('echocardiogram.csv')

# Preprocessing steps (already included in your original code)
data = data.drop(['name', 'group', 'aliveat1'], axis=1)
data = data.dropna(subset=['alive'])
discrete_features = ['pericardialeffusion']
continuous_features = data.drop(['pericardialeffusion', 'alive'], axis=1).columns

# ... (rest of preprocessing steps as in your original code)
# %%
for feature in continuous_features:
    data.boxplot(feature)
    plt.title(feature)
    plt.show()

# %%
features_with_outliers = ['wallmotion-score', 'wallmotion-index', 'mult']


# %%
for feature in continuous_features:
    if feature in features_with_outliers:
        data[feature].fillna(data[feature].median(), inplace=True)
    else:
        data[feature].fillna(data[feature].mean(), inplace=True)

# %%
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor()
outliers_rows = lof.fit_predict(data)

# %%
mask = outliers_rows != -1


# %%
data.isnull().sum()


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
data = data[mask]


# %%
from sklearn.preprocessing import StandardScaler

data1 = pd.get_dummies(data, columns = discrete_features, drop_first = True)
scaler = StandardScaler().fit(data1[continuous_features])

# Load the trained XGBoost model
model = xgboost.XGBClassifier()
model.load_model("xgboost_model.json")  # Replace "xgboost_model.bin" with the actual file name of your model

# Create the Streamlit web app
def main():
    st.title("Heart Disease Prediction")

    # Collect user input
    survival = st.number_input("Enter Survival", value=10, min_value=9, max_value=31, step=1)
    age = st.number_input("Enter Age", value=50, min_value=45, max_value=81, step=1)
    pericardialeffusion = st.selectbox("Pericardial Effusion", ["None", "Present"])
    fractionalshortening = st.number_input("Enter Fractional Shortening", value=0.15, min_value=0.0, max_value=0.5, step=0.01)
    epss = st.number_input("Enter EPSS Value", value=12.0, min_value=0.0, max_value=27.0, step=0.1)
    lvdd = st.number_input("Enter LVDD Value", value=5.0, min_value=3.0, max_value=7.0, step=0.1)
    wallmotion_score = st.slider("Enter Wallmotion Score", min_value=5, max_value=24, value=14, step=1)
    wallmotion_index = st.slider("Enter Wallmotion Index Value", min_value=1.0, max_value=3.0, value=1.0, step=0.01)
    mult = st.slider("Enter Mult Value", min_value=0.5, max_value=1.0, value=0.5, step=0.01)

    if st.button("Submit"):
        # Convert pericardial effusion to binary (0 or 1)
        pericardialeffusion = True if pericardialeffusion == "Present" else False
        features = ['survival', 'age', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score', 'wallmotion-index', 'mult' ,'pericardialeffusion']

        # Create a DataFrame from user input
        user_data = pd.DataFrame([[survival,age,fractionalshortening,epss,lvdd,wallmotion_score,wallmotion_index,mult,pericardialeffusion]], columns=features)

        # Standardize the user input data
        user_data[continuous_features] = scaler.transform(user_data[continuous_features])

        # Make prediction
        prediction = model.predict(user_data)
        
        # Display prediction
        st.subheader("Prediction")
        if prediction[0] == 1:
            st.write("The model predicts that the patient will stay alive within one year.")
        else:
            st.write(user_data)
            st.write("The model predicts that the patient will mot survive within one year.")

if __name__ == "__main__":
    main()