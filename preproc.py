import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

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
scaler = StandardScaler()
data1[continuous_features] = scaler.fit_transform(data1[continuous_features])
# Split data into features and target
X = data1.drop(['alive'], axis=1)
y = data1['alive']

# Choose your preferred model (e.g., XGBoost as you used earlier)
model = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
model.fit(X, y)

# Save the preprocessed data
data1.to_csv('processed_echocardiogram.csv', index=False)

# Save the trained model
import pickle
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)