# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data = pd.read_csv('echocardiogram.csv')
data.head()

# %%
data = data.drop(['name', 'group', 'aliveat1'], axis=1)
data.head()

# %%
data.isnull().sum()


# %%
features_with_null = [features for features in data.columns if data[features].isnull().sum()>0]
for feature in features_with_null:
    print(feature, ':', round(data[feature].isnull().mean(), 4), '%')

# %%
for feature in features_with_null:
    print(feature, ':', data[feature].unique())

# %%
data = data.dropna(subset=['alive'])
data['alive'].isnull().sum()

# %%
discrete_features = ['pericardialeffusion']
continuous_features = data.drop(['pericardialeffusion', 'alive'], axis=1).columns
label = ['alive']

print(continuous_features)

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

# %%
data1.head()


# %%
X = data1.drop(['alive'], axis=1)
y = data1['alive']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape

# %%
accuracy = {}


# %%
model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(accuracy_score(y_test, y_pred1))
accuracy[str(model1)] = accuracy_score(y_test, y_pred1)*100

# %%
model2 = DecisionTreeClassifier(max_depth=3)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(accuracy_score(y_test, y_pred2))
accuracy[str(model2)] = accuracy_score(y_test, y_pred2)*100

# %%
model3 = RandomForestClassifier(max_depth=6)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(accuracy_score(y_test, y_pred3))
accuracy[str(model3)] = accuracy_score(y_test, y_pred3)*100

# %%
model4 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(accuracy_score(y_test, y_pred4))
accuracy[str(model4)] = accuracy_score(y_test, y_pred4)*100

# %%
accuracy

# %%
algos = list(accuracy.keys())
accu_val = list(accuracy.values())

plt.bar(algos, accu_val, width=0.4)
plt.title('Accuracy Differences')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# %%
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost

# %%
param_combinations = {
    'learning_rate': np.arange(0.05, 0.4, 0.05),
    'max_depth': np.arange(3, 10),
    'min_child_weight': np.arange(1, 7, 2),
    'gamma': np.arange(0.0, 0.5, 0.1),
}

XGB = xgboost.XGBClassifier()
perfect_params = RandomizedSearchCV(XGB, param_distributions=param_combinations, n_iter=6, n_jobs=-1, scoring='roc_auc')

perfect_params.fit(X, y)
perfect_params.best_params_

# %%
model5 = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
score = cross_val_score(model5, X, y, cv=10)
model5.fit(X, y)
model5.save_model("xgboost_model.bin")

# %%
print(score)
print('Mean: ', score.mean())

# %%
import os

# Get the current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# List files in the current directory
files_in_directory = os.listdir(current_directory)
print("Files in current directory:", files_in_directory)


