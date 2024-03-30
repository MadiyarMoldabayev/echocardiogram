from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler

# Define FastAPI app
app = FastAPI()

# Load the trained XGBoost model
model = xgboost.XGBClassifier()
model.load_model("xgboost_model.json")  # Replace "xgboost_model.json" with the actual file name of your model

# Define input data model
class HeartData(BaseModel):
    survival: int = Query(..., ge=9, le=31)
    age: int = Query(..., ge=45, le=81)
    pericardialeffusion: bool
    fractionalshortening: float = Query(..., ge=0.0, le=0.5)
    epss: float = Query(..., ge=0.0, le=27.0)
    lvdd: float = Query(..., ge=3.0, le=7.0)
    wallmotion_score: int = Query(..., ge=5, le=24)
    wallmotion_index: float = Query(..., ge=1.0, le=3.0)
    mult: float = Query(..., ge=0.5, le=1.0)

# Preprocess continuous features
scaler = StandardScaler()

# Define prediction route
@app.post("/predict/")
async def predict(data: HeartData):
    # Convert pericardial effusion to binary (0 or 1)
    pericardialeffusion = 1 if data.pericardialeffusion else 0
    
    # Prepare input data
    user_data = pd.DataFrame({
        'survival': [data.survival],
        'age': [data.age],
        'fractionalshortening': [data.fractionalshortening],
        'epss': [data.epss],
        'lvdd': [data.lvdd],
        'wallmotion-score': [data.wallmotion_score],
        'wallmotion-index': [data.wallmotion_index],
        'mult': [data.mult],
        'pericardialeffusion': [pericardialeffusion]
    })

    # Standardize input data
    user_data[['survival', 'age', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score', 'wallmotion-index', 'mult']] = scaler.fit_transform(user_data[['survival', 'age', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score', 'wallmotion-index', 'mult']])
    
    # Make prediction
    prediction = model.predict(user_data)

    # Return prediction
    if prediction[0] == 1:
        return {"prediction": "The model predicts that the patient will stay alive within one year."}
    else:
        return {"prediction": "The model predicts that the patient will not survive within one year."}
