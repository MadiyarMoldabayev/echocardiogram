from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
    survival: int
    age: int
    pericardialeffusion: bool
    fractionalshortening: float
    epss: float
    lvdd: float
    wallmotion_score: int
    wallmotion_index: float
    mult: float

# Preprocess continuous features
scaler = StandardScaler()

# Define prediction route
@app.post("/predict/")
async def predict(request: Request, data: HeartData):
    try:
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
            return JSONResponse(content={"prediction": "The model predicts that the patient will stay alive within one year."})
        else:
            return JSONResponse(content={"prediction": "The model predicts that the patient will not survive within one year."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
