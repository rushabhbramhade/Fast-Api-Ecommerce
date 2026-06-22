import joblib
import pandas as pd
from fastapi import FastAPI , HTTPException
from pydantic import BaseModel , Field

app = FastAPI()

##load the model 
model = joblib.load("house_model.joblib")
feature_names = joblib.load("house_feature_names.joblib")
# Load additional model metadata including MAE for health endpoint
model_metadata = joblib.load("house_model_metadata.joblib")
mae = model_metadata.get("mae", 0)

## input schema 
class HouseInput(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of households")
    HouseAge: float = Field(ge=0, description="Average age of houses")
    AveRooms: float = Field(gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per household")
    Population: float = Field(gt=0, description="Total population in the block")
    AveOccup: float = Field(gt=0, description="Average number of occupants per household")
    Latitude: float = Field(ge=32, le=42, description="Latitude coordinate")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude coordinate")

## routes /  --home route 
@app.get("/")
def read_root():
    return {
        "message": "California Housing Price Model",
        "status": "running",
        "endpoint": "Send POST request to /predict",
    }
## route /health --health route 
@app.get("/health")
def read_health():
    return {
        "status": "running",
        "model": "Random Forest Regressor",
        "feature_names": feature_names,
        "avg_error": mae * 100000, # Convert to dollars (dataset uses $100k units)
    }
## route /predict --predict route 
@app.post("/predict")
def predict_house_price(input: HouseInput):
    # Convert input to DataFrame with correct feature order
    input_dict = input.dict()
    input_features = [input_dict[feature] for feature in feature_names]
    x = pd.DataFrame([input_features], columns=feature_names)
    # Add prediction error handling
    try:
        input_pred = pd.DataFrame(
            {"MedInc": [input_dict["MedInc"]],
             "HouseAge": [input_dict["HouseAge"]],
             "AveRooms": [input_dict["AveRooms"]],
             "AveBedrms": [input_dict["AveBedrms"]],
             "Population": [input_dict["Population"]],
             "AveOccup": [input_dict["AveOccup"]],
             "Latitude": [input_dict["Latitude"]],
             "Longitude": [input_dict["Longitude"]]},
            index=[0]
        )
        prediction = model.predict(input_pred)[0]
        return {"predicted_price": prediction * 100000}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
