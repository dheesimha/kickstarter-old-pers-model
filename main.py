from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_pipeline
import uvicorn
# from app.model.model import __version__ as model_version

app = FastAPI()

class PredictionRequest(BaseModel):
    advert: str
    age_fund: int
    age_mile: int
    relation_score: int
    signi_event: int
    second_round: str
    num_employ: int
    top500: str
    
class PredictionOut(BaseModel):
    success: int

@app.get("/")
def home():
    return {"helath_check": "OK"}

@app.post("/predict",response_model=PredictionOut)
def predict(payload: PredictionRequest):
    success = predict_pipeline(payload.advert, payload.age_fund,payload.age_mile,payload.relation_score,
                           payload.signi_event,payload.second_round,payload.num_employ,payload.top500)
    return {"success": success}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)