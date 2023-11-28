from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from train.use_model import prediccion
import logging

logging.basicConfig(filename='bitacora.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')



app = FastAPI()

# Define the data model using Pydantic
class PassengerData(BaseModel):
    pclass: int
    name: str
    sex: str
    age: int
    sibsp: int
    parch: int
    ticket: int
    fare: float
    cabin: str
    embarked: str
    boat: str
    body: int

# Endpoint to receive data via POST
@app.post("/predict_survival")
async def predict_survival(passenger_data: PassengerData):
    try:
        print(passenger_data.model_dump())
        my_model, name = prediccion(passenger_data.model_dump())
        # Here you can perform any processing or prediction with the received data
        # For now, let's just return the received data as a response
        result_dict = {
            "Received Data": int(my_model[0]),
            "Model Name":name }
        return(result_dict)
    
    except Exception as e:
        logging.error(f" error in {e}")
    
        

