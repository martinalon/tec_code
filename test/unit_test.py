# test_app.py

from fastapi.testclient import TestClient
from fastapi import HTTPException
from train.use_model import prediccion
from main import app  # assuming your FastAPI app is defined in a file named 'main'
import logging

logging.basicConfig(filename='bitacora.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

client = TestClient(app)


def test_predict_survival():
    # Test valid input
    valid_payload = {
        "pclass": 1,
        "name": "John Doe",
        "sex": "male",
        "age": 30,
        "sibsp": 1,
        "parch": 0,
        "ticket": 12345,
        "fare": 50.0,
        "cabin": "C23",
        "embarked": "S",
        "boat": "A1",
        "body": 0
    }

    response = client.post("/predict_survival", json=valid_payload)
    assert response.status_code == 200
    assert "Received Data" in response.json()
    assert "Model Name" in response.json()

    # Test invalid input (missing required field)
    invalid_payload = {
        "pclass": 1,
        "name": "John Doe",
        "sex": "male",
        "age": 30,
        "sibsp": 1,
        "parch": 0,
        "ticket": 12345,
        "fare": 50.0,
        "cabin": "C23",
        "embarked": "S",
        "boat": "A1",
        # "body": 0  # Uncommenting this line will make the payload invalid
    }

    response = client.post("/predict_survival", json=invalid_payload)
    assert response.status_code == 422  # 422 Unprocessable Entity (validation error)

    # Test exception handling
    # You might want to adjust this based on the specific exceptions raised in your code
    response = client.post("/predict_survival", json={})
    assert response.status_code == 500  # 500 Internal Server Error
    assert "error" in response.text.lower()


