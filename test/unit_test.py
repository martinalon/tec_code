# test_app.py

import logging
import httpx
from train.use_model import prediccion

logging.basicConfig(filename='bitacora.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def test_predict_survival():
    # Start the FastAPI application (you may want to move this to a fixture)
    url = "http://127.0.0.1:8000/predict_survival"

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

    response = httpx.post(url, json=valid_payload)
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

    response = httpx.post(url, json=invalid_payload)
    assert response.status_code == 422  # 422 Unprocessable Entity (validation error)

    # Test exception handling
    # You might want to adjust this based on the specific exceptions raised in your code
    response = httpx.post(url, json={})
    assert response.status_code == 500  # 500 Internal Server Error
    assert "error" in response.text.lower()

# To run the test, execute `pytest test_app.py` in the terminal.
