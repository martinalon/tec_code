# test_prediction.py
import os
import joblib
import pytest
from train.use_model import prediccion  # Replace 'your_module_name' with the actual module name




@pytest.fixture
def sample_data():
    return {
        "pclass": 1,
        "name": "Newell, Mr. Arthur Webster",
        "sex": "male",
        "age": 58,
        "sibsp": 0,
        "parch": 2,
        "ticket": 35273,
        "fare": 113.275,
        "cabin": "D48",
        "embarked": "C",
        "boat": "?",
        "body": 122
    }

def test_prediccion_with_valid_data(sample_data):
    predictions, model_name = prediccion(sample_data)
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]  # Assuming the model predicts 0 or 1 for binary classification
    assert model_name in ["LogisticRegression", "RandomForest"]
# Add more test cases as needed
