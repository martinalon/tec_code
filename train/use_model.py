import joblib
import os 
import pandas as pd


#separador = os.path.sep
#dir_actual = os.path.dirname(os.path.abspath(__file__))
#dir_tec = separador.join(dir_actual.split(separador)[:-1])


#dir_model = dir_tec+'/models/'
# Load the model from the .sav file
#model = joblib.load(dir_model+'model.sav')

"""
data = { 
    "pclass": 1,
    "name" : "Newell, Mr. Arthur Webster",
    "sex": "male",
    "age": 58,
    "sibsp": 0,
    "parch": 2, 
    "ticket": 35273,
    "fare":113.275,
    "cabin":"D48",
    "embarked": "c",
    "boat": "?",
    "body": 122 
}
"""

#data = pd.DataFrame([data])
# Now you can use the loaded model for predictions
#predictions = model.predict(data)
#print(predictions)

def prediccion(data:dict):
    data = pd.DataFrame([data])
    
    separador = os.path.sep
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    dir_tec = separador.join(dir_actual.split(separador)[:-1])


    dir_model = dir_tec+'/models/'
    # Load the model from the .sav file
    model = joblib.load(dir_model+'model.sav')
    predictions = model.predict(data)
    return(predictions)
