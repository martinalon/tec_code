# Titanic Pipeline
==============================

## Training
* Podemos configurar el training en src-config-config.py
* Podemos iniciar el training con `python -m train.train_model`
* Posteriormente, dirigirse a la carpeta `/models`, identificar el modelo recién entrenado

## Serving/Inference (Falta por implementar)
* Podemos iniciar el servidor para las predicciones con `uvicorn serve.predict:app --reload` .
* Podemos rquerir una prediccion mandando un `POST` (se puede usar Postman) a la url http://127.0.0.1:8000/prediction con un JSON en el Body en el siguiente formato:
```
{
    "pclass": 1,
    "name": "Allen, Miss. Elisabeth Walton",
    "sex": "female",
    "age": "29",
    "sibsp": 0,
    "parch": 0,
    "ticket": "24160",
    "fare": "211.3375",
    "cabin": "B5",
    "embarked": "S",
    "boat": "2",
    "body": "?",
    "home_dest": "St Louis, MO"
}
```
## Lista de Logs
| Path          | Description   | Severity   |
| ------------- | ------------- | ------------- |
| `titanic/serve/predict.py`  | Log para registrar para el modelo: id, version, datos de entrada, datos procesados y prediccion. Para el sistema: timepo de ejecución y memoria utilizada. | INFO |
| `titanic/serve/predict.py`  | Carga del modelo | CRITICAL |


## Opcional
* Falta agregar una página inicial con la documentación


## Posibles cambios a realizar 
1. Comentar código.
2. En el preprocesamiento, intentar reducir el número de carácteristicas con PCA.
3. mejorar los hiperáparametros al agregar un pipeline donde se utilice validación cruzada y una malla para los parémetros (en el caso de Random Forest).
4. Utilizar pycodestyle, black, flake8 y pylint para darle formato al código
5. crear diferentes clases para cada uno de los procesos en el codigo, es decir, separar la funcion train del archivo train_model.py en las siguientes clases:
	5.1. Importacion de datos: Una clase que importe los datos de una base con formato csv, xml,  etc.
	5.2. Hacer una clase par limpiar los datos.
	5.3. Crear una clase para el preprocesamiento (transformación de carácteristicas).
	5.4. Relizar una clase para entrenamiento, validación y exportación del modelo.
	5.5. Crear una clase que importe el modelo creado y prediga nuevos datos.