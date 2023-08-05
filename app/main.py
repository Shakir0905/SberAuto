import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Dict

app = FastAPI()

# Получение пути к текущему файлу
current_path = os.path.dirname(os.path.realpath(__file__))

# Загрузка модели и percentage_dicts
model = joblib.load(os.path.join(current_path, 'gradient_boosting_model.joblib'))
percentage_dicts = joblib.load(os.path.join(current_path, 'percentage_dicts.joblib'))

# Создаем класс для запроса
from typing import Dict, Union

class Query(BaseModel):
    data: Dict[str, Union[float, str, int]]

@app.post("/predict")
def predict(query: Query):
    # Преобразуем входные данные в DataFrame
    data = pd.DataFrame([query.data])
    
    # Применяем percentage_dicts к входным данным
    for column, percentages in percentage_dicts.items():
        if column in data.columns:
            data[column] = data[column].map(percentages).fillna(0)
    
    # Получаем предсказание от модели
    prediction = model.predict(data)
    
    return {"client_id": query.data['client_id'], "prediction": int(prediction[0])}