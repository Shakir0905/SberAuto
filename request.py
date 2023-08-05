import requests
import json

# Указываем URL вашего FastAPI сервера
url = "http://127.0.0.1:8000/predict"

# Входные данные в формате JSON
data = {
    "data": {
        "session_id": 1775882,
        "client_id": "8475905179017584554.1631362986.1631362986",
        "visit_number": 1973450458.163136,
        "utm_source": "Very Low",
        "utm_medium": "ZpYIoDJMcFzVoPFsHGJL",
        "utm_campaign": "banner",
        "utm_adcontent": "gecBYcKZCPMcVYdSSzKP",
        "device_category": "mobile",
        "device_brand": "Apple",
        "device_screen_resolution": "414x896",
        "device_browser": "Safari",
        "geo_city": "Moscow",
        "event_action": 0,
        "visit_hour": 15,
        "part_of_day": "Afternoon",
        "day_of_week": 5,
        "is_weekend": 1,
        "visit_counts_by_source": 1,
        "preferred_utm_source": "ZpYIoDJMcFzVoPFsHGJL"
    }
}

# Отправляем POST запрос с JSON данными
response = requests.post(url, json=data)

# Получаем результаты из ответа
results = response.json()
print(results)
