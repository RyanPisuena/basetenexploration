import requests
import os

model_id = "zq8g90yw"
baseten_api_key = os.environ["BASETEN_API_KEY"]

resp = requests.post(
    f"https://model-{model_id}.api.baseten.co/development/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={
        "messages": [
            {"role": "user", "content": "What is Baseten?"}
        ],
    }
)

print(resp.json())