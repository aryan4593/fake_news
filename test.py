import requests

url = "https://fake-news-api-5oxb.onrender.com/predict"

news = {"text": "Britisher queen is dead"}

response = requests.post(url, json=news)

print(response.json())
